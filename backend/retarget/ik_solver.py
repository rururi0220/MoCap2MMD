"""Target-side forward kinematics and foot / toe IK target back-calculation.

MMD bones have world-aligned rest frames, so a bone's VMD rotation is its local
rotation relative to its parent, and its VMD translation is an offset (in the
parent's rotated frame) from the rest position.
"""
from __future__ import annotations

import numpy as np

from ..parser.pmx_parser import PMXModel


def forward_kinematics(model: PMXModel, order: list[int], local_rot: dict[int, np.ndarray],
                       local_trans: dict[int, np.ndarray], n_frames: int,
                       skip: set[int] | None = None):
    """Compute world rotations / positions for every bone.

    ``local_rot[b]`` is ``(F,3,3)``; ``local_trans[b]`` is ``(F,3)``. Missing entries
    are identity / zero. Bones in ``skip`` are left as ``None``.
    Returns ``(G, P)`` dicts of ``(F,3,3)`` / ``(F,3)`` arrays.
    """
    rest = model.positions
    eye = np.broadcast_to(np.eye(3), (n_frames, 3, 3))
    G: dict[int, np.ndarray] = {}
    P: dict[int, np.ndarray] = {}
    for b in order:
        if skip and b in skip:
            continue
        p = model.bones[b].parent
        L = local_rot.get(b, eye)
        T = local_trans.get(b)
        if p < 0 or p not in G:
            G[b] = L
            P[b] = rest[b] + (T if T is not None else 0.0)
            P[b] = np.broadcast_to(P[b], (n_frames, 3)).copy()
        else:
            off = rest[b] - rest[p]
            if T is not None:
                off = off + T
            G[b] = G[p] @ L
            P[b] = P[p] + np.einsum("fij,...j->fi", G[p], off)
    return G, P


def solve_ik_offsets(model: PMXModel, ik_bones: list[int], G: dict[int, np.ndarray],
                     P: dict[int, np.ndarray], n_frames: int) -> dict[int, np.ndarray]:
    """For each IK bone, find the VMD translation that places it on the FK world
    position of its IK target bone. IK bones must be ordered parents-first
    (e.g. 足IK before つま先IK) and keep identity rotation.
    """
    rest = model.positions
    eye = np.broadcast_to(np.eye(3), (n_frames, 3, 3))
    out: dict[int, np.ndarray] = {}
    for b in ik_bones:
        bone = model.bones[b]
        tgt = bone.ik_target
        if tgt < 0 or tgt not in P:
            continue
        desired = P[tgt]
        p = bone.parent
        if p >= 0 and p in G:
            Gp, Pp, base = G[p], P[p], rest[b] - rest[p]
        else:
            Gp, Pp, base = eye, np.zeros((n_frames, 3)), rest[b]
        # desired = Pp + Gp @ (base + T)  ->  T = Gp^T (desired - Pp) - base
        T = np.einsum("fji,fj->fi", Gp, desired - Pp) - base
        out[b] = T
        # update this IK bone's world transform for its IK children (e.g. つま先IK)
        G[b] = Gp.copy()
        P[b] = desired.copy()
    return out
