"""BVH parser.

Supports arbitrary per-joint channel orders (e.g. ``Zrotation Xrotation Yrotation``)
and position channels on any joint. Outputs a :class:`SourceMotion` in the BVH's
native coordinate system (axes are canonicalized later by the retargeter).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..skeleton import SourceMotion


class BVHError(ValueError):
    pass


class _Joint:
    __slots__ = ("name", "parent", "offset", "channels", "ch_start")

    def __init__(self, name: str, parent: int):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.channels: list[str] = []
        self.ch_start = 0


def _tokenize(text: str):
    for line in text.splitlines():
        for tok in line.split():
            yield tok


def parse_bvh(path: str) -> SourceMotion:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    head, sep, motion_txt = text.partition("MOTION")
    if not sep:
        raise BVHError("MOTION section not found")

    # ---------------- HIERARCHY ----------------
    toks = list(_tokenize(head))
    if not toks or toks[0].upper() != "HIERARCHY":
        raise BVHError("HIERARCHY section not found")
    joints: list[_Joint] = []
    stack: list[int] = []
    i = 1
    n_ch = 0
    pending_end = False
    while i < len(toks):
        t = toks[i]
        tu = t.upper()
        if tu in ("ROOT", "JOINT"):
            name = toks[i + 1]
            joints.append(_Joint(name, stack[-1] if stack else -1))
            i += 2
        elif tu == "END":  # End Site
            pending_end = True
            i += 2
        elif t == "{":
            if pending_end:
                stack.append(-2)  # marker for End Site block
                pending_end = False
            else:
                stack.append(len(joints) - 1)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif tu == "OFFSET":
            vals = np.array([float(x) for x in toks[i + 1 : i + 4]])
            if stack and stack[-1] >= 0:
                joints[stack[-1]].offset = vals
            i += 4
        elif tu == "CHANNELS":
            cnt = int(toks[i + 1])
            j = joints[stack[-1]]
            j.channels = [c.lower() for c in toks[i + 2 : i + 2 + cnt]]
            j.ch_start = n_ch
            n_ch += cnt
            i += 2 + cnt
        else:
            i += 1
    if not joints:
        raise BVHError("No joints found")

    # ---------------- MOTION ----------------
    mtoks = motion_txt.split()
    frames = None
    frame_time = None
    k = 0
    while k < len(mtoks):
        if mtoks[k].lower().startswith("frames"):
            frames = int(mtoks[k + 1])
            k += 2
        elif mtoks[k].lower() == "frame" and mtoks[k + 1].lower().startswith("time"):
            frame_time = float(mtoks[k + 2])
            k += 3
            break
        else:
            k += 1
    if frames is None or frame_time is None:
        raise BVHError("Frames / Frame Time not found")
    data = np.array(mtoks[k : k + frames * n_ch], dtype=np.float64)
    frames = data.size // max(n_ch, 1)
    data = data[: frames * n_ch].reshape(frames, n_ch)

    J = len(joints)
    F = frames
    loc_rot = np.empty((F, J, 3, 3))
    loc_pos = np.empty((F, J, 3))
    for ji, jt in enumerate(joints):
        rot_order = ""
        rot_cols = []
        pos = np.broadcast_to(jt.offset, (F, 3)).copy()
        for ci, ch in enumerate(jt.channels):
            col = data[:, jt.ch_start + ci]
            if ch.endswith("rotation"):
                rot_order += ch[0].upper()
                rot_cols.append(col)
            elif ch.endswith("position"):
                pos[:, "xyz".index(ch[0])] = col
        if rot_order:
            # Uppercase = intrinsic: R = R_a * R_b * R_c in listed order (BVH convention)
            loc_rot[:, ji] = Rotation.from_euler(rot_order, np.stack(rot_cols, axis=1), degrees=True).as_matrix()
        else:
            loc_rot[:, ji] = np.eye(3)
        loc_pos[:, ji] = pos

    # Forward kinematics (BVH lists parents before children)
    g_rot = np.empty_like(loc_rot)
    g_pos = np.empty_like(loc_pos)
    rest_pos = np.zeros((J, 3))
    for ji, jt in enumerate(joints):
        p = jt.parent
        if p < 0:
            g_rot[:, ji] = loc_rot[:, ji]
            g_pos[:, ji] = loc_pos[:, ji]
            rest_pos[ji] = jt.offset
        else:
            g_rot[:, ji] = g_rot[:, p] @ loc_rot[:, ji]
            g_pos[:, ji] = g_pos[:, p] + np.einsum("fij,fj->fi", g_rot[:, p], loc_pos[:, ji])
            rest_pos[ji] = rest_pos[p] + jt.offset

    rest_rot = np.broadcast_to(np.eye(3), (J, 3, 3)).copy()

    return SourceMotion(
        format="bvh",
        names=[j.name for j in joints],
        parents=[j.parent for j in joints],
        rest_pos=rest_pos,
        rest_rot=rest_rot,
        pos=g_pos,
        rot=g_rot,
        fps=1.0 / frame_time if frame_time > 0 else 30.0,
        info={"channel_orders": {j.name: "".join(c[0] for c in j.channels if c.endswith("rotation")) for j in joints}},
    )
