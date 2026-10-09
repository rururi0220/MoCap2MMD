"""FBX parser built on ``ufbx`` (MIT / Unlicense).

The scene is converted by ufbx to right-handed, Y-up, meter units, then the
skeleton is sampled at ``sample_fps`` into a :class:`SourceMotion`.
"""
from __future__ import annotations

import numpy as np

from ..retarget.math_utils import orthonormalize
from ..skeleton import SourceMotion


class FBXError(ValueError):
    pass


def _mat(m) -> tuple[np.ndarray, np.ndarray]:
    """ufbx Matrix (column-major c0..c3) -> (rotation 3x3, translation 3)."""
    r = np.array([[m.c0.x, m.c1.x, m.c2.x], [m.c0.y, m.c1.y, m.c2.y], [m.c0.z, m.c1.z, m.c2.z]], dtype=np.float64)
    t = np.array([m.c3.x, m.c3.y, m.c3.z], dtype=np.float64)
    return r, t


def _is_skeleton_node(node) -> bool:
    if node.is_root:
        return False
    if node.bone is not None:
        return True
    # Null/empty nodes are also accepted (BVH->FBX conversions often use them);
    # anything carrying geometry, cameras or lights is not.
    try:
        if node.mesh is not None or node.camera is not None or node.light is not None:
            return False
    except AttributeError:
        pass
    return True


def parse_fbx(path: str, sample_fps: float = 30.0, stack_index: int | None = None) -> SourceMotion:
    try:
        import ufbx
    except ImportError as e:  # pragma: no cover
        raise FBXError("ufbx is not installed (pip install ufbx)") from e

    scene = ufbx.load_file(
        path,
        ignore_geometry=True,
        ignore_embedded=True,
        target_unit_meters=1.0,
    )

    all_nodes = list(scene.nodes)
    cand = [n for n in all_nodes if _is_skeleton_node(n)]
    # Keep only nodes that are bones or have a bone descendant / ancestor chain to one.
    bone_ids = {n.element_id for n in cand if n.bone is not None}
    if bone_ids:
        keep = set()
        for n in cand:
            if n.element_id in bone_ids:
                p = n
                while p is not None and not p.is_root:
                    keep.add(p.element_id)
                    p = p.parent
        cand = [n for n in cand if n.element_id in keep]
    if not cand:
        raise FBXError("No skeleton nodes found in FBX")

    id_to_idx = {n.element_id: i for i, n in enumerate(cand)}
    names = [n.name for n in cand]
    parents = []
    for n in cand:
        p = n.parent
        while p is not None and p.element_id not in id_to_idx:
            p = p.parent
        parents.append(id_to_idx[p.element_id] if p is not None else -1)

    # Order parents-before-children
    order = sorted(range(len(cand)), key=lambda i: _depth(parents, i))
    remap = {old: new for new, old in enumerate(order)}
    cand = [cand[i] for i in order]
    names = [names[i] for i in order]
    parents = [remap[parents[i]] if parents[i] >= 0 else -1 for i in order]
    ids = [n.element_id for n in cand]

    J = len(cand)
    rest_rot = np.empty((J, 3, 3))
    rest_pos = np.empty((J, 3))
    for i, n in enumerate(cand):
        r, t = _mat(n.node_to_world)
        rest_rot[i] = orthonormalize(r)
        rest_pos[i] = t

    # ---------------- animation ----------------
    stacks = list(scene.anim_stacks)
    if stacks:
        if stack_index is None:
            stack = max(stacks, key=lambda s: s.time_end - s.time_begin)
        else:
            stack = stacks[stack_index]
        anim = stack.anim
        t0, t1 = stack.time_begin, stack.time_end
        stack_name = stack.name
    else:
        anim = scene.anim
        t0, t1 = 0.0, 0.0
        stack_name = ""

    n_frames = max(1, int(np.floor((t1 - t0) * sample_fps + 1e-6)) + 1)
    pos = np.empty((n_frames, J, 3))
    rot = np.empty((n_frames, J, 3, 3))
    for f in range(n_frames):
        ev = ufbx.evaluate_scene(scene, anim, t0 + f / sample_fps)
        ev_nodes = {n.element_id: n for n in ev.nodes}
        for i, eid in enumerate(ids):
            r, t = _mat(ev_nodes[eid].node_to_world)
            rot[f, i] = r
            pos[f, i] = t
    rot = orthonormalize(rot.reshape(-1, 3, 3)).reshape(n_frames, J, 3, 3)

    src_fps = float(scene.settings.frames_per_second or sample_fps)
    return SourceMotion(
        format="fbx",
        names=names,
        parents=parents,
        rest_pos=rest_pos,
        rest_rot=rest_rot,
        pos=pos,
        rot=rot,
        fps=sample_fps,
        info={
            "source_fps": src_fps,
            "anim_stack": stack_name,
            "anim_stacks": [s.name for s in stacks],
            "has_blend_shapes": bool(len(list(scene.blend_shapes))) if hasattr(scene, "blend_shapes") else False,
        },
    )


def _depth(parents: list[int], i: int) -> int:
    d = 0
    while parents[i] >= 0:
        i = parents[i]
        d += 1
    return d
