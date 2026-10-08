"""Bone identification for source skeletons (FBX/BVH) and target PMX models.

Source side strategy
--------------------
1. **Dictionary matching** – names are normalized (namespace / rig prefixes and
   side tokens stripped) and matched against alias tables that cover Mixamo,
   VRoid, 3ds Max Biped, Unity Humanoid, UE Mannequin and common BVH rigs
   (mocopi, Perception Neuron, Rokoko, CMU).
2. **Hierarchy inference** – hips = LCA of both upper legs, spine chain = joints
   between hips and neck, shoulder = joint between chest and upper arm, etc.
3. **Geometric fallback** – for unknown naming schemes, limbs are inferred from
   the topology and rest-pose spatial layout.

Users can override any assignment via ``overrides`` (semantic -> joint name).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..parser.pmx_parser import PMXModel
from ..skeleton import SourceMotion

SIDES = ("L", "R")
FINGERS = ("thumb", "index", "middle", "ring", "little")
LIMB_SEMANTICS = ("shoulder", "upperArm", "lowerArm", "hand", "upperLeg", "lowerLeg", "foot", "toes")

# core (normalized, side-less) name -> semantic
_ALIASES = {
    "upperArm": ["upperarm", "arm", "uparm", "humerus", "shoulderjoint"],
    "lowerArm": ["forearm", "lowerarm", "lowarm", "elbow"],
    "hand": ["hand", "wrist"],
    "shoulder": ["shoulder", "clavicle", "collar", "collarbone"],
    "upperLeg": ["upleg", "upperleg", "thigh", "uleg", "femur"],
    "lowerLeg": ["leg", "lowerleg", "lowleg", "calf", "shin", "knee", "crus"],
    "foot": ["foot", "ankle"],
    "toes": ["toebase", "toes", "toe", "ball", "toe0", "footend"],
}
_CORE_TO_SEM = {alias: sem for sem, al in _ALIASES.items() for alias in al}

_FINGER_ALIASES = {
    "thumb": ["thumb"],
    "index": ["index", "pointer"],
    "middle": ["middle"],
    "ring": ["ring"],
    "little": ["pinky", "little", "small", "pinkie"],
}
_PREFIX_RE = re.compile(r"^(mixamorig\d*|bip\d+|j_bip|def|b|cc_base|valvebiped_bip01|rig|armature)(?=[_\s.\-]|$)", re.I)


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------
def _split_tokens(name: str) -> list[str]:
    name = name.split(":")[-1].split("|")[-1]
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)  # camelCase -> camel Case
    toks = [t for t in re.split(r"[\s_.\-]+", name) if t]
    return toks


def side_and_core(name: str) -> tuple[str | None, str, list[str]]:
    """Return (side 'L'/'R'/None, core string, token list) for a joint name."""
    raw = name.split(":")[-1].split("|")[-1]
    side = None
    # CMU style "LThumb", "RHipJoint"
    m = re.match(r"^([LR])([A-Z][a-z].*)$", raw)
    if m:
        side = m.group(1)
        raw = m.group(2)
    toks = _split_tokens(raw)
    # drop rig prefixes
    joined = "_".join(toks)
    joined2 = _PREFIX_RE.sub("", joined).strip("_")
    toks = [t for t in joined2.split("_") if t] if joined2 else toks
    out = []
    for t in toks:
        tl = t.lower()
        if tl in ("c", "center", "centre"):
            continue
        if tl in ("left", "l", "lft"):
            side = side or "L"
            continue
        if tl in ("right", "r", "rgt"):
            side = side or "R"
            continue
        if tl.startswith("left") and len(tl) > 4:
            side = side or "L"
            tl = tl[4:]
        elif tl.startswith("right") and len(tl) > 5:
            side = side or "R"
            tl = tl[5:]
        out.append(tl)
    core = "".join(out)
    return side, core, out


def detect_preset(names: list[str]) -> str:
    low = " ".join(names).lower()
    if "mixamorig" in low:
        return "Mixamo"
    if "j_bip_" in low:
        return "VRoid"
    if re.search(r"\bbip0?0?1\b", low):
        return "3ds Max Biped"
    if "torso_1" in low or "l_up_arm" in low:
        return "mocopi"
    if "lhipjoint" in low or "lowerback" in low:
        return "CMU"
    if "leftinhandindex" in low or "spine3" in low and "lefthandthumb1" in low:
        return "Perception Neuron"
    if "pelvis" in low and "spine_01" in low:
        return "UE Mannequin"
    if "leftthigh" in low and "leftshin" in low:
        return "Rokoko"
    if "leftupperleg" in low and "leftlowerleg" in low:
        return "Unity Humanoid"
    return "Generic"


# ---------------------------------------------------------------------------
# Source mapping
# ---------------------------------------------------------------------------
@dataclass
class SourceMap:
    bones: dict[str, int] = field(default_factory=dict)  # semantic -> joint idx
    spine: list[int] = field(default_factory=list)  # hips(excl) -> neck(excl)
    method: dict[str, str] = field(default_factory=dict)  # semantic -> 'name'|'hierarchy'|'geometry'|'manual'
    preset: str = "Generic"

    def get(self, sem: str) -> int:
        return self.bones.get(sem, -1)


def _key(sem: str, side: str | None = None) -> str:
    return f"{sem}.{side}" if side else sem


def map_source(sm: SourceMotion, overrides: dict[str, str] | None = None) -> SourceMap:
    res = SourceMap(preset=detect_preset(sm.names))
    depth = [sm.depth(j) for j in range(sm.num_joints)]

    def assign(key, j, how):
        if j is not None and j >= 0 and key not in res.bones:
            res.bones[key] = j
            res.method[key] = how

    # --- manual overrides first ---
    for key, nm in (overrides or {}).items():
        j = sm.find(nm)
        if j >= 0:
            res.bones[key] = j
            res.method[key] = "manual"

    # --- 1. dictionary matching ---
    cands: dict[str, list[int]] = {}
    finger_cands: dict[str, list[int]] = {}
    neck_c, head_c = [], []
    for j, nm in enumerate(sm.names):
        side, core, toks = side_and_core(nm)
        if any(w in core for w in ("twist", "roll", "share", "helper", "jiggle")):
            continue
        core_nodig = re.sub(r"\d+$", "", core)
        # fingers
        fsem = None
        for fname, als in _FINGER_ALIASES.items():
            if any(a in core for a in als):
                fsem = fname
                break
        mbip = re.match(r"^finger(\d)(\d?)$", core)
        if mbip:
            fsem = FINGERS[int(mbip.group(1))] if int(mbip.group(1)) < 5 else None
        if fsem and side:
            if "metacarpal" in core or "inhand" in core or core.endswith("base"):
                continue
            finger_cands.setdefault(_key(fsem, side), []).append(j)
            continue
        if side:
            sem = _CORE_TO_SEM.get(core) or _CORE_TO_SEM.get(core_nodig)
            if sem:
                cands.setdefault(_key(sem, side), []).append(j)
        else:
            if core_nodig == "neck":
                neck_c.append(j)
            elif core_nodig == "head":
                head_c.append(j)

    for key, js in cands.items():
        assign(key, min(js, key=lambda j: depth[j]), "name")
    if neck_c:
        assign("neck", min(neck_c, key=lambda j: depth[j]), "name")
    if head_c:
        assign("head", min(head_c, key=lambda j: depth[j]), "name")

    # --- 3. geometric fallback for limbs (done before hierarchy so hips can be derived) ---
    if not all(_key(s, sd) in res.bones for s in ("upperLeg", "upperArm") for sd in SIDES):
        _geometric_fallback(sm, res, assign)

    # --- 2. hierarchy inference ---
    for sd in SIDES:
        ul, ll, ft = (res.get(_key(s, sd)) for s in ("upperLeg", "lowerLeg", "foot"))
        if ul >= 0 and ft >= 0 and ll < 0:
            p = sm.path(ul, ft)
            if p:
                assign(_key("lowerLeg", sd), p[0], "hierarchy")
        if ft >= 0 and res.get(_key("toes", sd)) < 0:
            ch = [c for c in sm.children(ft)]
            if ch:
                assign(_key("toes", sd), ch[0], "hierarchy")
        ua, la, hd = (res.get(_key(s, sd)) for s in ("upperArm", "lowerArm", "hand"))
        if ua >= 0 and hd >= 0 and la < 0:
            p = sm.path(ua, hd)
            if p:
                assign(_key("lowerArm", sd), p[0], "hierarchy")
        if ua >= 0 and la >= 0 and hd < 0:
            ch = sm.children(la)
            if ch:
                assign(_key("hand", sd), ch[0], "hierarchy")

    ull, ulr = res.get("upperLeg.L"), res.get("upperLeg.R")
    if "hips" not in res.bones and ull >= 0 and ulr >= 0:
        assign("hips", sm.lca(ull, ulr), "hierarchy")
    hips = res.get("hips")

    ual, uar = res.get("upperArm.L"), res.get("upperArm.R")
    chest = sm.lca(ual, uar) if ual >= 0 and uar >= 0 else -1
    if res.get("neck") < 0 and chest >= 0:
        # neck = child of chest that is not on an arm path and goes upward
        arm_anc = set(sm.ancestors(ual)) | set(sm.ancestors(uar)) | {ual, uar}
        ups = [c for c in sm.children(chest) if c not in arm_anc]
        if ups:
            c = max(ups, key=lambda c: sm.rest_pos[c][1])
            assign("neck", c, "hierarchy")
    if res.get("head") < 0 and res.get("neck") >= 0:
        ch = sm.children(res.get("neck"))
        if ch:
            assign("head", ch[0], "hierarchy")

    neck = res.get("neck")
    if hips >= 0 and neck >= 0:
        res.spine = sm.path(hips, neck) or []
    elif hips >= 0 and chest >= 0:
        res.spine = (sm.path(hips, chest) or []) + [chest]

    # shoulders
    top = neck if neck >= 0 else chest
    for sd in SIDES:
        ua = res.get(_key("upperArm", sd))
        if ua >= 0 and res.get(_key("shoulder", sd)) < 0:
            anc = sm.ancestors(ua)
            spine_set = set(res.spine) | {hips, top}
            for a in anc:
                if a in spine_set:
                    break
                assign(_key("shoulder", sd), a, "hierarchy")
                break

    # fingers
    for sd in SIDES:
        hand = res.get(_key("hand", sd))
        for f in FINGERS:
            js = finger_cands.get(_key(f, sd), [])
            if hand >= 0:
                js = [j for j in js if hand in sm.ancestors(j)]
            js = sorted(js, key=lambda j: depth[j])[:3]
            for seg, j in enumerate(js, 1):
                assign(_key(f"{f}{seg}", sd), j, "name")
    return res


def _geometric_fallback(sm: SourceMotion, res: SourceMap, assign) -> None:
    """Infer hips / legs / arms / spine purely from topology and rest positions."""
    P = sm.rest_pos
    J = sm.num_joints
    children = [sm.children(j) for j in range(J)]

    def leaf_chain(j):
        """Longest chain from j down to a leaf."""
        best = [j]
        for c in children[j]:
            ch = [j] + leaf_chain(c)
            if len(ch) > len(best):
                best = ch
        return best

    def subtree_extent(j):
        stack, pts = [j], []
        while stack:
            k = stack.pop()
            pts.append(P[k])
            stack.extend(children[k])
        return np.array(pts)

    # up axis guess: largest extent of rest pose
    ext = P.max(0) - P.min(0)
    up = int(np.argmax(ext))
    lat = [a for a in range(3) if a != up]
    lat_axis = max(lat, key=lambda a: ext[a])

    hips = res.get("hips")
    if hips < 0:
        for j in sorted(range(J), key=sm.depth):
            branches = [c for c in children[j] if len(leaf_chain(c)) >= 3]
            if len(branches) >= 3:
                hips = j
                break
    if hips < 0:
        return
    assign("hips", hips, "geometry")

    branches = [c for c in children[hips] if len(leaf_chain(c)) >= 3]
    lowest = sorted(branches, key=lambda c: subtree_extent(c)[:, up].min())
    legs = lowest[:2]
    rest = [b for b in branches if b not in legs]
    legs_sorted = sorted(legs, key=lambda c: P[leaf_chain(c)[-1]][lat_axis], reverse=True)
    for sd, leg in zip(SIDES, legs_sorted):
        chain = leaf_chain(leg)
        # skip short "hip joint" offsets: thigh = segment with largest vertical drop
        drops = [P[chain[i]][up] - P[chain[i + 1]][up] for i in range(len(chain) - 1)]
        k = int(np.argmax(drops)) if drops else 0
        names = ["upperLeg", "lowerLeg", "foot", "toes"]
        for n, j in zip(names, chain[k:]):
            assign(_key(n, sd), j, "geometry")

    if not rest:
        return
    spine_root = max(rest, key=lambda c: subtree_extent(c)[:, up].max())
    # walk up until a joint with >= 3 meaningful branches (chest)
    j = spine_root
    while True:
        br = [c for c in children[j] if len(leaf_chain(c)) >= 2]
        if len(br) >= 3 or not children[j]:
            break
        j = max(children[j], key=lambda c: len(leaf_chain(c)))
    chest = j
    br = [c for c in children[chest] if len(leaf_chain(c)) >= 2]
    if len(br) < 3:
        return
    neck = max(br, key=lambda c: P[c][up] + (0 if abs(P[c][lat_axis] - P[chest][lat_axis]) < 1e-6 else -1e-3))
    arms = [b for b in br if b != neck]
    arms = sorted(arms, key=lambda c: P[leaf_chain(c)[-1]][lat_axis], reverse=True)[:2]
    assign("neck", neck, "geometry")
    nc = leaf_chain(neck)
    if len(nc) > 1:
        assign("head", nc[1], "geometry")
    for sd, arm in zip(SIDES, arms):
        chain = leaf_chain(arm)
        lens = [np.linalg.norm(P[chain[i + 1]] - P[chain[i]]) for i in range(len(chain) - 1)]
        # upper arm + forearm = the pair of consecutive segments with largest total length
        if len(lens) >= 2:
            k = int(np.argmax([lens[i] + lens[i + 1] for i in range(len(lens) - 1)]))
        else:
            k = 0
        if k > 0:
            assign(_key("shoulder", sd), chain[k - 1], "geometry")
        for n, jj in zip(["upperArm", "lowerArm", "hand"], chain[k:]):
            assign(_key(n, sd), jj, "geometry")


# ---------------------------------------------------------------------------
# Target (PMX) mapping
# ---------------------------------------------------------------------------
_JP_SIDE = {"L": "左", "R": "右"}
_PMX_LIMB = {
    "shoulder": ["肩"],
    "upperArm": ["腕"],
    "lowerArm": ["ひじ", "肘"],
    "hand": ["手首"],
    "upperLeg": ["足"],
    "lowerLeg": ["ひざ", "膝"],
    "foot": ["足首"],
    "toes": ["つま先", "足先EX"],
    "armTwist": ["腕捩", "腕捩れ"],
    "handTwist": ["手捩", "手捩れ"],
    "legIK": ["足ＩＫ", "足IK"],
    "toeIK": ["つま先ＩＫ", "つま先IK"],
}
_PMX_FINGER = {"thumb": "親指", "index": "人指", "middle": "中指", "ring": "薬指", "little": "小指"}


@dataclass
class TargetMap:
    bones: dict[str, int] = field(default_factory=dict)  # semantic -> pmx bone idx
    spine: list[int] = field(default_factory=list)  # 上半身, 上半身2, (上半身3)

    def get(self, sem: str) -> int:
        return self.bones.get(sem, -1)


def map_target(model: PMXModel) -> TargetMap:
    tm = TargetMap()

    def put(key, *names):
        i = model.find(*names)
        if i >= 0:
            tm.bones[key] = i

    put("center", "センター")
    put("root", "全ての親")
    put("groove", "グルーブ")
    put("hips", "下半身")
    put("neck", "首")
    put("head", "頭")
    for nm in ("上半身", "上半身2", "上半身3"):
        i = model.find(nm)
        if i >= 0:
            tm.spine.append(i)
    for sd in SIDES:
        jp = _JP_SIDE[sd]
        for sem, names in _PMX_LIMB.items():
            put(_key(sem, sd), *[jp + n for n in names])
        for f, jn in _PMX_FINGER.items():
            if f == "thumb":
                has0 = model.find(f"{jp}{jn}0", f"{jp}{jn}０") >= 0
                segs = [0, 1, 2] if has0 else [None, 1, 2]
            else:
                segs = [1, 2, 3]
            for k, s in enumerate(segs, 1):
                if s is None:
                    continue
                put(_key(f"{f}{k}", sd), f"{jp}{jn}{s}")
    return tm
