"""Minimal PMX 2.0 / 2.1 reader (bones only; geometry is skipped).

Positions are returned exactly as stored in the file, i.e. in MMD's left-handed,
Y-up coordinate system.
"""
from __future__ import annotations

import struct
import unicodedata
from dataclasses import dataclass, field

import numpy as np


class PMXError(ValueError):
    pass


# Bone flags
FLAG_TAIL_IS_BONE = 0x0001
FLAG_ROTATABLE = 0x0002
FLAG_MOVABLE = 0x0004
FLAG_VISIBLE = 0x0008
FLAG_OPERABLE = 0x0010
FLAG_IK = 0x0020
FLAG_APPEND_LOCAL = 0x0080
FLAG_APPEND_ROTATE = 0x0100
FLAG_APPEND_MOVE = 0x0200
FLAG_FIXED_AXIS = 0x0400
FLAG_LOCAL_AXIS = 0x0800
FLAG_AFTER_PHYSICS = 0x1000
FLAG_EXTERNAL_PARENT = 0x2000


@dataclass
class IKLink:
    bone: int
    has_limit: bool = False
    limit_min: tuple = (0.0, 0.0, 0.0)
    limit_max: tuple = (0.0, 0.0, 0.0)


@dataclass
class PMXBone:
    index: int
    name: str
    name_en: str
    position: np.ndarray
    parent: int
    layer: int
    flags: int
    tail_bone: int = -1
    tail_offset: np.ndarray | None = None
    append_parent: int = -1
    append_ratio: float = 0.0
    fixed_axis: np.ndarray | None = None
    ik_target: int = -1
    ik_loop: int = 0
    ik_limit: float = 0.0
    ik_links: list[IKLink] = field(default_factory=list)

    @property
    def is_ik(self) -> bool:
        return bool(self.flags & FLAG_IK)


@dataclass
class PMXModel:
    name: str
    name_en: str
    version: float
    bones: list[PMXBone]
    path: str = ""

    def __post_init__(self):
        self._by_norm = {}
        for b in self.bones:
            self._by_norm.setdefault(normalize_name(b.name), b.index)

    def find(self, *names: str) -> int:
        """Return the index of the first matching bone name (NFKC-insensitive), or -1."""
        for n in names:
            i = self._by_norm.get(normalize_name(n))
            if i is not None:
                return i
        return -1

    @property
    def positions(self) -> np.ndarray:
        return np.array([b.position for b in self.bones], dtype=np.float64)

    @property
    def parents(self) -> list[int]:
        return [b.parent for b in self.bones]

    def topo_order(self) -> list[int]:
        """Bone indices ordered parents-before-children."""
        n = len(self.bones)
        depth = [-1] * n

        def d(i, guard=0):
            if depth[i] >= 0:
                return depth[i]
            p = self.bones[i].parent
            if p < 0 or p >= n or guard > n:
                depth[i] = 0
            else:
                depth[i] = d(p, guard + 1) + 1
            return depth[i]

        for i in range(n):
            d(i)
        return sorted(range(n), key=lambda i: (depth[i], i))


def normalize_name(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()


class _Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.o = 0

    def read(self, fmt: str):
        s = struct.calcsize(fmt)
        v = struct.unpack_from("<" + fmt, self.d, self.o)
        self.o += s
        return v

    def skip(self, n: int):
        self.o += n

    def i32(self):
        return self.read("i")[0]

    def f32(self):
        return self.read("f")[0]

    def u8(self):
        return self.read("B")[0]

    def vec3(self):
        return np.array(self.read("3f"), dtype=np.float64)


def load_pmx(path: str) -> PMXModel:
    with open(path, "rb") as f:
        data = f.read()
    r = _Reader(data)
    magic = data[:4]
    if magic != b"PMX ":
        raise PMXError("Not a PMX file (magic=%r)" % magic)
    r.skip(4)
    version = r.f32()
    n_globals = r.u8()
    g = list(r.read("%dB" % n_globals))
    encoding = "utf-16-le" if g[0] == 0 else "utf-8"
    add_uv = g[1]
    vsize, tsize, msize, bsize = g[2], g[3], g[4], g[5]

    def text():
        n = r.i32()
        s = data[r.o : r.o + n].decode(encoding, errors="replace")
        r.skip(n)
        return s

    sidx = {1: "b", 2: "h", 4: "i"}

    def bidx():
        return r.read(sidx[bsize])[0]

    name = text()
    name_en = text()
    text()  # comment
    text()  # comment en

    # ---------------- vertices (skipped) ----------------
    n_vert = r.i32()
    base = 4 * (3 + 3 + 2) + 16 * add_uv
    for _ in range(n_vert):
        r.skip(base)
        wt = data[r.o]
        r.skip(1)
        if wt == 0:
            r.skip(bsize)
        elif wt == 1:
            r.skip(bsize * 2 + 4)
        elif wt == 2 or wt == 4:
            r.skip(bsize * 4 + 16)
        elif wt == 3:
            r.skip(bsize * 2 + 4 + 36)
        else:
            raise PMXError("Unknown vertex weight type %d" % wt)
        r.skip(4)  # edge scale

    # ---------------- faces ----------------
    n_idx = r.i32()
    r.skip(n_idx * vsize)

    # ---------------- textures ----------------
    for _ in range(r.i32()):
        text()

    # ---------------- materials ----------------
    for _ in range(r.i32()):
        text()
        text()
        r.skip(16 + 12 + 4 + 12 + 1 + 16 + 4)
        r.skip(tsize * 2)
        r.skip(1)  # sphere mode
        toon_flag = r.u8()
        r.skip(tsize if toon_flag == 0 else 1)
        text()  # memo
        r.skip(4)  # face count

    # ---------------- bones ----------------
    bones: list[PMXBone] = []
    for bi in range(r.i32()):
        bname = text()
        bname_en = text()
        pos = r.vec3()
        parent = bidx()
        layer = r.i32()
        flags = r.read("H")[0]
        b = PMXBone(bi, bname, bname_en, pos, parent, layer, flags)
        if flags & FLAG_TAIL_IS_BONE:
            b.tail_bone = bidx()
        else:
            b.tail_offset = r.vec3()
        if flags & (FLAG_APPEND_ROTATE | FLAG_APPEND_MOVE):
            b.append_parent = bidx()
            b.append_ratio = r.f32()
        if flags & FLAG_FIXED_AXIS:
            b.fixed_axis = r.vec3()
        if flags & FLAG_LOCAL_AXIS:
            r.skip(24)
        if flags & FLAG_EXTERNAL_PARENT:
            r.skip(4)
        if flags & FLAG_IK:
            b.ik_target = bidx()
            b.ik_loop = r.i32()
            b.ik_limit = r.f32()
            for _ in range(r.i32()):
                link = IKLink(bidx())
                if r.u8():
                    link.has_limit = True
                    link.limit_min = r.read("3f")
                    link.limit_max = r.read("3f")
                b.ik_links.append(link)
        bones.append(b)

    return PMXModel(name=name, name_en=name_en, version=version, bones=bones, path=path)
