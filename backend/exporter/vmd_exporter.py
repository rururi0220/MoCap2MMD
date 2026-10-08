"""VMD (Vocaloid Motion Data) binary writer / reader."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

HEADER = b"Vocaloid Motion Data 0002"

# Linear interpolation curve (x1=y1=20, x2=y2=107) for X, Y, Z, Rotation.
_ROW = bytes([20, 20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107])
LINEAR_INTERP = bytes(
    b for r in range(4) for b in (_ROW[r:] + bytes(r))
)
assert len(LINEAR_INTERP) == 64


def encode_name(name: str, size: int) -> bytes:
    """Shift-JIS encode and null-pad/truncate to ``size`` bytes (never splitting a char)."""
    raw = name.encode("cp932", errors="replace")
    if len(raw) > size:
        out = b""
        for ch in name:
            c = ch.encode("cp932", errors="replace")
            if len(out) + len(c) > size:
                break
            out += c
        raw = out
    return raw + b"\x00" * (size - len(raw))


def decode_name(raw: bytes) -> str:
    raw = raw.split(b"\x00", 1)[0]
    return raw.decode("cp932", errors="replace")


@dataclass
class BoneTrack:
    name: str
    frames: np.ndarray  # (N,) int
    positions: np.ndarray  # (N, 3)
    rotations: np.ndarray  # (N, 4) x,y,z,w


@dataclass
class VMDMotion:
    model_name: str
    bones: list[BoneTrack] = field(default_factory=list)
    morphs: list[tuple[str, int, float]] = field(default_factory=list)
    ik_states: list[tuple[int, list[tuple[str, bool]]]] = field(default_factory=list)

    @property
    def num_bone_keys(self) -> int:
        return int(sum(len(t.frames) for t in self.bones))


def write_vmd(path: str, motion: VMDMotion) -> None:
    parts = [HEADER + b"\x00" * (30 - len(HEADER)), encode_name(motion.model_name, 20)]

    # Bone keyframes
    parts.append(struct.pack("<I", motion.num_bone_keys))
    rec = struct.Struct("<15sI3f4f64s")
    for tr in motion.bones:
        nm = encode_name(tr.name, 15)
        for fr, p, q in zip(tr.frames, tr.positions, tr.rotations):
            parts.append(rec.pack(nm, int(fr), float(p[0]), float(p[1]), float(p[2]),
                                  float(q[0]), float(q[1]), float(q[2]), float(q[3]), LINEAR_INTERP))

    # Morph keyframes
    parts.append(struct.pack("<I", len(motion.morphs)))
    for name, fr, w in motion.morphs:
        parts.append(struct.pack("<15sIf", encode_name(name, 15), int(fr), float(w)))

    parts.append(struct.pack("<I", 0))  # camera
    parts.append(struct.pack("<I", 0))  # light
    parts.append(struct.pack("<I", 0))  # self shadow

    # Show / IK keyframes
    parts.append(struct.pack("<I", len(motion.ik_states)))
    for fr, states in motion.ik_states:
        parts.append(struct.pack("<IBI", int(fr), 1, len(states)))
        for nm, on in states:
            parts.append(struct.pack("<20sB", encode_name(nm, 20), 1 if on else 0))

    with open(path, "wb") as f:
        f.write(b"".join(parts))


def read_vmd(path: str) -> VMDMotion:
    """Read bone keyframes back (used for tests / preview)."""
    with open(path, "rb") as f:
        d = f.read()
    if not d.startswith(b"Vocaloid Motion Data"):
        raise ValueError("Not a VMD file")
    model = decode_name(d[30:50])
    o = 50
    (n,) = struct.unpack_from("<I", d, o)
    o += 4
    rec = struct.Struct("<15sI3f4f64s")
    tracks: dict[str, list] = {}
    for _ in range(n):
        nm, fr, px, py, pz, qx, qy, qz, qw, _ip = rec.unpack_from(d, o)
        o += rec.size
        tracks.setdefault(decode_name(nm), []).append((fr, (px, py, pz), (qx, qy, qz, qw)))
    m = VMDMotion(model)
    for nm, keys in tracks.items():
        keys.sort(key=lambda k: k[0])
        m.bones.append(BoneTrack(nm, np.array([k[0] for k in keys]),
                                 np.array([k[1] for k in keys]), np.array([k[2] for k in keys])))
    return m
