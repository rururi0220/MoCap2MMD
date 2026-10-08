"""Format-independent representation of a source motion (from FBX or BVH)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SourceMotion:
    """Skeleton + animation expressed entirely in world space.

    Attributes
    ----------
    names:      joint names
    parents:    parent index per joint (-1 for roots)
    rest_pos:   (J, 3)    world rest (bind) positions
    rest_rot:   (J, 3, 3) world rest orientations
    pos:        (F, J, 3) world positions per frame
    rot:        (F, J, 3, 3) world orientations per frame
    fps:        frames per second of ``pos`` / ``rot``
    """

    format: str
    names: list[str]
    parents: list[int]
    rest_pos: np.ndarray
    rest_rot: np.ndarray
    pos: np.ndarray
    rot: np.ndarray
    fps: float
    info: dict = field(default_factory=dict)

    @property
    def num_joints(self) -> int:
        return len(self.names)

    @property
    def num_frames(self) -> int:
        return int(self.pos.shape[0])

    def children(self, j: int) -> list[int]:
        return [i for i, p in enumerate(self.parents) if p == j]

    def depth(self, j: int) -> int:
        d = 0
        while self.parents[j] >= 0:
            j = self.parents[j]
            d += 1
        return d

    def ancestors(self, j: int) -> list[int]:
        """Ancestors of ``j`` from parent up to root."""
        out = []
        while self.parents[j] >= 0:
            j = self.parents[j]
            out.append(j)
        return out

    def path(self, a: int, b: int) -> list[int] | None:
        """Joints strictly between ancestor ``a`` and descendant ``b`` (top-down)."""
        chain = []
        j = self.parents[b]
        while j >= 0 and j != a:
            chain.append(j)
            j = self.parents[j]
        if j != a:
            return None
        return chain[::-1]

    def lca(self, a: int, b: int) -> int:
        anc_a = [a] + self.ancestors(a)
        set_b = set([b] + self.ancestors(b))
        for j in anc_a:
            if j in set_b:
                return j
        return -1

    def find(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            return -1
