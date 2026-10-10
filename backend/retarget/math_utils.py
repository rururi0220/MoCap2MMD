"""Math helpers: rotations, quaternions, slerp, swing-twist, coordinate conversion.

Conventions
-----------
* Rotation matrices are ``(..., 3, 3)`` numpy arrays acting on column vectors.
* Quaternions are ``(..., 4)`` arrays in ``(x, y, z, w)`` order (same as VMD / scipy).
* "World" arrays are stacked over frames as ``(F, J, ...)``.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

EPS = 1e-9

# Mirror Z: converts a right-handed (Y-up) coordinate system to MMD's left-handed one.
MIRROR_Z = np.diag([1.0, 1.0, -1.0])


# ---------------------------------------------------------------------------
# Basic vector helpers
# ---------------------------------------------------------------------------
def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, EPS)


def orthonormalize(m: np.ndarray) -> np.ndarray:
    """Project (possibly scaled / sheared) 3x3 matrices onto the nearest rotation."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    det = np.linalg.det(r)
    if np.ndim(det) == 0:
        if det < 0:
            u[..., :, -1] *= -1
            r = u @ vt
    else:
        bad = det < 0
        if np.any(bad):
            u[bad, :, -1] *= -1
            r = u @ vt
    return r


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation matrix that rotates direction ``a`` onto ``b``."""
    a = normalize(np.asarray(a, dtype=np.float64))
    b = normalize(np.asarray(b, dtype=np.float64))
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate around any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return Rotation.from_rotvec(normalize(axis) * np.pi).as_matrix()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


# ---------------------------------------------------------------------------
# Quaternion helpers (x, y, z, w)
# ---------------------------------------------------------------------------
def mat_to_quat(m: np.ndarray) -> np.ndarray:
    shape = m.shape[:-2]
    q = Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat()
    return q.reshape(*shape, 4)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    shape = q.shape[:-1]
    m = Rotation.from_quat(q.reshape(-1, 4)).as_matrix()
    return m.reshape(*shape, 3, 3)


def quat_make_continuous(q: np.ndarray) -> np.ndarray:
    """Flip signs along axis 0 (time) so consecutive quaternions stay in the same hemisphere."""
    q = q.copy()
    for i in range(1, q.shape[0]):
        d = np.sum(q[i] * q[i - 1], axis=-1)
        q[i][d < 0] *= -1
    return q


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t) -> np.ndarray:
    """Vectorized slerp. ``t`` broadcasts against the leading dims of q0/q1."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)[..., None]
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d)
    d = np.clip(d, -1.0, 1.0)
    theta = np.arccos(d)
    sin_t = np.sin(theta)
    small = sin_t < 1e-6
    safe = np.where(small, 1.0, sin_t)
    w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    w1 = np.where(small, t, np.sin(t * theta) / safe)
    return normalize(w0 * q0 + w1 * q1)


def mat_slerp(m0: np.ndarray, m1: np.ndarray, t) -> np.ndarray:
    return quat_to_mat(quat_slerp(mat_to_quat(m0), mat_to_quat(m1), t))


# ---------------------------------------------------------------------------
# Swing-twist decomposition
# ---------------------------------------------------------------------------
def swing_twist(q: np.ndarray, axis: np.ndarray):
    """Decompose ``q = swing * twist`` where ``twist`` rotates around ``axis``.

    ``q`` is ``(..., 4)`` (x, y, z, w); ``axis`` is a 3-vector expressed in the
    same frame as the rotation (i.e. the local frame before applying ``q``).
    Returns ``(swing, twist)`` quaternions.
    """
    axis = normalize(np.asarray(axis, dtype=np.float64))
    v = q[..., :3]
    proj = np.sum(v * axis, axis=-1, keepdims=True) * axis
    twist = np.concatenate([proj, q[..., 3:4]], axis=-1)
    n = np.linalg.norm(twist, axis=-1, keepdims=True)
    identity = np.zeros_like(twist)
    identity[..., 3] = 1.0
    twist = np.where(n < 1e-9, identity, twist / np.maximum(n, 1e-12))
    # swing = q * twist^-1
    tw_inv = twist * np.array([-1.0, -1.0, -1.0, 1.0])
    swing = quat_mul(q, tw_inv)
    return swing, twist


def twist_swing(q: np.ndarray, axis: np.ndarray):
    """Decompose ``q = twist * swing`` where ``twist`` rotates around ``axis``.

    Returns ``(twist, swing)`` quaternions.
    """
    axis = normalize(np.asarray(axis, dtype=np.float64))
    v = q[..., :3]
    proj = np.sum(v * axis, axis=-1, keepdims=True) * axis
    twist = np.concatenate([proj, q[..., 3:4]], axis=-1)
    n = np.linalg.norm(twist, axis=-1, keepdims=True)
    identity = np.zeros_like(twist)
    identity[..., 3] = 1.0
    twist = np.where(n < 1e-9, identity, twist / np.maximum(n, 1e-12))
    # swing = twist^-1 * q
    tw_inv = twist * np.array([-1.0, -1.0, -1.0, 1.0])
    swing = quat_mul(tw_inv, q)
    return twist, swing



def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Time resampling
# ---------------------------------------------------------------------------
def resample_globals(rot: np.ndarray, pos: np.ndarray, src_fps: float, dst_fps: float = 30.0):
    """Resample world rotations ``(F,J,3,3)`` and positions ``(F,J,3)`` to ``dst_fps``."""
    n_src = rot.shape[0]
    if n_src <= 1 or abs(src_fps - dst_fps) < 1e-6:
        return rot, pos
    duration = (n_src - 1) / src_fps
    n_dst = int(np.floor(duration * dst_fps + 1e-6)) + 1
    t_src = np.arange(n_dst) / dst_fps * src_fps
    i0 = np.clip(np.floor(t_src).astype(int), 0, n_src - 1)
    i1 = np.clip(i0 + 1, 0, n_src - 1)
    w = (t_src - i0)[:, None]  # (F,1)

    q = quat_make_continuous(mat_to_quat(rot))
    q_out = quat_slerp(q[i0], q[i1], np.broadcast_to(w, q[i0].shape[:-1]))
    pos_out = pos[i0] * (1 - w[..., None]) + pos[i1] * w[..., None]
    return quat_to_mat(q_out), pos_out


# ---------------------------------------------------------------------------
# Trajectory smoothing
# ---------------------------------------------------------------------------
def smooth_quaternions(q: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Smooth quaternion trajectory along time axis using Gaussian filter to remove mocap jitter."""
    if q.shape[0] < 3 or sigma <= 0.0:
        return q
    from scipy.ndimage import gaussian_filter1d

    q_cont = quat_make_continuous(q)
    q_smooth = gaussian_filter1d(q_cont, sigma=sigma, axis=0, mode="nearest")
    norm = np.linalg.norm(q_smooth, axis=-1, keepdims=True)
    return np.where(norm < 1e-9, q_cont, q_smooth / np.maximum(norm, 1e-12))


def smooth_positions(pos: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Smooth position trajectory along time axis using Gaussian filter."""
    if pos.shape[0] < 3 or sigma <= 0.0:
        return pos
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(pos, sigma=sigma, axis=0, mode="nearest")

