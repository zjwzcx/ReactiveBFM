from __future__ import annotations

import bisect
from typing import Deque, List, Tuple

import numpy as np

Sample = Tuple[float, np.ndarray, np.ndarray]  # t_mono, pos (B,3), quat (B,4) wxyz

def _batch_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Batch slerp for (B,4) wxyz quaternions at a single interpolation factor u."""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)  # (B,1)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)

    # small-angle linear fallback
    close = (dot > 0.9995)  # (B,1)

    # slerp branch
    dot_clamped = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot_clamped)          # (B,1)
    sin_theta = np.sin(theta)               # (B,1)
    # avoid /0 for near-identical quats (handled by close mask)
    sin_theta = np.where(sin_theta < 1e-12, 1.0, sin_theta)
    a = np.sin((1.0 - u) * theta) / sin_theta
    b = np.sin(u * theta) / sin_theta

    q_slerp = a * q0 + b * q1

    # lerp branch for near-identical
    q_lerp = (1.0 - u) * q0 + u * q1
    norm = np.linalg.norm(q_lerp, axis=-1, keepdims=True).clip(min=1e-12)
    q_lerp = q_lerp / norm

    result = np.where(close, q_lerp, q_slerp)
    # re-normalise for numerical safety
    result = result / np.linalg.norm(result, axis=-1, keepdims=True).clip(min=1e-12)
    return result.astype(np.float32)

def interp_pose_at_time(samples: List[Sample], t: float) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("empty samples")
    if t <= samples[0][0]:
        return samples[0][1].copy(), samples[0][2].copy()
    if t >= samples[-1][0]:
        return samples[-1][1].copy(), samples[-1][2].copy()

    # Binary search on timestamps
    timestamps = [s[0] for s in samples]
    idx = bisect.bisect_right(timestamps, t) - 1
    idx = max(0, min(idx, len(samples) - 2))

    t0, p0, q0 = samples[idx]
    t1, p1, q1 = samples[idx + 1]
    span = t1 - t0
    u = 0.0 if span < 1e-12 else float((t - t0) / span)

    pos = ((1.0 - u) * p0 + u * p1).astype(np.float32)
    quat = _batch_slerp_wxyz(q0, q1, u)
    return pos, quat

class DataBuffer:

    def __init__(self, max_frames: int = 5000):
        self.max_frames = int(max_frames)
        self._size = 0          # current number of valid frames
        self._head = 0          # next write position (ring)
        self._ts: np.ndarray | None = None      # (max_frames,) float64
        self._pos: np.ndarray | None = None     # (max_frames, B, 3) float32
        self._quat: np.ndarray | None = None    # (max_frames, B, 4) float32
        self._B: int = 0

    def _init_arrays(self, B: int) -> None:
        self._B = B
        self._ts = np.empty(self.max_frames, dtype=np.float64)
        self._pos = np.empty((self.max_frames, B, 3), dtype=np.float32)
        self._quat = np.empty((self.max_frames, B, 4), dtype=np.float32)

    def append(self, t: float, pos: np.ndarray, quat: np.ndarray) -> None:
        if self._ts is None:
            self._init_arrays(pos.shape[0])
        idx = self._head
        self._ts[idx] = t
        self._pos[idx] = pos
        self._quat[idx] = quat
        self._head = (self._head + 1) % self.max_frames
        if self._size < self.max_frames:
            self._size += 1

    def clear(self) -> None:
        self._size = 0
        self._head = 0

    def __len__(self) -> int:
        return self._size
    
    def _ordered_slice(self, tail_n: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self._size == 0:
            raise ValueError("empty buffer")

        if self._size < self.max_frames:
            if tail_n > 0 and tail_n < self._size:
                start = self._size - tail_n
                return self._ts[start:self._size], self._pos[start:self._size], self._quat[start:self._size]
            return self._ts[:self._size], self._pos[:self._size], self._quat[:self._size]

        if tail_n > 0 and tail_n < self._size:
            start = (self._head - tail_n) % self.max_frames
            if start < self._head:
                return self._ts[start:self._head], self._pos[start:self._head], self._quat[start:self._head]
            else:
                order = np.concatenate([np.arange(start, self.max_frames), np.arange(0, self._head)])
                return self._ts[order], self._pos[order], self._quat[order]

        order = np.concatenate([
            np.arange(self._head, self.max_frames),
            np.arange(0, self._head),
        ])
        return self._ts[order], self._pos[order], self._quat[order]

    def as_list(self) -> List[Sample]:
        ts, pos, quat = self._ordered_slice()
        return [(float(ts[i]), pos[i], quat[i]) for i in range(len(ts))]
    
    def latest(self) -> Sample | None:
        if self._size == 0:
            return None
        idx = (self._head - 1) % self.max_frames
        return (float(self._ts[idx]), self._pos[idx].copy(), self._quat[idx].copy())

    def interpolate(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        ts, pos_arr, quat_arr = self._ordered_slice()
        N = len(ts)

        if t <= ts[0]:
            return pos_arr[0].copy(), quat_arr[0].copy()
        if t >= ts[N - 1]:
            return pos_arr[N - 1].copy(), quat_arr[N - 1].copy()

        idx = int(np.searchsorted(ts, t, side='right')) - 1
        idx = max(0, min(idx, N - 2))
        span = ts[idx + 1] - ts[idx]
        u = 0.0 if span < 1e-12 else float((t - ts[idx]) / span)

        p = ((1.0 - u) * pos_arr[idx] + u * pos_arr[idx + 1]).astype(np.float32)
        q = _batch_slerp_wxyz(quat_arr[idx], quat_arr[idx + 1], u)
        return p, q

    def interpolate_batch(self, times: np.ndarray, recent_window: int = 200) -> tuple[np.ndarray, np.ndarray]:

        ts, pos_arr, quat_arr = self._ordered_slice(tail_n=recent_window)
        N = len(ts)
        K = len(times)

        indices = np.searchsorted(ts, times, side='right').astype(np.int64) - 1
        indices = np.clip(indices, 0, N - 2)

        B = pos_arr.shape[1]
        pos_out = np.empty((K, B, 3), dtype=np.float32)
        quat_out = np.empty((K, B, 4), dtype=np.float32)

        for k in range(K):
            i = int(indices[k])
            span = ts[i + 1] - ts[i]
            u = 0.0 if span < 1e-12 else float((times[k] - ts[i]) / span)
            pos_out[k] = (1.0 - u) * pos_arr[i] + u * pos_arr[i + 1]
            quat_out[k] = _batch_slerp_wxyz(quat_arr[i], quat_arr[i + 1], u)

        return pos_out, quat_out
