from __future__ import annotations

# from collections import deque
import bisect
from typing import Deque, List, Tuple

import numpy as np

Sample = Tuple[float, np.ndarray, np.ndarray]  # t_mono, pos (B,3), quat (B,4) wxyz

def _batch_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:

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

class XsensDataBuffer:

    def __init__(self, max_frames: int = 5000):
        self.max_frames = int(max_frames)
        # self._buf: Deque[Sample] = deque()
        self._size = 0          # current number of valid frames
        self._head = 0          # next write position (ring)
        self._ts: np.ndarray | None = None      # (max_frames,) float64
        self._pos: np.ndarray | None = None     # (max_frames, B, 3) float32
        self._quat: np.ndarray | None = None    # (max_frames, B, 4) float32
        self._qpos: np.ndarray | None = None
        self._B: int = 0
        self._H: int = 0

    def _init_arrays(self, B: int, H: int) -> None:
        self._B = B
        self._H = H
        self._ts = np.empty(self.max_frames, dtype=np.float64)
        self._pos = np.empty((self.max_frames, B, 3), dtype=np.float32)
        self._quat = np.empty((self.max_frames, B, 4), dtype=np.float32)
        self._qpos = np.empty((self.max_frames, H), dtype=np.float32)

    def append(self, t: float, pos: np.ndarray, quat: np.ndarray, qpos: np.ndarray) -> None:
        if self._ts is None:
            self._init_arrays(pos.shape[0], qpos.shape[0])
        idx = self._head
        self._ts[idx] = t
        self._pos[idx] = pos
        self._quat[idx] = quat
        self._qpos[idx] = qpos
        self._head = (self._head + 1) % self.max_frames
        if self._size < self.max_frames:
            self._size += 1

    def clear(self) -> None:
        self._size = 0
        self._head = 0

    def __len__(self) -> int:
        return self._size
    
    def _ordered_slice(self, tail_n: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (ts, pos, quat) arrays in chronological order.
        
        Args:
            tail_n: if > 0, only return the most recent `tail_n` frames (fast path).
        """
        if self._size == 0:
            raise ValueError("empty buffer")

        if self._size < self.max_frames:
            # buffer has not wrapped yet — contiguous 0..size
            if tail_n > 0 and tail_n < self._size:
                start = self._size - tail_n
                return self._ts[start:self._size], self._pos[start:self._size], self._quat[start:self._size]
            return self._ts[:self._size], self._pos[:self._size], self._quat[:self._size]

        # wrapped ring: build the contiguous view
        if tail_n > 0 and tail_n < self._size:
            # only read the last tail_n entries from the ring
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
        return (float(self._ts[idx]), self._pos[idx].copy(), self._quat[idx].copy(), self._qpos[idx].copy())

    # ---- batch: multiple time points at once ----
    def interpolate_batch(self, times: np.ndarray, recent_window: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate at K time points simultaneously.

        Args:
            times: (K,) array of monotonic timestamps.
            recent_window: only search the most recent N frames (fast path for real-time).
        Returns:
            positions: (K, B, 3) float32
            quaternions: (K, B, 4) float32  (wxyz)
        """
        ts, pos_arr, quat_arr = self._ordered_slice(tail_n=recent_window)
        N = len(ts)

        indices = np.searchsorted(ts, times, side='right').astype(np.int64) - 1
        indices = np.clip(indices, 0, N - 2)

        t0 = ts[indices]
        t1 = ts[indices + 1]
        span = t1 - t0
        u = np.where(span < 1e-12, 0.0, (times - t0) / span)
        u_expanded = u[:, None, None]
        
        pos_out = (1.0 - u_expanded) * pos_arr[indices] + u_expanded * pos_arr[indices+1]
        quat_out = _batch_slerp_wxyz(quat_arr[indices], quat_arr[indices + 1], u_expanded) # (ts, nb, 4)

        return pos_out, quat_out