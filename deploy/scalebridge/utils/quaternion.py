"""Quaternion order conversion helpers for the ReactiveBFM deployment boundary.

Convention contract
-------------------
- The ReactiveBFM planner and its qpos36 data layout use **xyzw** order
  (scalar-last), matching the open-source ``reactivebfm`` data contract.
- ScaleBridge internals (MuJoCo ``qpos``, the tracking policy, and all state
  buffers) use **wxyz** order (scalar-first), matching MuJoCo.

All planner-facing interfaces accept and return **xyzw**. Conversion happens
exactly once at the interface boundary (``MotionTrackingOnlineEnv``); the
helpers below are the only place where the reorder is implemented.
"""

from __future__ import annotations

import numpy as np
import torch


def xyzw_to_wxyz_np(quat_xyzw: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> (w, x, y, z), numpy, any leading dimensions."""
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    return quat_xyzw[..., [3, 0, 1, 2]]


def wxyz_to_xyzw_np(quat_wxyz: np.ndarray) -> np.ndarray:
    """(w, x, y, z) -> (x, y, z, w), numpy, any leading dimensions."""
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    return quat_wxyz[..., [1, 2, 3, 0]]


def xyzw_to_wxyz_torch(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """(x, y, z, w) -> (w, x, y, z), torch, any leading dimensions."""
    return quat_xyzw[..., [3, 0, 1, 2]]


def wxyz_to_xyzw_torch(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) -> (x, y, z, w), torch, any leading dimensions."""
    return quat_wxyz[..., [1, 2, 3, 0]]
