from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

import numpy as np

Sample = Tuple[float, np.ndarray]  # t_mono, pos (B,3), quat (B,4) wxyz

class ViveTrackerDataBuffer:

    def __init__(self, max_frames: int = 500):
        self.max_frames = int(max_frames)
        self._buf: Deque[Sample] = deque()

    def append(self, t: float, pos: np.ndarray) -> None:
        self._buf.append((float(t), pos))
        while len(self._buf) > self.max_frames:
            self._buf.popleft()

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def as_list(self) -> List[Sample]:
        return list(self._buf)
    
    def latest(self) -> Sample | None:
        if not self._buf:
            return None
        return self._buf[-1]
