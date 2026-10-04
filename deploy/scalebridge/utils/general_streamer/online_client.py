from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
from scalebridge.utils.logging import logger

from scalebridge.utils.general_streamer.receiver import Receiver
from scalebridge.utils.general_streamer.data_buffer import DataBuffer


class OnlineClient:

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9763,
        max_buffer_frames: int = 5000,
        poll_interval: float = 0.002,
    ):
        self.receiver = Receiver(host=host, port=port)
        self.buffer = DataBuffer(max_frames=max_buffer_frames)
        self.poll_interval = float(poll_interval)
        self._running = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._last_fc = 0

    def listen(self) -> None:
        self.receiver.listen()

    def accept_blocking(self) -> tuple[str, int]:
        return self.receiver.accept_blocking()

    def start(self) -> None:
        self.receiver.start_receiving()
        self._running.set()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="OnlinePoll")
        self._poll_thread.start()
        time.sleep(0.5)

    def stop(self) -> None:
        self._running.clear()
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        self._poll_thread = None
        self.receiver.stop_receiving()
        self.receiver.close()

    def wait_first_frame(self, timeout_s: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.receiver.frame_count >= 1:
                return True
            time.sleep(0.02)
        return False

    def _poll_loop(self) -> None:
        while self._running.is_set():
            fc = int(self.receiver.frame_count)
            if fc < 1 or fc == self._last_fc:
                time.sleep(self.poll_interval)
                continue
            self._last_fc = fc
            pos = self.receiver.pos.copy()
            ori = self.receiver.ori.copy()
            t_mono = time.monotonic()
            try:
                self.buffer.append(t_mono, pos, ori)
            except Exception as e:
                logger.debug(f"Xsens body process skip: {e}")
