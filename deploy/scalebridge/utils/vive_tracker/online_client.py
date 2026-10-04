from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
from scalebridge.utils.logging import logger

from scalebridge.utils.vive_tracker.receiver import ViveTrackerReceiver
from scalebridge.utils.vive_tracker.processor import ViveTrackerProcessor
from scalebridge.utils.vive_tracker.data_buffer import ViveTrackerDataBuffer

class ViveTrackerOnlineClient:

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        max_buffer_frames: int = 500, # 1s
        poll_interval: float = 0.002,
    ):
        self.receiver = ViveTrackerReceiver(host=host, port=port)
        self.processor = ViveTrackerProcessor()
        self.buffer = ViveTrackerDataBuffer(max_frames=max_buffer_frames)
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
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="ViveTrackerOnlinePoll")
        self._poll_thread.start()

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
            data = self.receiver.data.copy()
            t_mono = time.monotonic()
            try:
                processed_data = self.processor.process(data)
                self.buffer.append(t_mono, processed_data)
            except Exception as e:
                logger.debug(f"Xsens body process skip: {e}")

    def calibrate(self, robot_root_quat):
        self.processor.calibrate(robot_root_quat, self.buffer.latest()[-1])
    
    def get_root_pos(self):
        return self.buffer.latest()[-1][:3]
    
if __name__ == "__main__":
    client = ViveTrackerOnlineClient()
    client.listen()
    print("Waiting for client...")
    addr = client.accept_blocking()
    print(f"Client connected: {addr}")
    client.start()
    time.sleep(1.0)
    input()
    client.calibrate(robot_root_quat=np.array([1, 0, 0, 0], dtype=np.float64))
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        client.stop()