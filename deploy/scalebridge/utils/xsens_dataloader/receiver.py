import logging
import socket
import struct
from threading import Lock, Thread
from typing import Optional, Tuple

import numpy as np

class XsensReceiver:

    def __init__(self, host: str = "0.0.0.0", port: int = 9763, timeout: float = 3.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.ori = np.zeros((63, 4), dtype=np.float64) # (wxyz)
        self.pos = np.zeros((63, 3), dtype=np.float64)
        self.time_code = 0
        self.frame_count = 0
        self._frame_lock = Lock()
        self.last_received = np.nan
        self.angle = False
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.receiver_thread: Optional[Thread] = None
        self.running = False

    def listen(self) -> None:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

    def accept_blocking(self) -> Tuple[str, int]:
        assert self.server_socket is not None
        self.client_socket, addr = self.server_socket.accept()
        return str(addr[0]), int(addr[1])

    def parse_position_packet(self, message: bytes) -> bool:
        try:
            message_id = message[:6].decode("utf-8")
            message_type = int(message_id[-2:])
            sample_counter = struct.unpack(">I", message[6:10])[0] + 1
            all_segments = struct.unpack(">B", message[11:12])[0]
            time_code = struct.unpack(">I", message[12:16])[0]

            if message_type == 2:
                packet_size = 32
                for s in range(all_segments):
                    offset = s * packet_size
                    self.pos[s, 0] = struct.unpack_from(">f", message, offset + 4 + 24)[0]
                    self.pos[s, 1] = struct.unpack_from(">f", message, offset + 8 + 24)[0]
                    self.pos[s, 2] = struct.unpack_from(">f", message, offset + 12 + 24)[0]
                    self.ori[s, 0] = struct.unpack_from(">f", message, offset + 16 + 24)[0]
                    self.ori[s, 1] = struct.unpack_from(">f", message, offset + 20 + 24)[0]
                    self.ori[s, 2] = struct.unpack_from(">f", message, offset + 24 + 24)[0]
                    self.ori[s, 3] = struct.unpack_from(">f", message, offset + 28 + 24)[0]

                self.angle = False
                self.time_code = time_code
                with self._frame_lock:
                    self.frame_count += 1
            elif message_type == 20:
                self.angle = True
            self.last_received = sample_counter
            return True
        except Exception as e:
            logging.debug(f"Xsens parse error: {e}")
            return False

    def socket_receiver(self) -> None:
        assert self.client_socket is not None
        while self.running:
            try:
                message = self.client_socket.recv(8 * 2000)
                if not message:
                    break
                self.parse_position_packet(message)
            except (socket.error, OSError, BrokenPipeError) as e:
                logging.info(f"Xsens socket stopped: {e}")
                self.running = False
                break

    def start_receiving(self) -> None:
        if self.receiver_thread is None or not self.receiver_thread.is_alive():
            self.running = True
            self.receiver_thread = Thread(target=self.socket_receiver, daemon=True)
            self.receiver_thread.start()

    def stop_receiving(self) -> None:
        self.running = False
        if self.receiver_thread is not None:
            self.receiver_thread.join(timeout=2.0)
            self.receiver_thread = None

    def close(self) -> None:
        self.stop_receiving()
        try:
            if self.client_socket is not None:
                self.client_socket.close()
        except OSError:
            pass
        try:
            if self.server_socket is not None:
                self.server_socket.close()
        except OSError:
            pass
        self.client_socket = None
        self.server_socket = None
