import json
import socket
import time
from typing import Any, Callable


class TcpJsonSender:
    def __init__(self, host: str, port: int, timeout: float = 5.0, retry_interval: float = 1.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retry_interval = retry_interval
        self.socket = None

    def connect(self) -> None:
        if self.socket is not None:
            return
        self.socket = socket.create_connection((self.host, self.port), timeout=self.timeout)

    def send_message(self, payload: Any) -> None:
        message = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        while True:
            try:
                self.connect()
                if self.socket is None:
                    raise ConnectionError("TCP socket is not connected.")
                self.socket.sendall(message)
                return
            except (ConnectionError, OSError):
                self.close()
                if self.retry_interval <= 0:
                    raise
                time.sleep(self.retry_interval)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class TcpJsonReceiver:
    def __init__(self, host: str = "0.0.0.0", port: int = 5000, backlog: int = 5, accept_timeout: float = 0.5):
        self.host = host
        self.port = port
        self.backlog = backlog
        self.accept_timeout = accept_timeout
        self.server_socket = None
        self._running = False

    def serve_forever(self, handler: Callable[[Any, tuple], None]) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(self.backlog)
            server_socket.settimeout(self.accept_timeout)
            self.server_socket = server_socket
            self._running = True
            while self._running:
                try:
                    connection, address = server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._running:
                        raise
                    break
                with connection:
                    self._handle_connection(connection, address, handler)
            self.server_socket = None

    def _handle_connection(self, connection: socket.socket, address: tuple, handler: Callable[[Any, tuple], None]) -> None:
        buffer = b""
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                raw_message, buffer = buffer.split(b"\n", 1)
                if not raw_message.strip():
                    continue
                handler(json.loads(raw_message.decode("utf-8")), address)

    def close(self) -> None:
        self._running = False
        if self.server_socket is not None:
            self.server_socket.close()
            self.server_socket = None
