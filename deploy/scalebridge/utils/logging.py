"""Minimal Loguru-compatible logger backed by the Python standard library."""

from __future__ import annotations

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


class _Logger:
    def __init__(self) -> None:
        self._logger = logging.getLogger("scalebridge")

    @staticmethod
    def _format(message, args) -> str:
        text = str(message)
        if not args:
            return text
        try:
            return text.format(*args)
        except (IndexError, KeyError, ValueError):
            return " ".join([text, *(str(arg) for arg in args)])

    def debug(self, message, *args) -> None:
        self._logger.debug(self._format(message, args))

    def info(self, message, *args) -> None:
        self._logger.info(self._format(message, args))

    def warning(self, message, *args) -> None:
        self._logger.warning(self._format(message, args))

    def error(self, message, *args) -> None:
        self._logger.error(self._format(message, args))

    def exception(self, message, *args) -> None:
        self._logger.exception(self._format(message, args))


logger = _Logger()
