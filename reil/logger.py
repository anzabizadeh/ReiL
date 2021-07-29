import logging
from typing import Optional


class Logger:
    def __init__(
            self, logger_name: str, logger_level: Optional[int] = None,
            logger_filename: Optional[str] = None,
            fmt: Optional[str] = None) -> None:
        pass

        self._name = logger_name
        self._level = logger_level or logging.WARNING
        self._filename = logger_filename

        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(self._level)
        if not self._logger.hasHandlers():
            if self._filename is None:
                handler = logging.StreamHandler()
            else:
                handler = logging.FileHandler(self._filename)

            handler.setFormatter(logging.Formatter(
                fmt=fmt or ' %(name)s :: %(levelname)-8s :: %(message)s'))
            self._logger.addHandler(handler)
        else:
            self._logger.debug(
                f'logger {self._name} already has a handler.')

    def debug(self, msg: str):
        self._logger.debug(msg)

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def exception(self, msg: str):
        self._logger.exception(msg)

    def critical(self, msg: str):
        self._logger.critical(msg)
