# -*- coding: utf-8 -*-
'''
Logger class
============

Provides the logging capabilities for `ReiL` objects.
'''
import logging
from typing import Any

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

DEFAULT_FORMAT = ' %(name)s :: %(levelname)-8s :: %(message)s'


class Logger:
    '''
    Logger class.
    '''
    def __init__(
            self, logger_name: str, logger_level: int | None = None,
            logger_filename: str | None = None,
            fmt: str | None = None) -> None:
        '''
        Initializes a logger.

        Arguments
        ---------
        logger_name:
            The name of the logger.

        logger_level:
            The level of the logger. If not given, defaults to `WARNING`.

        logger_filename:
            The filename of the logger. If not given, the logger will be
            displayed on the standard output.

        fmt:
            The format of the logger. If not given, defaults to
            `DEFAULT_FORMAT`.
        '''
        self._name = logger_name
        self._level = logger_level or logging.WARNING
        self._filename = logger_filename
        self._fmt = fmt or DEFAULT_FORMAT

        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(self._level)
        if not self._logger.hasHandlers():
            if self._filename is None:
                handler = logging.StreamHandler()
            else:
                handler = logging.FileHandler(self._filename)

            handler.setFormatter(logging.Formatter(fmt=self._fmt))
            self._logger.addHandler(handler)
        else:
            self._logger.debug(
                f'logger {self._name} already has a handler.')

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        '''
        Initializes a logger from a configuration dictionary.

        Arguments
        ---------
        config:
            The configuration dictionary.

        Returns
        -------
        :
            A logger object.
        '''
        return cls(**config)

    def get_config(self) -> dict[str, Any]:
        '''
        Returns the configuration dictionary of the logger.

        Returns
        -------
        :
            The configuration dictionary of the logger.
        '''
        return self.__getstate__()

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

    def __getstate__(self):
        '''
        Return the object state for pickling.

        Returns
        -------
        :
            The object state.
        '''
        state = dict(
            name=self._name,
            level=self._level,
            filename=self._filename)

        if self._fmt != DEFAULT_FORMAT:
            state.update({'fmt': self._fmt})

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        '''
        Set the object state from pickling.

        Arguments
        ---------
        state:
            The object state.
        '''
        try:
            self.__init__(
                logger_name=state['name'], logger_level=state.get('level'),
                logger_filename=state.get('filename'), fmt=state.get('fmt'))
        except KeyError:  # compatibility with old versions
            try:
                self.__init__(
                    logger_name=state['_name'],
                    logger_level=state.get('_level'),
                    logger_filename=state.get('_filename'),
                    fmt=state.get('_fmt'))
            except KeyError:
                self.__init__(
                    logger_name=state['logger_name'],
                    logger_level=state.get('logger_level'),
                    logger_filename=state.get('logger_filename'),
                    fmt=state.get('_fmt'))
