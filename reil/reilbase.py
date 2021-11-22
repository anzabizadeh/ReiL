# -*- coding: utf-8 -*-
'''
ReilBase class
==============

The base class for reinforcement learning
'''

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Union

from reil.logger import Logger
from reil.pickler import PickleMe


class ReilBase:
    '''
    The base class of all classes in the `ReiL` package.
    '''
    _object_version: str = '0.0.0'

    def __init__(
            self,
            name: Optional[str] = None,
            path: Optional[pathlib.PurePath] = None,
            logger_name: Optional[str] = None,
            logger_level: Optional[int] = None,
            logger_filename: Optional[str] = None,
            persistent_attributes: Optional[List[str]] = None,
            save_zipped: bool = False,
            **kwargs: Any):
        '''
        Arguments
        ---------
        name:
            An optional name for the instance that can be used to `save` the
            instance.

        path:
            An optional path to be used to `save` the instance.

        logger_name:
            Name of the `logger` that records logging messages.

        logger_level:
            Level of logging.

        logger_filename:
            An optional filename to be used by the logger.

        persistent_attributes:
            A list of attributes that should be preserved when loading an
            instance.

            For example, one might need to `load` an instance, but keep the
            name of the current instance.

            Example
            -------
            >>> instance = ReilBase(name='my_instance',
            ...                     persistent_attributes=['name'])
            >>> another_instance = ReilBase(name='another_instance')
            >>> another_instance.save('another_instance')
            >>> instance._name
            my_instance
            >>> another_instance._name
            another_instance
            >>> instance.load('another_instance')
            >>> instance._name
            my_instance

        kwargs:
            Any other attributes to set for the object.
            Note that `ReilBase` accepts any attribute and adds an
            underscore before its name.

            Example
            -------
            >>> instance = ReilBase(name='my_instance', my_attr='test')
            >>> instance._my_attr
            test
        '''
        self._name = name or self.__class__.__qualname__.lower()
        self._path = pathlib.PurePath(path or '.')
        self._save_zipped = save_zipped

        self._persistent_attributes = [
            '_' + p
            for p in (persistent_attributes or [])]

        self._logger = Logger(
            logger_name=logger_name or self._name,
            logger_level=logger_level,
            logger_filename=logger_filename
        )

        self.set_params(**kwargs)

    @classmethod
    def _empty_instance(cls):
        return cls()

    @classmethod
    def from_pickle(
            cls, filename: str,
            path: Optional[Union[pathlib.PurePath, str]] = None):
        '''
        Load a pickled instance.

        Arguments
        ---------
        filename:
            Name of the pickle file.

        path:
            Path of the pickle file.

        Returns
        -------
        :
            A `ReilBase` instance.
        '''
        instance = cls._empty_instance()
        instance._logger = Logger(logger_name=instance.__class__.__qualname__)

        instance.load(filename=filename, path=path)

        return instance

    def set_params(self, **params: Dict[str, Any]) -> None:
        '''
        set parameters to values.

        Arguments
        ---------
        params:
            A dictionary containing parameter names and their values.
        '''
        for key, value in params.items():
            self.__dict__[f'_{key}'] = value

    def load(
            self, filename: str,
            path: Optional[Union[str, pathlib.PurePath]] = None) -> None:
        '''
        Load an object from a file.

        Arguments
        ---------
        filename:
            the name of the file to be loaded.

        path:
            the path in which the file is saved.

        Raises
        ------
        ValueError
            filename is not specified.

        RuntimeError
            Corrupted or inaccessible data file.
        '''
        pickler = PickleMe.get('pbz2' if self._save_zipped else 'pkl')
        new_instance = pickler.load(filename=filename, path=path or self._path)

        for key in set(
                self._persistent_attributes + ['_persistent_attributes']):
            new_instance.__dict__[key] = self.__dict__[key]

        self.__dict__.update(new_instance.__dict__)

    def save(
        self,
        filename: Optional[str] = None,
        path: Optional[Union[str, pathlib.PurePath]] = None
    ) -> pathlib.PurePath:
        '''
        Save the object to a file.

        Arguments
        ---------
        filename:
            the name of the file to be saved.

        path:
            the path in which the file should be saved.

        Returns
        -------
        :
            a `Path` object to the location of the saved file and its name
            as `str`
        '''
        pickler = PickleMe.get('pbz2' if self._save_zipped else 'pkl')
        return pickler.dump(
            obj=self, filename=filename or self._name,
            path=path or self._path)

    def reset(self) -> None:
        ''' Reset the object.'''
        pass

    def __repr__(self) -> str:
        return self.__class__.__qualname__

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # if '_object_version' not in state:
        #     state['_object_version'] = ReilBase._object_version
        self.__dict__.update(state)
