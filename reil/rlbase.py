# -*- coding: utf-8 -*-
'''
RLBase class
============

The base class for reinforcement learning

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import logging
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import dill  # type: ignore
from ruamel.yaml import YAML

from reil import rldata, utils

Observation = Dict[str, rldata.RLData]
History = List[Observation]


class RLBase:
    '''
    The base class of all classes in `reil` package.

    Methods
    -------
    from_pickle: create an `RLBase` instance from a pickled (dilled) `RLBase` object.

    from_yaml: create an `RLBase` instance using specifications from a `YAML` file.

    stats: compute statistics for the object and returns a dictionary.
 
    set_params: set parameters.

    load: load an object from a pickle file.

    save: save (pickle) the object to a file.
    '''

    version: str = "0.7"

    def __init__(self,
                 name: Optional[str] = None,
                 path: Optional[pathlib.Path] = None,
                 logger_name: Optional[str] = None,
                 logger_level: Optional[int] = None,
                 logger_filename: Optional[str] = None,
                 persistent_attributes: Optional[List[str]] = None,
                 **kwargs: Any):

        self._name = utils.get_argument(name, __name__.lower())
        self._path = pathlib.Path(utils.get_argument(path, '.'))

        self._persistent_attributes = ['_'+p
                                       for p in utils.get_argument(persistent_attributes, [])]

        self._logger_name = utils.get_argument(logger_name, __name__)
        self._logger_level = utils.get_argument(logger_level, logging.WARNING)
        self._logger_filename = logger_filename

        self._logger = logging.getLogger(self._logger_name)
        self._logger.setLevel(self._logger_level)
        if self._logger_filename is not None:
            self._logger.addHandler(logging.FileHandler(self._logger_filename))

        self.set_params(**kwargs)

    @classmethod
    def from_pickle(cls, filename: str,
                    path: Optional[Union[pathlib.Path, str]] = None):
        instance = cls()
        instance._logger_name = __name__
        instance._logger_level = logging.WARNING
        instance._logger_filename = None
        instance._logger = logging.getLogger(instance._logger_name)
        instance._logger.setLevel(instance._logger_level)
        if instance._logger_filename is not None:
            instance._logger.addHandler(
                logging.FileHandler(instance._logger_filename))

        instance.load(filename=filename, path=path)
        return instance

    @classmethod
    def from_yaml(cls, yaml_node_name: str,
                  filename: str, path: Optional[Union[pathlib.Path, str]] = None):
        _path = pathlib.Path(utils.get_argument(path, '.'))

        yaml = YAML()
        yaml_output = yaml.load(_path / f'{filename}.yaml')

        if yaml_node_name not in yaml_output:
            raise ValueError(f'{yaml_output} not found in {filename}.yaml')

        obj_type = yaml_output[yaml_node_name].get('type', '')
        if cls.__name__ != obj_type:
            raise TypeError(
                f'Attempted to load an object of type {obj_type} using class {cls.__name__}')

        instance = cls(**yaml_output[yaml_node_name])

        return instance

    def set_params(self, **params: Dict[str, Any]) -> None:
        '''
        set parameters to values.

        Arguments
        ---------
            params: a dictionary containing parameter names and their values.
        '''
        for key, value in params.items():
            self.__dict__[f'_{key}'] = value

    def set_defaults(self, **params: Dict[str, Any]) -> None:
        '''
        set parameters default values.

        Arguments
        ---------
            params: a dictionary containing parameter names and their default values.

        Note: this method overwrites all variable names.
        '''
        # if not hasattr(self, '_defaults'):
        #     self._defaults = {}
        # for key, value in params.items():
        #     # self._defaults[key] = value
        #     # self.__dict__[f'_{key}'] = value
        #     if not hasattr(self, f'_{key}') or self.__dict__.get(f'_{key}', -1) in (None, {}, []):
        #         self._defaults[key] = value
        #         self.__dict__[f'_{key}'] = value
        self.set_params(**params)

    def load(self, filename: str, path: Optional[Union[str, pathlib.Path]] = None) -> None:
        '''
        Load an object from a file.

        Arguments
        ---------
            filename: the name of the file to be loaded.
            path: the path of the file to be loaded. (Object's default path will be used if not provided)

        '''
        _path = pathlib.Path(utils.get_argument(path, self._path))

        with open(_path / f'{filename}.pkl', 'rb') as f:
            try:
                data = dill.load(f)  # type: ignore
            except EOFError:
                try:
                    # self._logger.info(f'First attempt failed to load {_path / f"{filename}.pkl"}.')
                    time.sleep(1)
                    data = dill.load(f)  # type: ignore
                except EOFError:
                    # self._logger.exception(f'Corrupted or inaccessible data file: {_path / f"{filename}.pkl"}')
                    raise RuntimeError(
                        f'Corrupted or inaccessible data file: {_path / f"{filename}.pkl"}')

            # self._logger.info(f'Changing the logger from {self._logger_name} to {data["_logger_name"]}.')

            persistent_attributes = self._persistent_attributes + \
                ['_persistent_attributes', 'version']
            for key, value in data.items():
                if key not in persistent_attributes:
                    self.__dict__[key] = value

            # TODO: classes should use `loaded_version` to compare old vs new and modify attributes if necessary.
            self.loaded_version = data.get('version')

            self._logger = logging.getLogger(self._logger_name)
            self._logger.setLevel(self._logger_level)
            if self._logger_filename is not None:
                self._logger.addHandler(
                    logging.FileHandler(self._logger_filename))

    def save(self,
             filename: Optional[str] = None,
             path: Optional[Union[str, pathlib.Path]] = None,
             data_to_save: Optional[Tuple[str, ...]] = None) -> Tuple[pathlib.Path, str]:
        '''
        Save the object to a file.

        Arguments
        ---------
            filename: the name of the object (Default=self._name)
            path: the path of the file to be loaded. (Default='.')
            data_to_save: what to save (Default: saves everything)
        '''
        if data_to_save is None:
            data = self.__dict__.copy()
        else:
            data = dict((d, self.__dict__[d])
                        for d in list(data_to_save) + ['_name', '_path'])

        if '_logger' in data:
            data.pop('_logger')

        data['version'] = self.version

        _filename: str = utils.get_argument(filename, self._name)
        _path: pathlib.Path = pathlib.Path(
            utils.get_argument(path, self._path))

        _path.mkdir(parents=True, exist_ok=True)
        with open(_path / f'{_filename}.pkl', 'wb+') as f:
            dill.dump(data, f, dill.HIGHEST_PROTOCOL)  # type: ignore

        return _path, _filename

    def __repr__(self) -> str:
        return self.__class__.__qualname__ + f"\t(Version = {self.version})"
