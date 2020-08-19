# -*- coding: utf-8 -*-
'''
RLBase class
=================

The base class for reinforcement learning

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import logging
import numbers
import pathlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dill  # type: ignore

from rl import data_collector, rldata

Observation = Dict[str, Union[rldata.RLData, numbers.Number]]
History = List[Observation]

class RLBase():
    '''
    Super class of all classes in rl package.
    
    Attributes
    ----------
        data_collector: a DataCollector object

    Methods
    -------
        set_params: set parameters.
        set_defaults: set default values for parameters.
        load: load an object from a file.
        save: save the object to a file.
    '''
    version: str = "0.5"
    def __init__(self,
                 name: Optional[str] = 'rlbase',
                 path: str = '.',
                 ex_protocol_options: Dict[str, Sequence[str]] = {},
                 ex_protocol_current: Dict[str, str] = {},
                 stats_list: Sequence[str] = (),
                 logger_name: str = __name__,
                 logger_level: int = logging.WARNING,
                 logger_filename: Optional[str] = None,
                 persistent_attributes: Sequence[str] = (),
                 **kwargs: Any):
        # self._defaults = {}

        self.data_collector = data_collector.DataCollector(object=self)

        self._name = name
        self._path = path

        self._ex_protocol_options = ex_protocol_options
        self._ex_protocol_current = dict((protocol, ex_protocol_current.get(protocol, value[0]))
                                        for protocol, value in self._ex_protocol_options.items())
        self._stats_list = stats_list

        self._logger_name = logger_name
        self._logger_level = logger_level
        self._logger_filename = logger_filename
        self._persistent_attributes = ['_'+p for p in persistent_attributes]

        self._logger = logging.getLogger(self._logger_name)
        self._logger.setLevel(self._logger_level)
        if self._logger_filename is not None:
            self._logger.addHandler(logging.FileHandler(self._logger_filename))

        self.set_params(**kwargs)

    @classmethod
    def from_file(cls, filename: str, path: Optional[Union[pathlib.Path, str]] = None):
        instance = cls()
        instance.load(filename=filename, path=path)
        return instance

    def stats(self, stats_list: Sequence[str]) -> Dict[str, Any]:
        '''
        Compute statistics.

        Arguments
        ---------
            stats_list: list of statistics to compute.
        '''
        return {}

    def has_stat(self, stat: str) -> bool:
        return stat in self._stats_list

    @property
    def exchange_protocol_options(self) -> Dict[str, Sequence[str]]:
        return self._ex_protocol_options

    @property
    def exchange_protocol(self) -> Dict[str, str]:
        return self._ex_protocol_current

    @exchange_protocol.setter
    def exchange_protocol(self, p: Dict[str, str]) -> None:
        for k, v in p.items():
            if k in self._ex_protocol_options.keys():
                if v in self._ex_protocol_options[k]:
                    self._ex_protocol_current[k] = v
                else:
                    raise KeyError(f'Protocol {k} does not have option {v}.')

    def set_params(self, **params: Dict[str, Any]) -> None:
        '''
        set parameters to values.

        Arguments
        ---------
            params: a dictionary containing parameter names and their values.
        '''
        for key, value in params.items():
            self.__dict__[f'_{key}'] = value
        # self.__dict__.update((f'_{key}', params.get(key, self._defaults[key]))
        #                       for key in self._defaults if key in params)

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
        _path = pathlib.Path(path if path is not None else self._path)

        with open(_path / f'{filename}.pkl', 'rb') as f:
            try:
                data = dill.load(f)  #type: ignore
            except EOFError:
                try:
                    self._logger.info(f'First attempt failed to load {_path / f"{filename}.pkl"}.')
                    time.sleep(1)
                    data = dill.load(f)  #type: ignore
                except EOFError:
                    self._logger.exception(f'Corrupted or inaccessible data file: {_path / f"{filename}.pkl"}')
                    raise RuntimeError(f'Corrupted or inaccessible data file: {_path / f"{filename}.pkl"}')
            
            self._logger.info(f'Changing the logger from {self._logger_name} to {data["_logger_name"]}.')

            persistent_attributes = self._persistent_attributes + ['_persistent_attributes', 'version']
            for key, value in data.items():
                if key not in persistent_attributes:
                    self.__dict__[key] = value

            # TODO: classes should use `loaded_version` to compare old vs new and modify attributes if necessary.
            self.loaded_version = data.get('version')

            self._logger = logging.getLogger(self._logger_name)
            self._logger.setLevel(self._logger_level)
            if self._logger_filename is not None:
                self._logger.addHandler(logging.FileHandler(self._logger_filename))

            self.data_collector.object = self

    def save(self,
             filename: Optional[str] = None,
             path: Optional[Union[str, pathlib.Path]] = None,
             data_to_save: Optional[Sequence[str]] = None) -> Tuple[pathlib.Path, str]:
        '''
        Save the object to a file.

        Arguments
        ---------
            filename: the name of the object (Default=self._name)
            path: the path of the file to be loaded. (Default='.')
            data: what to save (Default: saves everything)
        '''

        _filename = filename if filename is not None else self._name
        _path: pathlib.Path = pathlib.Path(path if path is not None else self._path)

        if data_to_save is None:
            data = self.__dict__
        else:
            data = {}
            for d in data_to_save:
                data[d] = self.__dict__[d]
            for key in ('_name', '_path'):  # these should be saved automatically
                data[key] = self.__dict__[key]
        data['version'] = self.version

        _path.mkdir(parents=True, exist_ok=True)
        if '_logger' in data:
            data.pop('_logger')
        with open(_path / f'{_filename}.pkl', 'wb+') as f:
            dill.dump(data, f, dill.HIGHEST_PROTOCOL)  # type: ignore

        return _path, _filename  # type: ignore

    @staticmethod
    def get_argument(x: Any, y: Any) -> Any:
        return x if x is not None else y

    def _report(self, **kwargs: Any) -> Any:
        '''
        Report statistics using `DataCollector` class. This method can be implemented if an agent/ a subject wants to use `DataCollector`.

        Arguments
        ---------
            kwargs: data by which different statistics are computed.
        '''
        raise NotImplementedError

    def __repr__(self) -> str:
        return 'RLBase'
