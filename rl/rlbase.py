# -*- coding: utf-8 -*-
'''
RLBase class
=================

The base class for reinforcement learning

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import logging
from pathlib import Path
from random import randrange
from time import sleep
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dill import HIGHEST_PROTOCOL, dump, load

from .data_collector import DataCollector


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
    def __init__(self, persistent_attributes: List[str] = [],
        **kwargs):
        self._defaults = {}
        self.data_collector = DataCollector(object=self)

        self.set_defaults(name=self.__repr__() + f'-{str(randrange(1, 1000000)):0<7}', version=0.3, path='.',
                          ex_protocol_options={}, ex_protocol_current={},  # requested_exchange_protocol={},
                          stats_list=[], logger_name=__name__, logger_level=logging.WARNING, logger_filename=f'{__name__}.log',
                          persistent_attributes=persistent_attributes)
        self.set_params(**kwargs)

        self._logger = logging.getLogger(self._logger_name)
        self._logger.setLevel(self._logger_level)
        self._logger.addHandler(logging.FileHandler(self._logger_filename))

        if False:
            self._name, self._version, self._path = [], [], []
            self._ex_protocol_options, self._ex_protocol_current, self._requested_exchange_protocol = {}, {}, {}
            self._stats_list = []
            self._persistent_attributes = []
    
    def stats(self, stats_list: Sequence) -> Dict[str, Any]:
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

    # @property
    # def requested_exchange_protocol(self) -> Dict[str, str]:
    #     return self._requested_exchange_protocol

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
            self.__dict__['_'+key] = value
        # self.__dict__.update(('_'+key, params.get(key, self._defaults[key]))
        #                       for key in self._defaults if key in params)

    def set_defaults(self, **params: Dict[str, Any]) -> None:
        '''
        set parameters default values.

        Arguments
        ---------
            params: a dictionary containing parameter names and their default values.

        Note: this method overwrites all variable names.
        '''
        if not hasattr(self, '_defaults'):
            self._defaults = {}
        for key, value in params.items():
            # self._defaults[key] = value
            # self.__dict__['_'+key] = value
            if not hasattr(self, '_'+key) or self.__dict__.get('_'+key, -1) in (None, {}, []):
                self._defaults[key] = value
                self.__dict__['_'+key] = value

    def load(self, filename: str, path: Optional[str] = None) -> None:
        '''
        Load an object from a file.

        Arguments
        ---------
            filename: the name of the file to be loaded.
            path: the path of the file to be loaded. (Object's default path will be used if not provided)

        '''
        _path = Path(path if path is not None else self._path)

        with open(_path / f'{filename}.pkl', 'rb') as f:
            try:
                data = load(f)
            except EOFError:
                try:
                    self._logger.info(f'First attempt failed to load {_path / f"{filename}.pkl"}.')
                    sleep(1)
                    data = load(f)
                except EOFError:
                    self._logger.exception(f'Corrupted or inaccessible data file: {_path / f"{filename}.pkl"}')
                    raise RuntimeError(f'Corrupted or inaccessible data file: {_path / f"{filename}.pkl"}')
            
            self._logger.info(f'Changing the logger from {self._logger_name} to {data["_logger_name"]}.')

            persistent_attributes = self._persistent_attributes + ['_persistent_attributes']
            for key, value in data.items():
                if key[1:] not in persistent_attributes:
                    self.__dict__[key] = value

            self._logger.removeHandler(logging.FileHandler(self._logger_filename))
            self._logger = logging.getLogger(self._logger_name)
            self._logger.setLevel(self._logger_level)
            self._logger.addHandler(logging.FileHandler(self._logger_filename))

            self.data_collector._object = self

    def save(self, filename: Optional[str] = None, path: Optional[str] = None, data_to_save: Optional[Sequence[str]] = None) -> Tuple[Path, str]:
        '''
        Save the object to a file.

        Arguments
        ---------
            filename: the name of the object (Default=self._name)
            path: the path of the file to be loaded. (Default='.')
            data: what to save (Default: saves everything)
        '''

        _filename: str = filename if filename is not None else self._name
        _path: Path = Path(path if path is not None else self._path)

        if data_to_save is None:
            data = self.__dict__
        else:
            data = {}
            for d in data_to_save:
                data[d] = self.__dict__[d]
            for key in ('_name', '_version', '_path'):  # these should be saved automatically
                data[key] = self.__dict__[key]

        _path.mkdir(parents=True, exist_ok=True)
        data.pop('_logger')
        with open(_path / f'{_filename}.pkl', 'wb+') as f:
            dump(data, f, HIGHEST_PROTOCOL)

        return _path, _filename

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
