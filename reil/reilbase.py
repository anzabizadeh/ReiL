# -*- coding: utf-8 -*-
'''
ReilBase class
============

The base class for reinforcement learning

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from __future__ import annotations

import importlib
import logging
import pathlib
import time
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

import dill
from ruamel.yaml import YAML



def get_argument(x: Any, y: Any) -> Any:
    return x if x is not None else y

class ReilBase:
    '''
    The base class of all classes in `reil` package.

    ### Methods
    from_pickle: create a `ReilBase` instance from a pickled (dilled) `ReilBase` object.

    from_yaml_file: create a `ReilBase` instance using specifications from a `YAML` file.

    parse_yaml: create a `ReilBase` instance using specifications from a parsed `YAML` document.

    set_params: set parameters.

    load: load an object from a pickle file.

    save: save (pickle) the object to a file.

    reset: reset the object.
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

        self._name = get_argument(name, __name__.lower())
        self._path = pathlib.Path(get_argument(path, '.'))

        self._persistent_attributes = ['_'+p
                                       for p in get_argument(persistent_attributes, [])]

        self._logger_name = get_argument(logger_name, __name__)
        self._logger_level = get_argument(logger_level, logging.WARNING)
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
    def from_yaml_file(cls, node_reference: Tuple[str, ...],
                       filename: str, path: Optional[Union[pathlib.Path, str]] = None):
        _path = pathlib.Path(get_argument(path, '.'))

        yaml = YAML()
        with open(_path / f'{filename}.yaml', 'r') as f:
            yaml_output = yaml.load(f)

        temp_yaml = yaml_output
        for key in node_reference:
            temp_yaml = temp_yaml[key]

        return cls.parse_yaml(temp_yaml)

    @staticmethod
    def parse_yaml(data: OrderedDict):
        # cls._validate_parsed_yaml(yaml_node_name, data)
        if isinstance(data, (int, float, str)):
            return data

        if len(data) == 1:
            k, v = next(iter(data.items()))
            result = ReilBase._load_component_from_yaml(k, v)
            if result is not None:
                return result

        args = {}
        for k, v in data.items():
            if isinstance(v, dict):
                v_obj = ReilBase.parse_yaml(data[k])
            elif isinstance(v, list):
                v_obj = [ReilBase.parse_yaml(v_i)
                         for v_i in v]
            elif isinstance(v, str) and 'lambda' in v:
                v_obj = eval(v)
            else:
                v_obj = v

            args.update({k: v_obj})
                # if isinstance(v, dict):
                #     new_k, new_v = next(iter(v.items()))
                #     try:
                #         v_obj = ReilBase._load_component_from_yaml(
                #             new_k, new_v)
                #         args.update({k: v_obj})
                #     except ValueError:  # Empty module name
                #         args.update({k: v})
                # elif isinstance(v, str) and 'lambda' in v:
                #     args.update({k: eval(v)})
                # else:
                #     args.update({k: v})

        return args

    @staticmethod
    def _load_component_from_yaml(name: str, args: Any):
        temp = name.split('.')
        try:
            module = importlib.import_module('.'.join(temp[:-1]))
        except ValueError:
            return None

        f = getattr(module, temp[-1])
        if hasattr(f, 'parse_yaml'):
            result = f(**f.parse_yaml(args))
        else:
            result = f(**ReilBase.parse_yaml(args))

        return result

    def set_params(self, **params: Dict[str, Any]) -> None:
        '''
        set parameters to values.

        Arguments
        ---------
            params: a dictionary containing parameter names and their values.
        '''
        for key, value in params.items():
            self.__dict__[f'_{key}'] = value

    def load(self, filename: str, path: Optional[Union[str, pathlib.Path]] = None) -> None:
        '''
        Load an object from a file.

        Arguments
        ---------
            filename: the name of the file to be loaded.
            path: the path of the file to be loaded. (Object's default path will be used if not provided)

        '''
        _path = pathlib.Path(get_argument(path, self._path))

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

            # TODO: classes should use `loaded_version` to compare old vs new
            # and modify attributes if necessary.
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

        _filename: str = get_argument(filename, self._name)
        _path: pathlib.Path = pathlib.Path(
            get_argument(path, self._path))

        _path.mkdir(parents=True, exist_ok=True)
        with open(_path / f'{_filename}.pkl', 'wb+') as f:
            dill.dump(data, f, dill.HIGHEST_PROTOCOL)  # type: ignore

        return _path, _filename

    def reset(self) -> None:
        ''' Resets the object.'''
        pass

    def __repr__(self) -> str:
        return self.__class__.__qualname__ + f"\t(Version = {self.version})"
