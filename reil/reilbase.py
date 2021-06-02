# -*- coding: utf-8 -*-
'''
ReilBase class
==============

The base class for reinforcement learning
'''

from __future__ import annotations

import bz2
import copy
import importlib
import logging
import pathlib
import time
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

import dill
from ruamel.yaml import YAML

import reil


class ReilBase:
    '''
    The base class of all classes in the `ReiL` package.
    '''

    def __init__(self,
                 name: Optional[str] = None,
                 path: Optional[pathlib.PurePath] = None,
                 logger_name: Optional[str] = None,
                 logger_level: Optional[int] = None,
                 logger_filename: Optional[str] = None,
                 persistent_attributes: Optional[List[str]] = None,
                 save_zipped: bool = True,
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

        self._logger_name = logger_name or __name__
        self._logger_level = logger_level or logging.WARNING
        self._logger_filename = logger_filename

        self._logger = logging.getLogger(self._logger_name)
        self._logger.setLevel(self._logger_level)
        if self._logger_filename is not None:
            self._logger.addHandler(logging.FileHandler(self._logger_filename))

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
                       filename: str,
                       path: Optional[Union[pathlib.PurePath, str]] = None):
        '''
        Create an instance based on a yaml file.

        Arguments
        ---------
        node_reference:
            A list of node names that determines the location of the
            specification in the yaml tree.

        filename:
            Name of the pickle file.

        path:
            Path of the pickle file.

        Returns
        -------
        :
            The generated instance.
        '''
        _path = pathlib.Path(path or '.')
        _filename = filename if filename.endswith((
            '.yaml', '.yml')) else f'{filename}.yaml'

        yaml = YAML()
        with open(_path / _filename, 'r') as f:
            yaml_output = yaml.load(f)

        temp_yaml = yaml_output
        for key in node_reference:
            temp_yaml = temp_yaml[key]

        return cls.parse_yaml(temp_yaml)

    @staticmethod
    def parse_yaml(data: OrderedDict):  # noqa: C901
        '''
        Parse a yaml tree.

        This method reads a yaml tree and recursively creates objects specified
        by it.

        Arguments
        ---------
        data:
            A yaml tree data.

        Returns
        -------
        :
            Based on the tree, the method returns:

            * A python object, e.g. `int`, `float`, `str`.
            * A dictionary of arguments and their values to be fed to the
              'parse_yaml` caller.
            * An instance of an object derived from `ReilBase`.
        '''
        if isinstance(data, (int, float, str)):
            return data

        if 'eval' in data:
            args = {'reil': reil}
            if 'args' in data:
                args.update(ReilBase.parse_yaml(data['args']))
            # else:
            #     args = {}
            return eval(data['eval'], args)

        if len(data) == 1:
            k, v = next(iter(data.items()))
            result = ReilBase._create_component_from_yaml(k, v)
            if result is not None:
                return result

        args: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                v_obj = ReilBase.parse_yaml(data[k])
            elif isinstance(v, list):
                v_obj = [ReilBase.parse_yaml(v_i)
                         for v_i in v]
            elif isinstance(v, str):
                if v.startswith('lambda'):
                    v_obj = eval(v, {})
                elif v.startswith('eval'):
                    v_obj = eval(v[4:], {})
                else:
                    v_obj = v
            else:
                v_obj = v

            args.update({k: v_obj})

        return args

    @staticmethod
    def _create_component_from_yaml(name: str, args: OrderedDict):
        '''
        Create a component from yaml data.

        This method attempts to import the `reil` class specified in `name`,
        parse arguments specified in `args` and create an instance of the
        class using the parsed arguments. If such class does not exist,
        `None` will be returned.

        Arguments
        ---------
        name:
            Name of the object to be created.

        args:
            A yaml tree section that contains arguments and values to create
            the object.

        Returns
        -------
        :
            The created object or `None`.
        '''
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
        params:
            A dictionary containing parameter names and their values.
        '''
        for key, value in params.items():
            self.__dict__[f'_{key}'] = value

    def load(self, filename: str,  # noqa: C901
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
        _filename = (filename if filename.endswith(('.pkl', '.pbz2'))
                     else
                     f'{filename}{".pbz2" if self._save_zipped else ".pkl"}')
        full_path = pathlib.Path(path or self._path) / _filename

        data = None
        for i in range(1, 6):
            try:
                if self._save_zipped:
                    with bz2.BZ2File(full_path, 'r') as f:
                        data = dill.load(f)
                else:
                    with open(full_path, 'rb') as f:
                        data = dill.load(f)
            except FileNotFoundError:
                raise
            except (EOFError, OSError):
                self._logger.info(
                    f'Attempt {i} failed to load '
                    f'{full_path}.')
                time.sleep(2)
            if data is not None:
                break

        if data is None:
            self._logger.exception(
                'Corrupted or inaccessible data file: '
                f'{full_path}')
            raise RuntimeError(
                f'Corrupted or inaccessible data file: '
                f'{full_path}')

        self._logger.info(
            'Changing the logger from '
            f'{self._logger_name} to {data["_logger_name"]}.')

        persistent_attributes = self._persistent_attributes + \
            ['_persistent_attributes']
        for key, value in data.items():
            if key not in persistent_attributes:
                self.__dict__[key] = value

        self._logger = logging.getLogger(self._logger_name)
        self._logger.setLevel(self._logger_level)
        if self._logger_filename is not None:
            self._logger.addHandler(
                logging.FileHandler(self._logger_filename))

    def save(self,
             filename: Optional[str] = None,
             path: Optional[Union[str, pathlib.PurePath]] = None,
             data_to_save: Optional[Tuple[str, ...]] = None
             ) -> Tuple[pathlib.PurePath, str]:
        '''
        Save the object to a file.

        Arguments
        ---------
        filename:
            the name of the file to be saved.

        path:
            the path in which the file should be saved.

        data_to_save:
            a list of variables that should be pickled. If omitted,
            the `agent` is saved completely.

        Returns
        -------
        :
            a `Path` object to the location of the saved file and its name
            as `str`
        '''
        if data_to_save is None:
            data = self.__dict__
        else:
            data = {d: self.__dict__[d]
                    for d in list(data_to_save) + ['_name', '_path']}

        temp = None
        if '_logger' in data:
            temp = copy.deepcopy(self._logger)
            data.pop('_logger')

        _filename = filename or self._name
        _path = pathlib.Path(path or self._path)
        _path.mkdir(parents=True, exist_ok=True)

        if self._save_zipped:
            with bz2.BZ2File(_path / f'{_filename}.pbz2', 'w') as f:
                dill.dump(data, f, dill.HIGHEST_PROTOCOL)
        else:
            with open(_path / f'{_filename}.pkl', 'wb+') as f:
                dill.dump(data, f, dill.HIGHEST_PROTOCOL)

        if temp:
            self._logger = temp

        return pathlib.PurePath(_path), _filename

    def reset(self) -> None:
        ''' Reset the object.'''
        pass

    def __repr__(self) -> str:
        return self.__class__.__qualname__


if __name__ == '__main__':
    config = '''
        test:
            eval: change_array_to_missing(1)
            args:
                change_array_to_missing:
                    eval: reil.datatypes.feature.change_array_to_missing
    '''
    # '''
    #     training_config: &training_config
    #     reil.utils.InstanceGenerator:
    #         object:
    #         reil.healthcare.subjects.Warfarin:
    #             patient:
    #             reil.healthcare.PatientWarfarinRavvaz:
    #                 model:
    #                 reil.healthcare.mathematical_models.HambergPKPD:
    #                     randomized: True
    #                 allow_missing_genotypes: True
    #         instance_counter_stops:
    #         eval: "[i*size for i in range(1, iterations + 1)]"
    #         args:
    #             size: 1000
    #             iterations: 100
    # '''
    x = ReilBase.parse_yaml(YAML().load(config))
