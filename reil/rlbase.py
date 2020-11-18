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
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dill  # type: ignore
from ruamel.yaml import YAML

from reil import rldata, utils
from reil.stats import rl_functions

Observation = Dict[str, rldata.RLData]
History = List[Observation]
StateComponentFunction = Callable[..., Union[Dict[str, Any], rldata.RLData]]
StateComponentTuple = namedtuple('StateComponentTuple', ('func', 'kwargs'),
                                 defaults=({}))
ComponentInfo = Union[str, Tuple[str, Dict[str, Any]]]


class RLBase:
    '''
    The base class of all classes in `reil` package.

    Methods
    -------
    from_pickle: create an `RLBase` instance from a pickled (dilled) `RLBase` object.

    from_yaml: create an `RLBase` instance using specifications from a `YAML` file.

    state: the state of the subject as an RLData. Different state definitions
        can be introduced using `add_state_definition` method. _id is
        available, in case in the implementation, State is agent-dependent.
        (For example in games with partial map visibility).
        For subjects that are turn-based, it is a good practice to check
        that an agent is retrieving the state only when it is the agent's
        turn.

    default_state: the default state definition provided by the subject.
        This can be a more efficient implementation of the state, when it is
        possible.

    complete_state: returns an RLData consisting of all available state
        components. _id is available, in case in the implementation, State is
        agent-dependent.

    statistic: computes the value of the given statistic for the agent `_id`
        based on the statistic definition `name`. It should normally be called
        after each sampled path (trajectory).

    default_statistic: returns the default statistic for the agent `_id`. This
        can be a more efficient implementation of the statistic, when possible.

    add_state_definition: add a new state definition consisting of a `name`,
        and a list of state components. Each element in the list can be
        string representing component's name, a tuple representing name and
        positional arguments,  a tuple representing name and keyword
        arguments, or a tuple representing name, positional and keyword arguments.

    add_statistic_definition: add a new statistic definition consisting of a
        `name`, and statistic function, and a state definition name.
 
    set_params: set parameters.

    load: load an object from a pickle file.

    save: save (pickle) the object to a file.

    _generate_state_components: used by the subject during the `__init__`
        to create state components.
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

        self._state_definitions: Dict[str,
                                      List[StateComponentTuple]] = {'default': []}
        self._statistic_definitions: Dict[str,
                                          Tuple[rl_functions.RLFunction, str]] = {}

        self._available_state_components: Dict[str,
                                               StateComponentFunction] = {}
        # self._generate_state_components()


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

    def state(self, name: str = 'default', _id: Optional[int] = None) -> rldata.RLData:
        '''
        Returns the current state of the subject as agent `_id` might see, based
        on the state definition `name`.

        ### Arguments

        name: name of the state definition. If omitted, output of the
            `default_state` method will be returned. 

        _id: ID of the agent that calls the state method. In a multi-agent
            setting, e.g. an RTS game with fog of war, agents would see the world
            differently.
        '''
        if name.lower() == 'default':
            return self.default_state(_id)

        return rldata.RLData([f.func(**f.kwargs)
                              for f in self._state_definitions[name.lower()]])

    def default_state(self, _id: Optional[int] = None) -> rldata.RLData:
        '''
        Returns the default state definition of the subject as agent `_id` might
        see.

        ### Arguments

        _id: ID of the agent that calls the state method. In a multi-agent
            setting, e.g. an RTS game with fog of war, agents would see the world
            differently.
        '''
        return self.complete_state(_id)

    def complete_state(self, _id: Optional[int] = None) -> rldata.RLData:
        '''
        Returns all the information that the subject can provide.

         The default implementation returns all available state components with
         their default settings. Based on the state component definition of a
         child class, this can include redundant or incomplete information.

        ### Arguments

        _id: ID of the agent that calls the complete_state method.
        '''
        return rldata.RLData([f()  # type: ignore
                              for f in self._available_state_components.values()])

    def statistic(self, name: str = 'default', _id: Optional[int] = None) -> rldata.RLData:
        '''
        Returns the statistic that agent `_id` requests, based on the statistic
        definition `name`.

        ### Arguments

        name: name of the statistic definition. If omitted, output of the
            `default_statistic` method will be returned. 

        _id: ID of the agent that calls the retrieves the statistic.
        '''
        if name.lower() == 'default':
            return self.default_statistic(_id)

        f, s = self._statistic_definitions[name.lower()]
        temp = f(self.state(s, _id))

        return rldata.RLData({'name': 'reward', 'value': temp, 'lower': None, 'upper': None})

    def default_statistic(self, _id: Optional[int] = None) -> rldata.RLData:
        '''
        Returns the default statistic definition of the subject for agent `_id`.

        ### Arguments

        _id: ID of the agent that calls the reward method.
        '''
        return rldata.RLData({'name': 'default_stat', 'value': 0.0, 'lower': None, 'upper': None})

    def add_state_definition(self, name: str,
                             component_list: Tuple[ComponentInfo, ...]) -> None:
        '''
        Adds a new state definition called `name` with state components provided
        in `component_list`.

        ### Arguments

        name: name of the new state definition. ValueError is raise if the state
            already exists.

        component_list: A tuple consisting of component information. Each element
            in the list should be either (1) name of the component, or (2)
            a tuple with the name and a dict of kwargs.
        '''
        _name = name.lower()
        if _name in self._state_definitions:
            raise ValueError(f'State definition {name} already exists.')

        self._state_definitions[_name] = []
        for component in component_list:
            if isinstance(component, str):
                f = self._available_state_components[component]
                kwargs = {}
            elif isinstance(component, (tuple, list)):
                f = self._available_state_components[component[0]]
                kwargs = utils.get_argument(component[1], {})
            else:
                raise ValueError('Items in the component_list should be one of: '
                                 '(1) name of the component, '
                                 '(2) a tuple with the name and a dict of kwargs.')
            self._state_definitions[_name].append(
                StateComponentTuple(f, kwargs))

    def add_statistic_definition(self, name: str,
                                 rl_function: rl_functions.RLFunction,
                                 state_name: str) -> None:
        '''
        Adds a new statistic definition called `name` with function `rl_function`
        that uses state `state_name`.

        ### Arguments

        name: name of the new statistic definition. ValueError is raise if the
            statistic already exists.

        rl_function: An instance of `RLFunction` that gets the state of the
            subject, and computes the statistic. The rl_function should have the
            list if arguments from the state in its definition.

        state_name: The name of the state definition that should be used to
            compute the statistic. ValueError is raise if the state_name is
            undefined.
        '''
        if name.lower() in self._statistic_definitions:
            raise ValueError(f'Statistic definition {name} already exists.')

        if state_name.lower() not in self._state_definitions:
            raise ValueError(f'Unknown state name: {state_name}.')

        self._statistic_definitions[name.lower()] = (rl_function, state_name)

    def _generate_state_components(self) -> None:
        '''
        Generates all state components.

        This method should be implemented for all subjects. Each state component
        is a function/ method that computes the given state component. The
        function can have arguments with default values. It should have **kwargs
        arguments to avoid raising exceptions if unnecessary arguments are passed
        on to it.

        Finally, the function should fill a dictionary of state component names
        as keys and functions as values.

        >>> class Dummy(Subject):
        ...     some_attribute = None
        ...     def _generate_state_components(self) -> None:
        ...         def get_some_attribute(**kwargs):
        ...             return self.some_attribute
        ...         self._available_state_components = {
        ...             'some_attribute': get_some_attribute
        ...         }
        '''
        raise NotImplementedError


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
