# -*- coding: utf-8 -*-
'''
Stateful class
==============

The base class of all stateful classes in `reil` package.

Methods
-------
state:
    the state of the subject as an ReilData. Different state definitions
    can be introduced using `add_state_definition` method. _id is
    available, in case in the implementation, State is agent-dependent.
    (For example in games with partial map visibility).
    For subjects that are turn-based, it is a good practice to check
    that an agent is retrieving the state only when it is the agent's
    turn.

default_state:
    the default state definition provided by the subject.
    This can be a more efficient implementation of the state, when it is
    possible.

complete_state:
    returns an ReilData consisting of all available state
    components. _id is available, in case in the implementation, State is
    agent-dependent.

statistic:
    computes the value of the given statistic for the agent `_id`
    based on the statistic definition `name`. It should normally be called
    after each sampled path (trajectory).

default_statistic:
    returns the default statistic for the agent `_id`. This
    can be a more efficient implementation of the statistic, when possible.

add_state_definition:
    add a new state definition consisting of a `name`,
    and a list of state components. Each element in the list can be
    string representing component's name, a tuple representing name and
    positional arguments,  a tuple representing name and keyword
    arguments, or a tuple representing name, positional and keyword arguments.

add_statistic_definition:
    add a new statistic definition consisting of a
    `name`, and statistic function, and a state definition name.

_generate_state_components:
    used by the subject during the `__init__`
    to create state components.
'''

from __future__ import annotations

import dataclasses
import pathlib
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from reil import reilbase
from reil.datatypes import reildata
from reil.stats import rl_functions


@dataclasses.dataclass
class Observation:
    state: Optional[reildata.ReilData] = None
    action: Optional[reildata.ReilData] = None
    reward: Optional[reildata.ReilData] = None

History = List[Observation]
StateComponentFunction = Callable[..., Union[Dict[str, Any], reildata.ReilData]]
StateComponentTuple = namedtuple('StateComponentTuple', ('func', 'kwargs'),
                                 defaults=({}))
ComponentInfo = Union[str, Tuple[str, Dict[str, Any]]]


class Stateful(reilbase.ReilBase):
    '''
    The base class of all stateful classes in `reil` package.
    '''

    def __init__(self,
                 name: Optional[str] = None,
                 path: Optional[pathlib.Path] = None,
                 logger_name: Optional[str] = None,
                 logger_level: Optional[int] = None,
                 logger_filename: Optional[str] = None,
                 persistent_attributes: Optional[List[str]] = None,
                 **kwargs: Any):

        self._state_definitions: Dict[str,
                                      List[StateComponentTuple]] = {'default': []}
        self._statistic_definitions: Dict[str,
                                          Tuple[rl_functions.RLFunction, str]] = {}

        self._available_state_components: Dict[str,
                                               StateComponentFunction] = {}
        super().__init__(name=name,
                         path=path,
                         logger_name=logger_name,
                         logger_level=logger_level,
                         logger_filename=logger_filename,
                         persistent_attributes=persistent_attributes,
                         **kwargs)

    def state(self, name: str = 'default', _id: Optional[int] = None) -> reildata.ReilData:
        '''
        Return the current state of the instance.

        Return the state based on the state definition `name`, and optional `_id`
        of the caller.

        Arguments
        ---------
        name:
            Name of the state definition. If omitted, output of the
            `default_state` method will be returned. 

        _id:
            ID of the agent that calls the state method. In a multi-agent
            setting, e.g. an RTS game with fog of war, agents would see the world
            differently.

        Returns
        -------
        :
            State of the instance.
        '''
        if name.lower() == 'default':
            return self.default_state(_id)

        return reildata.ReilData([f.func(**f.kwargs)
                              for f in self._state_definitions[name.lower()]])

    def default_state(self, _id: Optional[int] = None) -> reildata.ReilData:
        '''
        Return the default state definition of the subject.

        Arguments
        ---------
        _id:
            ID of the agent that calls the state method. In a multi-agent
            setting, e.g. an RTS game with fog of war, agents would see the world
            differently.

        Returns
        -------
        :
            State of the instance.

        Notes
        -----
        `default_state` can be an efficient implementation of the state, compared
        to the `state` that composes the state on the fly.

        `default_state` can be different for different callers. This can be
        implemented using `_id`.
        '''
        return self.complete_state(_id)

    def complete_state(self, _id: Optional[int] = None) -> reildata.ReilData:
        '''
        Return all the information that the subject can provide.

        Arguments
        ---------
        _id
            ID of the agent that calls the complete_state method.

        Returns
        -------
        :
            State of the instance.

        Notes
        -----
        The default implementation returns all available state components with
        their default settings. Based on the state component definition of a
        child class, this can include redundant or incomplete information.
        '''
        return reildata.ReilData([f()  # type: ignore
                              for f in self._available_state_components.values()])

    def statistic(self, name: str = 'default', _id: Optional[int] = None) -> reildata.ReilData:
        '''
        Return the statistic that caller `_id` has requested, based on the
        statistic definition `name`.

        Arguments
        ---------
        name:
            Name of the statistic definition. If omitted, output of the
            `default_statistic` method will be returned. 

        _id:
            ID of the agent that calls the retrieves the statistic.

        Returns
        -------
        :
            The requested statistic.
        '''
        if name.lower() == 'default':
            return self.default_statistic(_id)

        f, s = self._statistic_definitions[name.lower()]
        temp = f(self.state(s, _id))

        return reildata.ReilData.single_member(
            name='statistic', value=temp)  # type: ignore

    def default_statistic(self, _id: Optional[int] = None) -> reildata.ReilData:
        '''
        Return the default statistic definition of the subject for caller `_id`.

        Arguments
        ---------
        _id:
            ID of the agent that calls the reward method.

        Returns
        -------
        :
            The default statistic.
        '''
        return reildata.ReilData.single_member(
            name='default_stat', value=0.0)  # type: ignore

    def add_state_definition(self, name: str,
                             component_list: Tuple[ComponentInfo, ...]) -> None:
        '''
        Add a new state definition.

        Add a new state definition called `name` with state components provided
        in `component_list`.

        Arguments
        ---------
        name:
            Name of the new state definition.

        component_list:
            A tuple consisting of component information. Each element
            in the list should be either (1) name of the component, or (2)
            a tuple with the name and a dict of kwargs.

        Raises
        ------
        ValueError
            if the state already exists.
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
                kwargs = reilbase.get_argument(component[1], {})
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
        Add a new statistic definition.

        Add a new statistic definition called `name` with function `rl_function`
        that uses state `state_name`.

        Arguments
        ---------
        name:
            Name of the new statistic definition.

        rl_function:
            An instance of `RLFunction` that gets the state of the
            subject, and computes the statistic. The rl_function should have the
            list of arguments from the state in its definition.

        state_name:
            The name of the state definition that should be used to
            compute the statistic. ValueError is raise if the state_name is
            undefined.

        Raises
        ------
        ValueError
            if the statistic already exists.

        ValueError 
            if the state_name is undefined.
        '''
        if name.lower() in self._statistic_definitions:
            raise ValueError(f'Statistic definition {name} already exists.')

        if state_name.lower() not in self._state_definitions:
            raise ValueError(f'Unknown state name: {state_name}.')

        self._statistic_definitions[name.lower()] = (rl_function, state_name)

    def _generate_state_components(self) -> None:
        '''
        Generate all state components.

        Notes
        -----
        This method should be implemented for all subjects. Each state component
        is a function/ method that computes the given state component. The
        function can have arguments with default values. It should have `**kwargs`
        arguments to avoid raising exceptions if unnecessary arguments are passed
        on to it.

        Finally, the function should fill a dictionary of state component names
        as keys and functions as values.

        Example
        -------
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
