# -*- coding: utf-8 -*-
'''
Stateful class
==============

The base class of all stateful classes in `reil` package.

Methods
-------
state:
    the state of the entity (`agent` or `subject`) as an ReilData. Different
    state definitions can be introduced using `state.add_definition` method.
    _id is available, in case in the implementation, state is caller-dependent.
    (For example in games with partial map visibility).
    For subjects that are turn-based, it is a good practice to check
    that an agent is retrieving the state only when it is the agent's
    turn.

statistic:
    compute the value of the given statistic for the entity `_id`
    based on the statistic definition `name`. It should normally be called
    after each sampled path (trajectory). Different statistic definitions
    can be introduced using `statistic.add_definition` method.

_extract_sub_components:
    Extract methods that begin with `_sub_comp_`.

register:
    Register an external `entity` (`agents` for `subjects` and vice versa.)

deregister:
    Deregister an external `entity` (`agents` for `subjects` and vice versa.)
'''

from __future__ import annotations

import dataclasses
import functools
import pathlib
from reil.datatypes.components import Statistic
from typing import Any, Dict, List, Optional, Tuple, cast

from reil import reilbase
from reil.datatypes import (PrimaryComponent, ReilData, SecondayComponent,
                            SubComponentInfo)


@dataclasses.dataclass
class Observation:
    state: Optional[ReilData] = None
    action: Optional[ReilData] = None
    reward: Optional[ReilData] = None


History = List[Observation]


class EntityRegister:
    '''
    Create and maintain a list of registered `entities`.


    :meta private:
    '''

    def __init__(self, min_entity_count: int, max_entity_count: int,
                 unique_entities: bool = True):
        '''
        Arguments
        ---------
        min_entity_count:
            The minimum number of `entities` needed to be registered so that
            the current `instance` is ready for interaction.

        max_entity_count:
            The maximum number of `entities` that can interact with this
            instance.

        unique_entities:
            If `True`, each `entity` can be registered only once.
        '''
        self._id_list: List[int] = []
        self._entity_list: List[str] = []
        self._min_entity_count = min_entity_count
        self._max_entity_count = max_entity_count
        self._unique_entities = unique_entities

    @property
    def ready(self) -> bool:
        '''
        Determine if enough `entities` are registered.

        Returns
        -------
        :
            `True` if enough `entities` are registered, else `False`.
        '''
        return len(self._id_list) >= self._min_entity_count

    def append(self, entity_name: str, _id: Optional[int] = None) -> int:
        '''
        Add a new `entity` to the end of the list.

        Parameters
        ----------
        entity_name:
            The name of the `entity` to add.

        _id:
            If provided, method tries to register the `entity` with the given
            ID.

        Returns
        -------
        :
            The ID assigned to the `entity`.

        Raises
        ------
        ValueError:
            Capacity is reached. No new `entities` can be registered.

        ValueError:
            ID is already taken.

        ValueError:
            `entity_name` is already registered with a different ID.
        '''
        if (0 < self._max_entity_count < len(self._id_list)):
            raise ValueError('Capacity is reached. No new entities can be'
                             ' registered.')

        new_id = cast(int, _id)
        if self._unique_entities:
            if entity_name in self._entity_list:
                current_id = self._id_list[
                    self._entity_list.index(entity_name)]
                if _id is None or _id == current_id:
                    return current_id
                else:
                    raise ValueError(
                        f'{entity_name} is already registered with '
                        f'ID: {current_id}.')
            elif _id is None:
                new_id = max(self._id_list, default=0) + 1
            elif _id in self._id_list:
                raise ValueError(f'{_id} is already taken.')
            # else:
            #     new_id = _id
        elif _id is None:
            new_id = max(self._id_list, default=0) + 1
        elif _id in self._id_list:
            current_entity = self._entity_list[
                self._id_list.index(_id)]
            if entity_name == current_entity:
                return _id
            else:
                raise ValueError(f'{_id} is already taken.')
        # else:
        #     new_id = _id

        self._entity_list.append(entity_name)
        self._id_list.append(new_id)

        return new_id

    def remove(self, _id: int):
        '''
        Remove the `entity` registered by ID=`_id`.

        Arguments
        ---------
        _id:
            ID of the `entity` to remove.
        '''
        entity_name = self._entity_list[self._id_list.index(_id)]
        self._entity_list.remove(entity_name)
        self._id_list.remove(_id)

    def __contains__(self, _id: int) -> bool:
        return _id in self._id_list

class Stateful(reilbase.ReilBase):
    '''
    The base class of all stateful classes in the `ReiL` package.
    '''

    def __init__(self,
                 min_entity_count: int = 1,
                 max_entity_count: int = -1,
                 unique_entities: bool = True,
                 name: Optional[str] = None,
                 path: Optional[pathlib.Path] = None,
                 logger_name: Optional[str] = None,
                 logger_level: Optional[int] = None,
                 logger_filename: Optional[str] = None,
                 persistent_attributes: Optional[List[str]] = None,
                 **kwargs: Any):

        super().__init__(name=name,
                         path=path,
                         logger_name=logger_name,
                         logger_level=logger_level,
                         logger_filename=logger_filename,
                         persistent_attributes=persistent_attributes,
                         **kwargs)

        self.sub_comp_list = self._extract_sub_components()
        self.state = PrimaryComponent(
            self.sub_comp_list, self._default_state_definition)
        self.statistic = Statistic(
            name='statistic', primary_component=self.state,
            default_definition=self._default_statistic_definition)

        self._entity_list = EntityRegister(min_entity_count=min_entity_count,
                                           max_entity_count=max_entity_count,
                                           unique_entities=unique_entities)

    def _default_state_definition(
            self, _id: Optional[int] = None) -> ReilData:
        return ReilData.single_base(name='default_state', value=None)

    def _default_statistic_definition(
            self, _id: Optional[int] = None) -> Tuple[ReilData, float]:
        return (self._default_state_definition(_id), 0.0)

    def _extract_sub_components(self) -> Dict[str, SubComponentInfo]:
        '''
        Extract all sub components.

        Notes
        -----
        Each sub component is a method that computes the value of the given
        sub component. The method should have the following properties:
        * Method's name should start with "_sub_comp_".
        * The first argument (except for `self`) should be `_id` which is the
          ID of the object using this sub component.
        * Method should have `**kwargs` argument to avoid raising exceptions if
          unnecessary arguments are passed on to it.
        * Method can have arguments with default values
        * Method should return a dictionary with mandatory keys `name` and
          `value` and optional keys, such as `lower` and `upper`, and
          `categories`.

        Example
        -------
        >>> class Dummy(Stateful):
        ...     def __init__(self) -> None:
        ...         self._some_attribute = 'some value'
        ...         sub_comp_list = self._extract_sub_components()
        ...         self.a_component = Component(tuple(sub_comp_list))
        ...
        ...     def _sub_comp_01(self, _id, **kwargs):
        ...         return {'name': 'sub_comp_01', 'value': 'something'}
        ...
        ...     def _sub_comp_02(self, _id, arg_01, **kwargs):
        ...         return {'name': 'sub_comp_02',
        ...                 'value': self._some_attribute * arg_01}
        >>> d = Dummy()
        >>> d.a_component.add_definition(
        ...     'a_definition',
        ...     (SubComponentInstance('01'),
        ...      SubComponentInstance('02', {'arg_01': 3})))
        >>> print(d.a_component('a_definition', _id=1).value)
        {'sub_comp_01': 'something', 'sub_comp_02':
        'some valuesome valuesome value'}
        >>> d._some_attribute = 'new value'
        >>> print(d.a_component('a_definition', _id=1).value)
        {'sub_comp_01': 'something', 'sub_comp_02':
        'new valuenew valuenew value'}
        '''
        sub_comp_list = {}
        for k, v in self.__class__.__dict__.items():
            if callable(v) and k[:10] == '_sub_comp_':
                keywords = list(v.__code__.co_varnames)
                if 'self' in keywords:
                    keywords.remove('self')
                    f = functools.partial(v, self)
                else:
                    f = v

                if 'kwargs' in keywords:
                    keywords.remove('kwargs')

                if len(keywords) == 0 or keywords[0] != '_id':
                    raise ValueError(
                        f'Error in {k} signature: '
                        'The first argument, except for "self", '
                        'should be "_id".')

                if '_id' in keywords:
                    keywords.remove('_id')

                sub_comp_list[k[10:]] = (f, tuple(keywords))

        return sub_comp_list

    def register(self, entity_name: str, _id: Optional[int] = None) -> int:
        '''
        Register an `entity` and return its ID. If the `entity` is new, a new ID
        is generated and the `entity_name` is added to the list of
        registered entities.

        Arguments
        ---------
        entity_name:
            The name of the `entity` to be registered.

        _id:
            The ID of the entity to be used. If not provided, instance will
            assign an ID to the `entity`.

        Returns
        -------
        :
            ID of the registered `entity`.

        Raises
        ------
        ValueError:
            Attempt to register an already registered `entity` with a new ID.

        ValueError:
            Attempt to register an `entity` with an already assigned ID.

        ValueError:
            Reached max capacity.
        '''
        return self._entity_list.append(entity_name=entity_name, _id=_id)

    def deregister(self, entity_id: int) -> None:
        '''
        Deregister an `entity` given its ID.

        Arguments
        ---------
        entity_id:
            The ID of the `entity` to be deregistered.
        '''
        self._entity_list.remove(entity_id)


# StateComponentFunction = Callable[...,
#                                   Union[Dict[str, Any], ReilData]]
# StateComponentTuple = namedtuple('StateComponentTuple', ('func', 'kwargs'),
#                                  defaults=({}))
# ComponentInfo = Union[str, Tuple[str, Dict[str, Any]]]

# def statistic(self,
#               name: str = 'default',
#               _id: Optional[int] = None) -> ReilData:
#     '''
#     Return the statistic that caller `_id` has requested, based on the
#     statistic definition `name`.

#     Arguments
#     ---------
#     name:
#         Name of the statistic definition. If omitted, output of the
#         `default_statistic` method will be returned.

#     _id:
#         ID of the caller the retrieves the statistic.

#     Returns
#     -------
#     :
#         The requested statistic.
#     '''
#     return self._statistic(name, _id)


# def state(self,
#           name: str = 'default',
#           _id: Optional[int] = None) -> ReilData:
#     '''
#     Return the current state of the instance.

#     Return the state based on the state definition `name`,
#     and optional `_id` of the caller.

#     Arguments
#     ---------
#     name:
#         Name of the state definition. If omitted, output of the
#         `default_state` method will be returned.

#     _id:
#         ID of the agent that calls the state method. In a multi-agent
#         setting, e.g. an RTS game with fog of war, agents would see the
#         world differently.

#     Returns
#     -------
#     :
#         State of the instance.
#     '''
#     return self._state(name, _id)

# def state(self,
#           name: str = 'default',
#           _id: Optional[int] = None) -> ReilData:
#     if name.lower() == 'default':
#         return self.default_state(_id)

#     return ReilData([f.func(**f.kwargs)
#                      for f in self._state_definitions[name.lower()]])

# def add_state_definition(
#    self, name: str,
#    component_list: Tuple[ComponentInfo, ...]) -> None:
#     '''
#     Add a new state definition.

#     Add a new state definition called `name` with state components
#     provided in `component_list`.

#     Arguments
#     ---------
#     name:
#         Name of the new state definition.

#     component_list:
#         A tuple consisting of component information. Each element
#         in the list should be either (1) name of the component, or (2)
#         a tuple with the name and a dict of kwargs.

#     Raises
#     ------
#     ValueError
#         if the state already exists.
#     '''
#     _name = name.lower()
#     if _name in self._state_definitions:
#         raise ValueError(f'State definition {name} already exists.')

#     self._state_definitions[_name] = []
#     for component in component_list:
#         if isinstance(component, str):
#             f = self._available_state_components[component]
#             kwargs = {}
#         elif isinstance(component, (tuple, list)):
#             f = self._available_state_components[component[0]]
#             kwargs = reilbase.get_argument(component[1], {})
#         else:
#             raise ValueError(
#                 'Items in the component_list should be one of: '
#                 '(1) name of the component, '
#                 '(2) a tuple with the name and a dict of kwargs.')
#         self._state_definitions[_name].append(
#             StateComponentTuple(f, kwargs))

# def default_state(self, _id: Optional[int] = None) -> ReilData:
#     '''
#     Return the default state definition of the subject.

#     Arguments
#     ---------
#     _id:
#         ID of the agent that calls the state method. In a multi-agent
#         setting, e.g. an RTS game with fog of war, agents would see the world
#         differently.

#     Returns
#     -------
#     :
#         State of the instance.

#     Notes
#     -----
#     `default_state` can be an efficient implementation of the state, compared
#     to the `state` that composes the state on the fly.

#     `default_state` can be different for different callers. This can be
#     implemented using `_id`.
#     '''
#     return self.complete_state(_id)

# def complete_state(self, _id: Optional[int] = None) -> ReilData:
#     '''
#     Return all the information that the subject can provide.

#     Arguments
#     ---------
#     _id
#         ID of the agent that calls the complete_state method.

#     Returns
#     -------
#     :
#         State of the instance.

#     Notes
#     -----
#     The default implementation returns all available state components with
#     their default settings. Based on the state component definition of a
#     child class, this can include redundant or incomplete information.
#     '''
#     return ReilData([f()  # type: ignore
#                      for f in self._available_state_components.values()])

# def statistic(self,
#               name: str = 'default',
#               _id: Optional[int] = None) -> ReilData:
#     if name.lower() == 'default':
#         return self.default_statistic(_id)

#     f, s = self._statistic_definitions[name.lower()]
#     temp = f(self.state(s, _id))

#     return ReilData.single_base(
#         name='statistic', value=temp)

# def add_statistic_definition(self, name: str,
#                              rl_function: reil_functions.ReilFunction,
#                              state_name: str) -> None:
#     '''
#     Add a new statistic definition.

#     Add a new statistic definition called `name` with function `rl_function`
#     that uses state `state_name`.

#     Arguments
#     ---------
#     name:
#         Name of the new statistic definition.

#     rl_function:
#         An instance of `ReilFunction` that gets the state of the
#         subject, and computes the statistic. The rl_function should have the
#         list of arguments from the state in its definition.

#     state_name:
#         The name of the state definition that should be used to
#         compute the statistic. ValueError is raise if the state_name is
#         undefined.

#     Raises
#     ------
#     ValueError
#         if the statistic already exists.

#     ValueError
#         if the state_name is undefined.
#     '''
#     if name.lower() in self._statistic_definitions:
#         raise ValueError(f'Statistic definition {name} already exists.')

#     if state_name.lower() not in self._state_definitions:
#         raise ValueError(f'Unknown state name: {state_name}.')

#     self._statistic_definitions[name.lower()] = (rl_function, state_name)

# def default_statistic(self, _id: Optional[int] = None) -> ReilData:
#     '''
#     Return the default statistic definition of the subject for caller `_id`.

#     Arguments
#     ---------
#     _id:
#         ID of the agent that calls the reward method.

#     Returns
#     -------
#     :
#         The default statistic.
#     '''
#     return ReilData.single_base(
#         name='default_stat', value=0.0)

# def _generate_state_components(self) -> None:
#     '''
#     Generate all state components.

#     Notes
#     -----
#     This method should be implemented for all subjects. Each state component
#     is a function/ method that computes the given state component. The
#     function can have arguments with default values. It should have
#     `**kwargs`
#     arguments to avoid raising exceptions if unnecessary arguments are passed
#     on to it.

#     Finally, the function should fill a dictionary of state component names
#     as keys and functions as values.

#     Example
#     -------
#     >>> class Dummy(Subject):
#     ...     some_attribute = None
#     ...     def _generate_state_components(self) -> None:
#     ...         def get_some_attribute(**kwargs):
#     ...             return self.some_attribute
#     ...         self._available_state_components = {
#     ...             'some_attribute': get_some_attribute
#     ...         }
#     '''
#     raise NotImplementedError
