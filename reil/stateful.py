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
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from reil import reilbase
from reil.datatypes import PrimaryComponent, ReilData, SubComponentInfo
from reil.datatypes.components import Statistic


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
        self.state = PrimaryComponent(self,
                                      self.sub_comp_list,
                                      self._default_state_definition)
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
        for func_name, func in self.__class__.__dict__.items():
            if callable(func) and func_name[:10] == '_sub_comp_':
                keywords = list(func.__code__.co_varnames)
                # if 'self' in keywords:
                #     keywords.remove('self')
                #     f = functools.partial(v, self)
                # else:
                #     f = v

                if 'kwargs' in keywords:
                    keywords.remove('kwargs')

                if len(keywords) < 2 or keywords[1] != '_id':
                    raise ValueError(
                        f'Error in {func_name} signature: '
                        'At least two arguments should be accepted. '
                        'The first argument will receive a reference to the '
                        'object (self), and the second argument should be '
                        '"_id", which is the ID of the caller.')

                keywords.remove('_id')

                sub_comp_list[func_name[10:]] = (func, tuple(keywords))

        return sub_comp_list

    def register(self, entity_name: str, _id: Optional[int] = None) -> int:
        '''
        Register an `entity` and return its ID. If the `entity` is new, a new
        ID is generated and the `entity_name` is added to the list of
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

    def load(self, filename: str,
             path: Optional[Union[str, pathlib.Path]]) -> None:
        super().load(filename, path=path)

        # # Reassign reference to `self` in sub components that use `partial`.
        # for comp in self.state.sub_components.values():
        #     if isinstance(comp[0], functools.partial):
        #         comp[0].__setstate__(
        #             (comp[0].func, (self, *comp[0].args[1:]),
        #              comp[0].keywords, None))

        self.state.object_ref = self
        self.state.set_default_definition(
            self._default_state_definition)
        self.statistic.set_default_definition(
            self._default_statistic_definition)
