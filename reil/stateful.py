# -*- coding: utf-8 -*-
'''
Stateful class
==============

The base class of all stateful classes in `reil` package.

Methods
-------
state:
    the state of the entity (`agent` or `subject`) as a FeatureArray. Different
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

import pathlib
from typing import Any, Dict, Optional, Tuple, Union

from reil import reilbase
from reil.datatypes.components import (PrimaryComponent, Statistic,
                                       SubComponentInfo)
from reil.datatypes.entity_register import EntityRegister
from reil.datatypes.feature import Feature, FeatureArray


class Stateful(reilbase.ReilBase):
    '''
    The base class of all stateful classes in the `ReiL` package.
    '''

    def __init__(
            self,
            min_entity_count: int = 1,
            max_entity_count: int = -1,
            unique_entities: bool = True,
            **kwargs: Any):

        super().__init__(**kwargs)

        self.sub_comp_list = self._extract_sub_components()
        self.state = PrimaryComponent(
            self, self.sub_comp_list, self._default_state_definition)
        self.statistic = Statistic(
            name='statistic', primary_component=self.state,
            default_definition=self._default_statistic_definition)

        self._entity_list = EntityRegister(
            min_entity_count=min_entity_count,
            max_entity_count=max_entity_count,
            unique_entities=unique_entities)

    def _default_state_definition(
            self, _id: Optional[int] = None) -> FeatureArray:
        return FeatureArray(Feature[Any](name='default_state'))

    def _default_statistic_definition(
            self, _id: Optional[int] = None) -> Tuple[FeatureArray, float]:
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
        sub_comp_list: Dict[str, SubComponentInfo] = {}
        for func_name, func in ((f, getattr(self, f).__func__)
                                for f in dir(self)
                                if f.startswith('_sub_comp_')):
            if callable(func):
                keywords = list(func.__code__.co_varnames)

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

    def load(
            self, filename: str,
            path: Optional[Union[str, pathlib.PurePath]]) -> None:

        super().load(filename, path=path)

        self.state.object_ref = self
        self.statistic.set_primary_component(self.state)
        self.state.set_default_definition(self._default_state_definition)
        self.statistic.set_default_definition(
            self._default_statistic_definition)

    def save(
            self, filename: Optional[str] = None,
            path: Optional[Union[str, pathlib.PurePath]] = None,
            data_to_save: Optional[Tuple[str, ...]] = None
    ) -> Tuple[pathlib.PurePath, str]:

        object_ref_temp, self.state.object_ref = self.state.object_ref, None
        state_default, self.state._default = self.state._default, None

        prim_comp, self.statistic._primary_component = (  # type: ignore
            self.statistic._primary_component, None)
        statistic_default, self.statistic._default = (
            self.statistic._default, None)

        try:
            f, p = super().save(filename, path=path, data_to_save=data_to_save)
        finally:
            self.state.object_ref = object_ref_temp
            self.state._default = state_default

            self.statistic._primary_component = prim_comp
            self.statistic._default = statistic_default

        return f, p

    def reset(self):
        super().reset()
        self._entity_list.clear()
