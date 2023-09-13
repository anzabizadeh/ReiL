# -*- coding: utf-8 -*-
'''
Stateful class
==============

The base class of all stateful classes in `reil` package.

Methods
-------
state:
    the state of the entity (`agent` or `subject`) as a FeatureSet. Different
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

from typing import Any

from reil import reilbase
from reil.datatypes.components import State, Statistic, SubComponentInfo
from reil.datatypes.entity_register import EntityRegister
from reil.datatypes.feature import Feature, FeatureSetDumper, NoneFeature
from reil.utils.metrics import MetricProtocol
from reil.utils.tf_utils import SummaryWriter


class Stateful(reilbase.ReilBase):
    '''
    The base class of all stateful classes in the `ReiL` package.
    '''

    def __init__(
            self,
            min_entity_count: int = 1,
            max_entity_count: int = -1,
            unique_entities: bool = True,
            state_dumper: FeatureSetDumper | None = None,
            summary_writer: SummaryWriter | None = None,
            **kwargs: Any):
        '''
        Initialize the stateful object and extract sub_components.

        Arguments
        ---------
        min_entity_count:
            minimum number of entities that can be registered.

        max_entity_count:
            maximum number of entities that can be registered. If -1, there is
            no limit.

        unique_entities:
            if True, only one entity can be registered for each `_id`.

        state_dumper:
            the dumper to use in `State` to dump the state value if needed.
            For example, `Single` environment calls the dumper of the subjects
            to dump the subject's state.

        summary_writer:
            the summary writer to use to dump summaries. `Stateful` does not
            use the `summary_writer`, but provides it for the children
            classes. For example, `Agent` uses it to dump the agent's
            performance metrics.

        kwargs:
            additional arguments to be passed to the base class.

        Notes
        -----
        During the initialization, `_extract_sub_components` is called to
        extract all sub components.

        Each `Stateful` object has a `state` property (hence "stateful") and
        a `statistic` property.

        The `state` property if a `State` object that can dynamically compose
        and return the state of the object using the state name and the object's
        `sub_component`s. Each derived class from `Stateful` can introduce its
        own `sub_component`s and state definitions. This base class only defines
        the `none` state.

        The `statistic` property is a `Statistic` object that
        can dynamically compute the value of the given statistic for the entity
        `_id` based on the statistic definition `name`.
        '''

        super().__init__(**kwargs)

        self._metrics: dict[str, MetricProtocol] = {}
        self._computed_metrics: dict[str, float] = {}
        self._summary_writer = summary_writer

        sub_comp_list = self._extract_sub_components()
        self.state = State(
            self, sub_comp_list,
            dumper=state_dumper, pickle_stripped=True)

        self.statistic = Statistic(
            name='statistic', state=self.state,
            pickle_stripped=True)

        self._entity_list = EntityRegister(
            min_entity_count=min_entity_count,
            max_entity_count=max_entity_count,
            unique_entities=unique_entities)

    def _generate_state_defs(self) -> None:
        '''
        Generate the state definitions.

        Notes
        -----
        See the notes on `_state_def_reference()`.
        '''
        if 'none' not in self.state.definitions:
            self.state.add_definition('none', ('none', {}))

    def _state_def_reference(
            self, name: str) -> tuple[tuple[str, dict[str, Any]], ...] | None:
        '''
        Return the state definition reference for the given state name.

        Arguments
        ---------
        name:
            the name of the state definition.

        Notes
        -----
        `Stateful` provides two ways to define a state: `_generate_state_defs()`
        and `_state_def_reference()`. The first one is called at the time of
        object initialization. Hence, the object's `state` will have the
        definitions available to use. However, this increases the size of the
        object. The second method is used by the `state` if it does not find the
        definition. Therefore, it defines the state if it is used by the object
        with minimal performance loss (one function call) and without using
        too much memory. The use of `_state_def_reference()` is generally preferred.
        '''
        if name == 'none':
            return (('none', {}),)

    def _generate_statistic_defs(self) -> None:
        '''
        Generate the statistic definitions.
        '''
        raise NotImplementedError

    def _extract_sub_components(self) -> dict[str, SubComponentInfo]:
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
        sub_comp_list: dict[str, SubComponentInfo] = {}
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

    def _sub_comp_none(
            self, _id: int, **kwargs: Any
    ) -> Feature:
        '''
        The sub component shared by all `Stateful` objects akin to `None`.
        '''
        return NoneFeature

    def _update_metrics(self, **kwargs: Any) -> None:
        '''
        Update the metrics.
        '''
        pass

    def register(self, entity_name: str, _id: int | None = None) -> int:
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

    def reset(self):
        '''
        Reset the object to its initial state and clear the entity list.
        '''
        super().reset()
        self._entity_list.clear()

    def __setstate__(self, state: dict[str, Any]) -> None:
        '''
        Set the object state from pickling.

        Arguments
        ---------
        state:
            The object state.
        '''
        super().__setstate__(state)

        self.state.object_ref = self
        try:
            self.statistic.set_state(self.state)
        except ValueError:
            self._logger.warning(
                'Primary component is already set for `statistic` to .'
                f'{self.statistic._state}. Resetting the value!')
            self.statistic._state = self.state
