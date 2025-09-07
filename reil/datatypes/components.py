# -*- coding: utf-8 -*-
'''
State, ActionSet, Reward and Statistic classes
==============================================

A datatype used to specify entity components, such as `state`, `reward`,
and `statistic`.
'''
from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import pandas as pd

from reil.datatypes.feature import FeatureGeneratorType, FeatureSet, FeatureSetDumper

SubComponentInfo = tuple[Callable[..., dict[str, Any]], tuple[str, ...]]

ArgsType = TypeVar('ArgsType', str, tuple[str, ...], dict[str, Any])
ComponentReturnType = TypeVar('ComponentReturnType')

PrimaryDefRefType = Callable[[str], tuple[tuple[str, dict[str, Any]], ...] | None]
SecondaryDefRefType = Callable[[str], tuple[Callable[..., ComponentReturnType], str] | None]


@dataclasses.dataclass
class SubComponentInstance(Generic[ArgsType]):
    '''
    A `dataclass` to store an instance of a sub component.

    :meta private:
    '''
    name: str
    args: ArgsType
    fn: Callable[..., Any]


class State:
    '''
    The datatype to specify `state`.
    '''

    def __init__(
            self,
            object_ref: object,
            available_sub_components: dict[str, SubComponentInfo] | None = None,
            dumper: FeatureSetDumper | None = None,
            pickle_stripped: bool = False):
        '''
        Arguments
        ----------
        object_ref:
            The object to be used as reference for the state.

        available_sub_components:
            A dictionary with sub component names as keys and a tuple of
            function and its argument list as values.

        dumper:
            A dumper to dump the feature set.

        pickle_stripped:
            Whether to pickle the stripped object or not.

        Notes
        -----
        The `object_ref` should be pickleable for `pickle_stripped` to be
        `False`. Otherwise, the object cannot be reconstructed at time of
        unpickling.
        '''
        self._available_sub_components: dict[str, SubComponentInfo] = {}
        self._definitions: dict[str, list[
            SubComponentInstance[dict[str, Any]]]] = {}
        self._definition_reference_function: PrimaryDefRefType | None = None

        if available_sub_components is not None:
            self.sub_components = available_sub_components

        self.object_ref = object_ref
        self._dumper = dumper
        self._pickle_stripped = pickle_stripped

    @property
    def definitions(self):
        '''
        Return the dictionary of component definitions.

        Returns
        -------
        :
            The dictionary of component definitions.
        '''
        return self._definitions

    @property
    def sub_components(self) -> dict[str, SubComponentInfo]:
        '''
        Get and set the dictionary of sub components.

        Returns
        -------
        :
            Sub components

        Notes
        -----
        Sub components info can only be set once.
        '''
        return self._available_sub_components

    @sub_components.setter
    def sub_components(self, sub_components: dict[str, SubComponentInfo]):
        if self._available_sub_components:
            raise ValueError(
                'Available sub components list is already set. Cannot modify it.')
        self._available_sub_components = sub_components

    def add_definition(
            self,
            name: str,
            *sub_components: tuple[str, dict[str, Any]]) -> None:
        '''
        Add a new component definition.

        Arguments
        ---------
        name:
            The name of the new component.

        sub_components:
            Sub components that form this new component. Each sub component
            should be specified as a tuple. The first item is the name of the
            sub component, and the second item is a dictionary of kwargs and
            values for that sub component.

        Raises
        ------
        ValueError
            Definition already exists for this name.

        ValueError
            Unknown sub component.

        ValueError
            Unknown keyword argument.
        '''
        if self._definitions.get(name, []):
            raise ValueError(f'Definition {name} already exists.')

        unknown_sub_components = set(
            sc for sc, _ in sub_components).difference(
            self._available_sub_components)

        if unknown_sub_components:
            raise ValueError(
                f'Unknown sub components: {unknown_sub_components}')

        self._definitions[name] = []
        for sub_comp_name, kwargs in sub_components:
            fn, arg_list = self._available_sub_components[sub_comp_name]

            unknown_keywords = set(kwargs).difference(arg_list)
            if unknown_keywords:
                raise ValueError(
                    f'Unknown keyword argument(s): {unknown_keywords}.')

            self._definitions[name].append(SubComponentInstance(
                name=sub_comp_name, fn=fn, args=kwargs))

    def definition_reference_function(
            self, f: PrimaryDefRefType, available_definitions: list[str]):
        '''
        Set the function to get the component definition.

        Arguments
        ---------
        f:
            The function to get the component definition.

        available_definitions:
            The list of available component definitions.
        '''
        self._definition_reference_function = f
        for d in set(available_definitions).difference(self._definitions):
            self._definitions[d] = []

    def __call__(self, name: str, _id: int | None = None) -> FeatureSet:
        '''
        Generate the component based on the specified `name` for the
        specified caller.

        Arguments
        ---------
        name:
            The name of the component definition.

        _id:
            ID of the caller.

        Returns
        -------
        :
            The component with the specified definition `name`.

        Raises
        ------
        ValueError
            Definition not found.
        '''
        if not self._definitions.get(name, []):
            if self._definition_reference_function:
                def_args = self._definition_reference_function(name)
                if def_args:
                    self.add_definition(name, *def_args)

        try:
            def_ = self._definitions[name]
        except KeyError:
            raise ValueError(f'Definition {name} not found.')

        return FeatureSet(d.fn(
            self.object_ref, _id=_id, **d.args)  # type: ignore
            for d in def_)

    def dump(
            self, name: str, _id: int | None = None,
            additional_info: dict[str, Any] | None = None
    ) -> None:
        '''
        Dump the component based on the specified `name` for the
        specified caller.

        Arguments
        ---------
        name:
            The name of the component definition.

        _id:
            ID of the caller.

        additional_info:
            Additional information to be dumped.
        '''
        if self._dumper:
            self._dumper.dump(
                component=self.__call__(name, _id),
                additional_info=additional_info)

    def __getstate__(self):
        if not self._pickle_stripped:
            return self.__dict__

        state = self.__dict__.copy()
        state['object_ref'] = None
        state['_default'] = None

        return state


class SecondayComponent(Generic[ComponentReturnType]):
    '''
    The datatype to specify secondary components, e.g. `statistic` and
    `reward`.
    '''

    def __init__(
            self,
            name: str,
            state: State | None = None,
            enabled: bool = True,
            pickle_stripped: bool = False):
        '''

        Arguments
        ---------
        name:
            The name of the secondary component.

        state:
            An instance of a `State` from which component
            definitions are used.

        enabled:
            Whether to return the computed value or `None`.
        '''
        self._name = name
        self._state = state
        self._enabled: bool = enabled
        self._pickle_stripped = pickle_stripped

        self._definitions: dict[
            str, SubComponentInstance[str] | None] = {}
        self._definition_reference_function: SecondaryDefRefType[ComponentReturnType] | None = None

    @property
    def definitions(self):
        '''Return the dictionary of component definitions.

        Returns
        -------
        :
            The dictionary of component definitions.
        '''
        return self._definitions

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def set_state(self, state: State) -> None:
        '''Set the primary component.

        Arguments
        ----------
        state:
            An instance of a `State` from which component
            definitions are used.

        Raises
        ------
        ValueError
            Primary component is already set.
        '''
        if self._state is not None:
            raise ValueError(
                'Primary component is already set. Cannot modify it.')

        self._state = state

    def add_definition(
            self, name: str, fn: Callable[..., ComponentReturnType],
            state_name: str) -> None:
        '''
        Add a new component definition.

        Arguments
        ----------
        name:
            The name of the new component.

        fn:
            The function that will receive the state instance and
            computes the value of the secondary component.

        state_name:
            The state name that will be used by `fn`.

        Raises
        ------
        ValueError
            Definition already exists for this name.

        ValueError
            Undefined primary component name.
        '''
        if self._definitions.get(name, []):
            raise ValueError(f'Definition {name} already exists.')

        if self._state is None:
            raise ValueError(
                'Primary component is not defined. '
                'Use `set_state` to specify it.')

        # if state_name not in self._state.definitions:
        #     raise ValueError(f'Undefined {state_name}.')

        self._definitions[name] = SubComponentInstance(
            name=name, fn=fn, args=state_name)

    def definition_reference_function(
            self, f: SecondaryDefRefType[ComponentReturnType], available_definitions: list[str]):
        self._definition_reference_function = f
        for d in set(available_definitions).difference(self._definitions):
            self._definitions[d] = None

    def __call__(  # noqa: C901
            self, name: str, _id: int | None = None) -> ComponentReturnType | None:
        '''
        Generate the component based on the specified `name` for the
        specified caller.

        Arguments
        ----------
        name:
            The name of the component definition.

        _id:
            ID of the caller.

        Returns
        -------
        :
            The component with the specified definition `name`.

        Raises
        ------
        ValueError
            Definition not found.
        '''
        if not self._enabled:
            return None

        if self._definitions.get(name) is None:
            if self._definition_reference_function:
                def_args = self._definition_reference_function(name)
                if def_args:
                    self.add_definition(name, *def_args)

        d = self._definitions.get(name)

        if d is None:
            raise ValueError(f'Definition {name} not found.')

        if self._state is None:
            raise ValueError(
                'State is not defined. '
                'Use `set_state` to specify it.')

        p = self._state(name=d.args, _id=_id)

        return d.fn(p)

    def __getstate__(self):
        if not self._pickle_stripped:
            return self.__dict__

        state = self.__dict__.copy()
        state['_state'] = None
        state['_default'] = None

        return state


class Statistic:
    '''
    A component similar to `SecondaryComponent`, but with history and
    aggregator.
    '''

    def __init__(
            self,
            name: str,
            state: State | None = None,
            enabled: bool = True,
            pickle_stripped: bool = False):
        '''

        Arguments
        ----------
        name:
            The name of the secondary component.

        state:
            An instance of a `State` from which component
            definitions are used.

        default_definition:
            The `default` definition.

        enabled:
            Whether to return the computed value or `None`.
        '''
        self._name = name
        self._state = state
        self._enabled = enabled
        self._pickle_stripped = pickle_stripped

        self._definitions: dict[
            str, SubComponentInstance[tuple[str, str]] | None] = {}
        self._definition_reference_function: Callable[
            [str], tuple[Callable[..., Any], str, str] | None] | None = None

        self._history: dict[
            int, list[tuple[FeatureSet, float]]] = defaultdict(list)
        self._history_none: list[tuple[FeatureSet, float]] = []

    @property
    def definitions(self):
        '''
        Return the dictionary of component definitions.

        Returns
        -------
        :
            The dictionary of component definitions.
        '''
        return self._definitions

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def set_state(
            self,
            state: State) -> None:
        '''
        Set the primary component.

        Arguments
        ---------
        state:
            An instance of a `State` from which component
            definitions are used.

        Raises
        ------
        ValueError
            Primary component is already set.
        '''
        if self._state is not None:
            raise ValueError(
                'Primary component is already set. Cannot modify it.')

        self._state = state

    def add_definition(
            self, name: str, fn: Callable[..., Any],
            stat_component: str, aggregation_component: str) -> None:
        '''
        Add a new component definition.

        Arguments
        ---------
        name:
            The name of the new component.

        fn:
            The function that will receive the primary component instance and
            computes the value of the secondary component.

        stat_component:
            The component name that will be used by `fn`.

        aggregation_component:
            The component name that will be used to do aggregation.

        Raises
        ------
        ValueError
            Definition already exists for this name.

        ValueError
            Undefined primary component name.
        '''
        if self._definitions.get(name, []):
            raise ValueError(f'Definition {name} already exists.')

        if self._state is None:
            raise ValueError(
                'Primary component is not defined. '
                'Use `set_state` to specify it.')

        if stat_component not in self._state.definitions:
            raise ValueError(f'Undefined {stat_component}.')

        if aggregation_component not in self._state.definitions:
            raise ValueError(f'Undefined {aggregation_component}.')

        self._definitions[name] = SubComponentInstance[tuple[str, str]](
            name=name, fn=fn, args=(aggregation_component, stat_component))

    def definition_reference_function(
        self,
        f: Callable[
            [str], tuple[Callable[..., Any], str, str] | None],
        available_definitions: list[str]
    ):
        self._definition_reference_function = f
        for d in set(available_definitions).difference(self._definitions):
            self._definitions[d] = None

    def __call__(
            self, name: str, _id: int | None = None
    ) -> tuple[FeatureSet, float] | None:
        '''
        Generate the component based on the specified `name` for the
        specified caller.

        Arguments
        ----------
        name:
            The name of the component definition.

        _id:
            ID of the caller.

        Returns
        -------
        :
            The component with the specified definition `name`.

        Raises
        ------
        ValueError
            Definition not found.
        '''
        if not self._enabled:
            return None

        if self._definitions.get(name) is None:
            if self._definition_reference_function:
                def_args = self._definition_reference_function(name)
                if def_args:
                    self.add_definition(name, *def_args)

        d = self._definitions.get(name)

        if d is None:
            raise ValueError(f'Definition {name} not found.')

        if self._state is None:
            raise ValueError(
                'Primary component is not defined. '
                'Use `set_state` to specify it.')

        agg, comp_name = d.args

        return (self._state(name=agg, _id=_id),
                d.fn(self._state(name=comp_name, _id=_id)))

    def append(self,
               name: str,
               _id: int | None = None) -> None:
        '''
        Generate the stat and append it to the history.

        Arguments
        ---------
        name:
            The name of the component definition.

        _id:
            ID of the caller.

        Raises
        ------
        ValueError
            Definition not found.
        '''
        s = self.__call__(name, _id)
        if s is not None:
            if _id is None:
                self._history_none.append(s)
            else:
                self._history[_id].append(s)

    def aggregate(
            self,
            aggregators: tuple[str, ...] | None = None,
            groupby: tuple[str, ...] | None = None,
            _id: int | None = None,
            reset_history: bool = False,
            n: int | None = None):
        '''
        Aggregate the history of the component.

        Arguments
        ---------
        aggregators:
            The aggregation function to use.

        groupby:
            The column names to group by.

        _id:
            ID of the caller.

        reset_history:
            Whether to reset the history after aggregation.

        n:
            The number of instances to aggregate.

        Returns
        -------
        :
            The aggregated result.
        '''
        temp = self._history_none if _id is None else self._history[_id]
        if not temp:
            return None

        if n is not None:
            temp = temp[-n:]

        df = pd.DataFrame(
            {'instance_id': i,  # type: ignore
             **x[0].value,
             'value': x[1]}
            for i, x in enumerate(temp))
        temp_group_by = ['instance_id'] if groupby is None else list(groupby)
        grouped_df = df.groupby(temp_group_by)

        def no_change(x: Any) -> Any:
            return x

        result: pd.DataFrame = grouped_df['value'].agg(  # type: ignore
            aggregators or no_change)  # type: ignore

        if reset_history:
            self._history: dict[
                int, list[tuple[FeatureSet, float]]] = defaultdict(list)
            self._history_none: list[tuple[FeatureSet, float]] = []

        return result

    def __getstate__(self):
        if not self._pickle_stripped:
            return self.__dict__

        state = self.__dict__.copy()
        state['_state'] = None
        state['_default'] = None

        return state


ActionSet = SecondayComponent[FeatureGeneratorType]
Reward = SecondayComponent[float]
