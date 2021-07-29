# -*- coding: utf-8 -*-
'''
PrimaryComponent and SecondayComponent classes
==============================================

A datatype used to specify entity components, such as `state`, `reward`,
and `statistic`.
'''
from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import (Any, Callable, DefaultDict, Dict, Generic, List, Optional,
                    Tuple, TypeVar, Union)

import pandas as pd
from reil.datatypes.feature import FeatureArray
from reil.datatypes.feature_array_dumper import FeatureArrayDumper

SubComponentInfo = Tuple[Callable[..., Dict[str, Any]], Tuple[str, ...]]

ArgsType = TypeVar('ArgsType', str, Tuple[str, ...], Dict[str, Any])
ComponentReturnType = TypeVar('ComponentReturnType')


@dataclasses.dataclass
class SubComponentInstance(Generic[ArgsType]):
    '''
    A `dataclass` to store an instance of a sub component.

    :meta private:
    '''
    name: str
    args: ArgsType
    fn: Callable[..., Any]


class PrimaryComponent:
    '''
    The datatype to specify primary component, e.g., `state`.
    '''

    def __init__(
            self,
            object_ref: object,
            available_sub_components: Optional[
                Dict[str, SubComponentInfo]] = None,
            default_definition: Optional[Callable[[
                Optional[int]], FeatureArray]] = None,
            dumper: Optional[FeatureArrayDumper] = None,
            pickle_stripped: bool = False):
        '''
        Parameters
        ----------
        available_sub_components:
            A dictionary with sub component names as keys and a tuple of
            function and its argument list as values.
        '''
        self._available_sub_components: Dict[str, SubComponentInfo] = {}
        self._definitions: Dict[str, List[
            SubComponentInstance[Dict[str, Any]]]] = defaultdict(list)

        if available_sub_components is not None:
            self.sub_components = available_sub_components

        self.object_ref = object_ref
        self._default = default_definition
        self._dumper = dumper
        self._pickle_stripped = pickle_stripped

    @property
    def sub_components(self) -> Dict[str, SubComponentInfo]:
        '''Get and set the dictionary of sub components.

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
    def sub_components(self, sub_components: Dict[str, SubComponentInfo]):
        if self._available_sub_components:
            raise ValueError('Available sub components list is already set. '
                             'Cannot modify it.')
        self._available_sub_components = sub_components

    def set_default_definition(
            self, default_definition: Callable[[Optional[int]], FeatureArray]
    ) -> None:
        '''Add a new component definition.

        Parameters
        ----------
        default_definition:
            A function that can optionally accept `_id`, and returns a
            `FeatureArray`.
        '''
        self._default = default_definition

    def add_definition(
            self,
            name: str,
            *sub_components: Tuple[str, Dict[str, Any]]) -> None:
        '''Add a new component definition.

        Parameters
        ----------
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
        _name = name.lower()

        if _name == 'default':
            raise ValueError('Use `set_default_definition` for the default '
                             'definition')

        if _name in self._definitions:
            raise ValueError(f'Definition {name} already exists.')

        unknown_sub_components = set(
            sc for sc, _ in sub_components).difference(
            self._available_sub_components)

        if unknown_sub_components:
            raise ValueError('Unknown sub components: '
                             f'{unknown_sub_components}')

        for sub_comp_name, kwargs in sub_components:
            fn, arg_list = self._available_sub_components[sub_comp_name]

            unknown_keywords = set(kwargs).difference(arg_list)
            if unknown_keywords:
                raise ValueError(
                    f'Unknown keyword argument(s): {unknown_keywords}.')

            self._definitions[_name].append(SubComponentInstance(
                name=sub_comp_name, fn=fn, args=kwargs))

    @property
    def definitions(self):
        '''Return the dictionary of component definitions.

        Returns
        -------
        :
            The dictionary of component definitions.
        '''
        return self._definitions

    def default(self, _id: Optional[int] = None) -> FeatureArray:
        '''
        Generate the default component definition.

        Parameters
        ----------
        _id:
            ID of the caller object

        Returns
        -------
        :
            The component with the default definition.
        '''
        if self._default is None:
            raise AttributeError('Default definition not found.')

        return self._default(_id)

    def __call__(self, name: str, _id: Optional[int] = None) -> FeatureArray:
        '''
        Generate the component based on the specified `name` for the
        specified caller.

        Parameters
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
        if name == 'default':
            try:
                return self.default(_id)
            except AttributeError:
                pass

        if name not in self._definitions:
            raise ValueError(f'Definition {name} not found.')

        return FeatureArray(d.fn(
            self.object_ref, _id=_id, **d.args)  # type: ignore
            for d in self._definitions[name.lower()])

    def dump(
            self, name: str, _id: Optional[int] = None,
            additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
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
            primary_component: Optional[PrimaryComponent] = None,
            default_definition: Optional[Callable[[
                Optional[int]], ComponentReturnType]] = None,
            enabled: bool = True,
            pickle_stripped: bool = False):
        '''

        Parameters
        ----------
        name:
            The name of the secondary component.

        primary_component:
            An instance of a `PrimaryComponent` from which component
            definitions are used.

        default_definition:
            The `default` definition.

        enabled:
            Whether to return the computed value or `None`.
        '''
        self._name = name
        self._primary_component = primary_component
        self._default = default_definition
        self._enabled = enabled
        self._pickle_stripped = pickle_stripped

        self._definitions: Dict[
            str, SubComponentInstance[str]] = defaultdict(None)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def set_primary_component(
            self,
            primary_component: PrimaryComponent) -> None:
        '''Set the primary component.

        Parameters
        ----------
        primary_component:
            An instance of a `PrimaryComponent` from which component
            definitions are used.

        Raises
        ------
        ValueError
            Primary component is already set.
        '''
        if self._primary_component is not None:
            raise ValueError('Primary component is already set. '
                             'Cannot modify it.')

        self._primary_component = primary_component

    def set_default_definition(
            self,
            default_definition: Callable[[Optional[int]], ComponentReturnType]
    ) -> None:
        '''Add a new component definition.

        Parameters
        ----------
        default_definition:
            A function that can optionally accept `_id`, and returns a value.
        '''
        self._default = default_definition

    def add_definition(
            self, name: str, fn: Callable[..., ComponentReturnType],
            primary_component_name: str = 'default') -> None:
        '''
        Add a new component definition.

        Parameters
        ----------
        name:
            The name of the new component.

        fn:
            The function that will receive the primary component instance and
            computes the value of the secondary component.

        primary_component_name:
            The component name that will be used by `fn`.

        Raises
        ------
        ValueError
            Definition already exists for this name.

        ValueError
            Undefined primary component name.
        '''
        _name = name.lower()
        _primary_component_name = primary_component_name.lower()

        if _name == 'default':
            raise ValueError('Use `set_default_definition` for the default '
                             'definition')

        if _name in self._definitions:
            raise ValueError(f'Definition {name} already exists.')

        if self._primary_component is None:
            raise ValueError(
                'Primary component is not defined. '
                'Use `set_primary_component` to specify it.')

        if _primary_component_name not in self._primary_component.definitions:
            raise ValueError(f'Undefined {_primary_component_name}.')

        self._definitions[_name] = SubComponentInstance(
            name=_name,
            fn=fn,
            args=_primary_component_name)

    def default(self, _id: Optional[int] = None) -> ComponentReturnType:
        '''
        Generate the default component definition.

        Parameters
        ----------
        _id:
            ID of the caller object

        Returns
        -------
        :
            The component with the default definition.
        '''
        if self._default is not None:
            return self._default(_id)

        raise AttributeError('Default definition not found.')

    def __call__(
            self, name: str,
            _id: Optional[int] = None) -> Union[ComponentReturnType, None]:
        '''
        Generate the component based on the specified `name` for the
        specified caller.

        Parameters
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

        _name = name.lower()

        if _name == 'default':
            try:
                return self.default(_id)
            except AttributeError:
                pass

        if self._primary_component is None:
            raise ValueError(
                'Primary component is not defined. '
                'Use `set_primary_component` to specify it.')

        try:
            d = self._definitions[_name]
        except KeyError:
            raise ValueError(f'Definition {name} not found.')

        p = self._primary_component(name=d.args, _id=_id)

        return d.fn(p)

    def __getstate__(self):
        if not self._pickle_stripped:
            return self.__dict__

        state = self.__dict__.copy()
        state['_primary_component'] = None
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
            primary_component: Optional[PrimaryComponent] = None,
            default_definition: Optional[Callable[[
                Optional[int]], Tuple[FeatureArray, float]]] = None,
            enabled: bool = True,
            pickle_stripped: bool = False):
        '''

        Parameters
        ----------
        name:
            The name of the secondary component.

        primary_component:
            An instance of a `PrimaryComponent` from which component
            definitions are used.

        default_definition:
            The `default` definition.

        enabled:
            Whether to return the computed value or `None`.
        '''
        self._name = name
        self._primary_component = primary_component
        self._default = default_definition
        self._enabled = enabled
        self._pickle_stripped = pickle_stripped

        self._definitions: Dict[
            str, SubComponentInstance[Tuple[str, str]]] = defaultdict(None)

        self._history: Dict[
            int,
            List[Tuple[FeatureArray, float]]] = DefaultDict(list)
        self._history_none: List[Tuple[FeatureArray, float]] = []

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def set_primary_component(
            self,
            primary_component: PrimaryComponent) -> None:
        '''Set the primary component.

        Parameters
        ----------
        primary_component:
            An instance of a `PrimaryComponent` from which component
            definitions are used.

        Raises
        ------
        ValueError
            Primary component is already set.
        '''
        if self._primary_component is not None:
            raise ValueError('Primary component is already set. '
                             'Cannot modify it.')

        self._primary_component = primary_component

    def set_default_definition(
            self,
            default_definition: Callable[
                [Optional[int]], Tuple[FeatureArray, float]]) -> None:
        '''Add a new component definition.

        Parameters
        ----------
        default_definition:
            A function that can optionally accept `_id`, and returns a
            `FeatureArray`.
        '''
        self._default = default_definition

    def add_definition(
            self, name: str, fn: Callable[..., Any],
            stat_component: str, aggregation_component: str) -> None:
        '''
        Add a new component definition.

        Parameters
        ----------
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
        _name = name.lower()
        _stat_component = stat_component.lower()
        _aggregation_component = aggregation_component.lower()
        if _name == 'default':
            raise ValueError('Use `set_default_definition` for the default '
                             'definition')

        if _name in self._definitions:
            raise ValueError(f'Definition {name} already exists.')

        if self._primary_component is None:
            raise ValueError(
                'Primary component is not defined. '
                'Use `set_primary_component` to specify it.')

        if _stat_component not in self._primary_component.definitions:
            raise ValueError(f'Undefined {_stat_component}.')

        if _aggregation_component not in self._primary_component.definitions:
            raise ValueError(f'Undefined {_aggregation_component}.')

        self._definitions[_name] = SubComponentInstance[Tuple[str, str]](
            name=_name,
            fn=fn,
            args=(_aggregation_component, _stat_component))

    def default(self, _id: Optional[int] = None) -> Tuple[FeatureArray, float]:
        '''
        Generate the default component definition.

        Parameters
        ----------
        _id:
            ID of the caller object

        Returns
        -------
        :
            The component with the default definition.
        '''
        if self._default is not None:
            return self._default(_id)

        raise AttributeError('Default definition not found.')

    def __call__(
            self,
            name: str,
            _id: Optional[int] = None
    ) -> Union[Tuple[FeatureArray, float], None]:
        '''
        Generate the component based on the specified `name` for the
        specified caller.

        Parameters
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

        _name = name.lower()

        if _name == 'default':
            try:
                return self.default(_id)
            except AttributeError:
                pass

        if self._primary_component is None:
            raise ValueError(
                'Primary component is not defined. '
                'Use `set_primary_component` to specify it.')

        try:
            d = self._definitions[_name]
        except KeyError:
            raise ValueError(f'Definition {name} not found.')

        agg, comp_name = d.args

        return (self._primary_component(name=agg, _id=_id),
                d.fn(self._primary_component(name=comp_name, _id=_id)))

    def append(self,
               name: str,
               _id: Optional[int] = None) -> None:
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

    def aggregate(self,
                  aggregators: Optional[Tuple[str, ...]] = None,
                  groupby: Optional[Tuple[str, ...]] = None,
                  _id: Optional[int] = None,
                  reset_history: bool = False):
        temp = self._history_none if _id is None else self._history[_id]
        if not temp:
            return None

        df = pd.DataFrame({'instance_id': i,  # type: ignore
                           **x[0].value,
                           'value': x[1]}
                          for i, x in enumerate(temp))
        temp_group_by = ['instance_id'] if groupby is None else list(groupby)
        grouped_df = df.groupby(temp_group_by)

        def no_change(x: Any) -> Any:
            return x

        result: pd.DataFrame = grouped_df['value'].agg(  # type: ignore
            aggregators or no_change)

        if reset_history:
            self._history: Dict[
                int, List[Tuple[FeatureArray, float]]] = DefaultDict(list)
            self._history_none: List[Tuple[FeatureArray, float]] = []

        return result

    def __getstate__(self):
        if not self._pickle_stripped:
            return self.__dict__

        state = self.__dict__.copy()
        state['_primary_component'] = None
        state['_default'] = None

        return state
