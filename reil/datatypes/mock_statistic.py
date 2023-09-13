from __future__ import annotations
from collections import defaultdict

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd

from reil.datatypes.components import State
from reil.datatypes.feature import FeatureSet

if TYPE_CHECKING:
    from reil.stateful import Stateful


class MockStatistic:
    '''
    A component that mocks `Statistic` class, and uses another object's
    `Statistic` methods, except for `append` and `aggregate`.
    '''

    def __init__(self, obj: Stateful) -> None:
        '''

        Arguments
        ----------
        obj:
            The object that provides actual `Statistic` capabilities.
        '''
        self._obj = obj
        self._history: dict[
            int | None,
            list[tuple[FeatureSet, float]]] = defaultdict(list)

    def set_object(self, obj: Stateful) -> None:
        self._obj = obj

    def enable(self) -> None:
        return self._obj.statistic.enable()

    def disable(self) -> None:
        return self._obj.statistic.disable()

    def set_state(
            self,
            state: State) -> None:
        return self._obj.statistic.set_state(state)

    def add_definition(
            self,
            name: str,
            fn: Callable[..., Any],
            stat_component: str,
            aggregation_component: str) -> None:
        return self._obj.statistic.add_definition(
            name, fn, stat_component, aggregation_component)

    def __call__(
            self, name: str, _id: int | None = None
    ) -> tuple[FeatureSet, float] | None:
        return self._obj.statistic.__call__(name, _id)

    def append(self, name: str, _id: int | None = None) -> None:
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
        s = self._obj.statistic.__call__(name, _id)
        if s is not None:
            self._history[_id].append(s)

    def aggregate(
            self,
            aggregators: tuple[str, ...] | None = None,
            groupby: tuple[str, ...] | None = None,
            _id: int | None = None,
            reset_history: bool = False):
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
        temp = self._history[_id]
        if not temp:
            return None

        df = pd.DataFrame(
            {'instance_id': i, **x[0].value, 'value': x[1]}  # type: ignore
            for i, x in enumerate(temp))
        temp_group_by = ['instance_id'] if groupby is None else list(groupby)
        grouped_df = df.groupby(temp_group_by)

        def no_change(x: Any) -> Any:
            return x

        result: pd.DataFrame = grouped_df['value'].agg(  # type: ignore
            aggregators or no_change)  # type: ignore

        if reset_history:
            self._history: dict[
                int | None,
                list[tuple[FeatureSet, float]]] = defaultdict(list)

        return result
