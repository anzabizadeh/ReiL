from typing import (Any, Callable, DefaultDict, Dict, List, Optional, Tuple,
                    Union)

import pandas as pd
from reil.datatypes.components import PrimaryComponent
from reil.datatypes.feature import FeatureArray
from reil.stateful import Stateful


class MockStatistic:
    '''
    A component that mocks `Statistic` class, and uses another object's
    `Statistic` methods, except for `append` and `aggregate`.
    '''

    def __init__(self, obj: Stateful) -> None:
        '''

        Parameters
        ----------
        obj:
            The object that provides actual `Statistic` capabilities.
        '''
        self._obj = obj
        self._history: Dict[
            int,
            List[Tuple[FeatureArray, float]]] = DefaultDict(list)
        self._history_none: List[Tuple[FeatureArray, float]] = []

    def set_object(self, obj: Stateful) -> None:
        self._obj = obj

    def enable(self) -> None:
        return self._obj.statistic.enable()

    def disable(self) -> None:
        return self._obj.statistic.disable()

    def set_primary_component(
            self,
            primary_component: PrimaryComponent) -> None:
        return self._obj.statistic.set_primary_component(primary_component)

    def add_definition(
            self,
            name: str,
            fn: Callable[..., Any],
            stat_component: str,
            aggregation_component: str) -> None:
        return self._obj.statistic.add_definition(
            name, fn, stat_component, aggregation_component)

    def default(self, _id: Optional[int] = None) -> Tuple[FeatureArray, float]:
        return self._obj.statistic.default(_id)

    def __call__(
            self,
            name: str,
            _id: Optional[int] = None
    ) -> Union[Tuple[FeatureArray, float], None]:
        return self._obj.statistic.__call__(name, _id)

    def append(
            self,
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
        s = self._obj.statistic.__call__(name, _id)
        # print(s[0].value, s[1])
        if s is not None:
            if _id is None:
                self._history_none.append(s)
            else:
                self._history[_id].append(s)

    def aggregate(
            self,
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
