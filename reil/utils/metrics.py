from abc import abstractmethod
import itertools
from typing import Any, Literal, Protocol

from reil.datatypes.feature import FeatureSet
from reil.utils.reil_functions import PercentInRange, interpolate


class MetricProtocol(Protocol):
    '''
    The base class for all `metric` classes. It has the same main
    functions as tf.keras.Metric, so any keras Metric
    can be a ReiL metric.
    '''

    @abstractmethod
    def __init__(self, name: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update_state(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def result(self):
        raise NotImplementedError

    @abstractmethod
    def reset_states(self):
        raise NotImplementedError


class PTTRMetric(MetricProtocol):
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.reset_states()
        self.in_range_fn = PercentInRange(
            name='PTTR', y_var_name='INR_history', x_var_name='duration_history',
            acceptable_range=(2.0, 3.0), exclude_first=True)

    def update_state(self, states: tuple[FeatureSet, ...]):
        self.inrs.append(states[0]['INR_history'].value[0])  # type: ignore

        for state in states:
            s: dict[str, list[Any]] = state.value  # type: ignore
            self.inrs.append(s['INR_history'][-1])
            self.durations.append(s['duration_history'][-1])

        self.durations.append(1)  # end of the trajectory

    def result(self) -> float:
        return self.in_range_fn._default_function(
            self.inrs, self.durations[:-1])

    def reset_states(self):
        self.inrs: list[float] = []
        self.durations: list[int] = []


class INRMetric(MetricProtocol):

    def __init__(
            self, name: str,
            mode: Literal['scalar', 'histogram'] = 'histogram', **kwargs):
        self.name = name
        self._mode = mode
        self.reset_states()

    def update_state(self, states: tuple[FeatureSet, ...]):
        self.inrs.append(states[0]['INR_history'].value[0])  # type: ignore

        for state in states:
            s: dict[str, list[Any]] = state.value  # type: ignore
            self.inrs.append(s['INR_history'][-1])
            self.durations.append(s['duration_history'][-1])

        self.durations.append(1)  # end of the trajectory

    def result(self) -> float | tuple[float, ...]:
        _y = self.inrs
        _x = self.durations[:-1]
        if self._mode == 'scalar':
            return sum(sum(
                interpolate(_y[i], _y[i + 1], x_i))
                for i, x_i in enumerate(_x)) / sum(_x)
        return tuple(itertools.chain.from_iterable((
            interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x)))

    def reset_states(self):
        self.inrs: list[float] = []
        self.durations: list[int] = []


class ActionMetric(MetricProtocol):
    def __init__(self, name: str, component_index: int, **kwargs):
        self.name = name
        self._component_index = component_index
        self.reset_states()

    def update_state(self, action_index_list: tuple[tuple[int, ...], ...]):
        self._records.extend(
            [a[self._component_index] for a in action_index_list])

    def result(self) -> list[int]:
        return self._records

    def reset_states(self):
        self._records: list[int] = []
