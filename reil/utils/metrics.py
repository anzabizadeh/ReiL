from abc import abstractmethod
from typing import Any, Protocol

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
            name='PTTR', y_var_name='INR_history', x_var_name='interval_history',
            acceptable_range=(2.0, 3.0), exclude_first=True)

    def update_state(self, states: tuple[FeatureSet, ...]):
        self.inrs.append(states[0]['INR_history'].value[0])  # type: ignore

        for state in states:
            s: dict[str, list[Any]] = state.value  # type: ignore
            self.inrs.append(s['INR_history'][-1])
            self.intervals.append(s['interval_history'][-1])

        self.intervals.append(1)  # end of the trajectory

    def result(self) -> float:
        return self.in_range_fn._default_function(
            self.inrs, self.intervals[:-1])

    def reset_states(self):
        self.inrs: list[float] = []
        self.intervals: list[int] = []


class INRMetric(MetricProtocol):

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.reset_states()

    def update_state(self, states: tuple[FeatureSet, ...]):
        self.inrs.append(states[0]['INR_history'].value[0])  # type: ignore

        for state in states:
            s: dict[str, list[Any]] = state.value  # type: ignore
            self.inrs.append(s['INR_history'][-1])
            self.intervals.append(s['interval_history'][-1])

        self.intervals.append(1)  # end of the trajectory

    def result(self) -> float:
        _y = self.inrs
        _x = self.intervals[:-1]
        return sum(sum(
            interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x)) / sum(_x)

    def reset_states(self):
        self.inrs: list[float] = []
        self.intervals: list[int] = []
