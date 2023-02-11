from abc import abstractmethod
from typing import Protocol

from reil.datatypes.feature import FeatureSet


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

    def update_state(self, states: tuple[FeatureSet, ...]):
        in_range_list = [
            2.0 <= state['INR_history'].value[-1] <= 3.0  # type: ignore
            for state in states
        ]
        self.count = len(in_range_list)
        self.in_range = sum(in_range_list)

    def result(self):
        return self.in_range / self.count

    def reset_states(self):
        self.count: int = 0
        self.in_range: int = 0


class INRMetric(MetricProtocol):

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.reset_states()

    def update_state(self, states: tuple[FeatureSet, ...]):
        INRs: list[float] = [
            state['INR_history'].value[-1]  # type: ignore
            for state in states
        ]
        self.count += len(INRs)
        self.INR += sum(INRs)

    def result(self):
        return self.INR / self.count

    def reset_states(self):
        self.count: int = 0
        self.INR = 0.0
