# -*- coding: utf-8 -*-  pylint: disable=undefined-variable
'''
HealthSubject class
===================

This `HealthSubject` class implements interaction with patients.
'''

from typing import Any, Dict, List, Optional, Tuple, Union

from reil.datatypes.feature import (MISSING, Feature, FeatureGenerator,
                                    FeatureGeneratorSet)
from reil.healthcare.patient import Patient
from reil.serialization import deserialize, serialize
from reil.subjects.subject import Subject


class HealthSubject(Subject):
    '''
    A HealthSubject subject class with a patient.
    '''

    def __init__(
            self,
            patient: Optional[Patient],
            measurement_name: str,
            measurement_range: Tuple[float, float],
            max_day: int,
            interval_range: Tuple[int, int],
            backfill: bool,
            interval_step: int = 1,
            **kwargs: Any):
        '''
        Arguments
        ---------
        patient:
            An optional patient object that generates new patients and
            computes measurements by calling their `model` method.

        measurement_name:
            Name of the measured value after dose administration, e.g. BGL,
            INR, etc.

        measurement_range:
            A tuple that shows the minimum and maximum possible values of each
            measurement.

        max_day:
            Maximum duration of each trial.

        interval_range:
            A tuple that shows the minimum and maximum duration of each measurement
            decision.

        backfill:
            If `True`, components such as measurement_history
            will be backfilled with the first value in the list.

        interval_step:
            Minimum number of days between two measurements.
        '''

        super().__init__(max_entity_count=1, **kwargs)

        self._patient = patient
        if not self._patient:
            return

        self._interval_range = interval_range
        self._interval_step = interval_step
        self._measurement_range = measurement_range
        self._measurement_name = measurement_name

        self._max_day = max_day
        self._backfill = backfill

        self.feature_gen_set = FeatureGeneratorSet(
            FeatureGenerator.continuous(
                name=name, lower=lower, upper=upper)
            for name, lower, upper in (
                (f'{self._measurement_name}_history',
                 *self._measurement_range),
                (f'daily_{self._measurement_name}_history',
                 *self._measurement_range))
        ) + FeatureGeneratorSet(
            FeatureGenerator.discrete(
                name=name, lower=lower, upper=upper, step=step)
            for name, lower, upper, step in (
                ('interval_history', *self._interval_range,
                    self._interval_step),
                ('interval', *self._interval_range, self._interval_step),
            )
        ) + FeatureGenerator.discrete(
            name='day', lower=0, upper=self._max_day - 1,
            generator=lambda _: None)

        self.action_gen_set = FeatureGeneratorSet()

        HealthSubject._generate_state_defs(self)

        self._day: int = 0
        self._full_measurement_history = [0.0] * self._max_day
        self._decision_points_measurement_history = [0.0] * (self._max_day + 1)
        self._decision_points_interval_history: List[int] = [1] * self._max_day
        self._decision_points_index: int = 0

        self._full_measurement_history[0] = self._patient.model(
            measurement_days=[0])[self._measurement_name][-1]
        self._decision_points_measurement_history[0] = \
            self._full_measurement_history[0]

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        patient_info = config['patient']
        if patient_info is not None:
            config['patient'] = deserialize(patient_info)

        instance = super().from_config(config)

        return instance

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        if self._patient is None:
            config.update({'patient': None})
        else:
            config.update({'patient': serialize(self._patient)})
        config.update(dict(
            measurement_name=self._measurement_name,
            measurement_range=self._measurement_range,
            interval_range=self._interval_range,
            interval_step=self._interval_step,
            max_day=self._max_day
        ))

        return config

    def _generate_state_defs(self):
        if 'day' not in self.state.definitions:
            self.state.add_definition(
                'day', ('day', {}))

    @classmethod
    def _empty_instance(cls):
        return cls(None, '', (0, 1), 1, (1, 2), True)

    @staticmethod
    def generate_interval_values(
            min_interval: int = 1,
            max_interval: int = 28,
            interval_increment: int = 1) -> List[int]:

        return list(range(min_interval, max_interval, interval_increment))

    def is_terminated(self, _id: Optional[int] = None) -> bool:
        return self._day >= self._max_day

    def reset(self) -> None:
        Subject.reset(self)

        self._day: int = 0
        self._full_measurement_history = [0.0] * self._max_day
        self._decision_points_measurement_history = [0.0] * (self._max_day + 1)
        self._decision_points_interval_history: List[int] = [1] * self._max_day
        self._decision_points_index: int = 0

        if self._patient is not None:
            self._patient.generate()
            self._full_measurement_history[0] = self._patient.model(
                measurement_days=[0])[self._measurement_name][-1]
        self._decision_points_measurement_history[0] = \
            self._full_measurement_history[0]

    def _numerical_sub_comp(self, name: str) -> Feature:
        if self._patient is None:
            raise RuntimeError('Patient is not set.')

        return self._patient.feature_set[name]

    def _categorical_sub_comp(self, name: str, missing: bool = False) -> Feature:
        if self._patient is None:
            raise RuntimeError('Patient is not set.')

        if missing:
            self._patient.feature_gen_set[name](MISSING)

        return self._patient.feature_set[name]

    def _get_history(
            self, list_name: str, length: int, backfill: bool = False
    ) -> Feature:
        if length == 0:
            raise ValueError(
                'length should be a positive integer, or '
                '-1 for full length output.')

        filler: Union[float, int]
        _list: Union[List[float], List[int]]
        if list_name == f'{self._measurement_name}_history':
            _list = self._decision_points_measurement_history
            index = self._decision_points_index + 1
            filler = 0.0
        elif list_name == f'daily_{self._measurement_name}_history':
            _list = self._full_measurement_history
            index = self._day + 1
            filler = 0.0
        elif list_name == 'interval_history':
            _list = self._decision_points_interval_history
            index = self._decision_points_index
            filler = int(1)
        else:
            raise ValueError(f'Unknown list_name: {list_name}.')

        if length == -1:
            result = _list[:index]
        else:
            if length > index:
                i1, i2 = length - index, 0
            else:
                i1, i2 = 0, index - length
            if backfill:
                filler = _list[i2]
            result = [filler] * i1 + _list[i2:index]

        return self.feature_gen_set[list_name](tuple(result))

    def _sub_comp_measurement_history(
            self, _id: int, length: int = 1, backfill: bool = True,
            **kwargs: Any
    ) -> Feature:
        return self._get_history(  # type: ignore
            f'{self._measurement_name}_history', length, backfill)

    def _sub_comp_interval_history(
            self, _id: int, length: int = 1, backfill: bool = False,
            **kwargs: Any
    ) -> Feature:
        return self._get_history(  # type: ignore
            'interval_history', length, backfill)

    def _sub_comp_day(self, _id: int, **kwargs: Any) -> Feature:
        return self.feature_gen_set['day'](  # type: ignore
            value=self._day if 0 <= self._day < self._max_day else None)

    def _sub_comp_daily_measurement_history(
            self, _id: int, length: int = 1, backfill: bool = False,
            **kwargs: Any
    ) -> Feature:
        return self._get_history(  # type: ignore
            f'daily_{self._measurement_name}_history', length, backfill)

    def __repr__(self) -> str:
        temp = 'No Patient'
        if self._patient:
            try:
                temp = ', '.join(''.join(
                    (v.name, ': ',
                     ('{:4.2f}' if v.is_numerical else '{}').format(v.value)))
                    for v in self._patient.feature_set)
            except (AttributeError, ValueError, KeyError):
                pass

        return (f'{self.__class__.__qualname__} [{temp}]')
