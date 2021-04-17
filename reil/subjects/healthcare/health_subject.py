# -*- coding: utf-8 -*-  pylint: disable=undefined-variable
'''
HealthSubject class
===================

This `HealthSubject` class implements interaction with patients.
'''

from typing import Any, List, Optional, Tuple

from reil.datatypes import FeatureArray
from reil.datatypes.feature import Feature, FeatureGenerator
from reil.subjects import Subject, healthcare


class HealthSubject(Subject):
    '''
    A HealthSubject subject class with a patient.
    '''

    def __init__(self,
                 patient: healthcare.Patient,
                 measurement_name: str,
                 measurement_range: Tuple[float, float],
                 dose_range: Tuple[float, float],
                 interval_range: Tuple[int, int],
                 max_day,
                 **kwargs: Any):
        '''
        Arguments
        ---------
        patient:
            A patient object that generates new patients and models
            interaction between dose and INR.

        measurement_name:
            Name of the measured value after dose administration, e.g. BGL,
            INR, etc.

        dose_range:
            A tuple that shows the minimum and maximum amount of dose of
            the medication.

        interval_range:
            A tuple that shows the minimum and maximum duration of each dosing
            decision.

        measurement_range:
            A tuple that shows the minimum and maximum possible values of each
            measurement.

        max_day:
            Maximum duration of each trial.
        '''

        super().__init__(max_agent_count=1, **kwargs)

        self._patient = patient
        if not self._patient:
            return

        self._dose_range = dose_range
        self._interval_range = interval_range
        self._measurement_range = measurement_range
        self._measurement_name = measurement_name

        self._max_day = max_day

        self.feature_gen_set = {
            name: FeatureGenerator.numerical(
                name=name, lower=lower, upper=upper)
            for name, lower, upper in (
                (f'{self._measurement_name}_history',
                 *self._measurement_range),
                (f'daily_{self._measurement_name}_history',
                 *self._measurement_range),
                ('dose_history', *self._dose_range),
                ('daily_dose_history', *self._dose_range),
                ('interval_history', *self._interval_range),
                ('dose', *self._dose_range),
                ('interval', *self._interval_range),
                ('day', 0, self._max_day - 1)
            )
        }

        self._generate_state_defs()
        self._generate_reward_defs()
        self._generate_statistic_defs()
        self._generate_action_defs()

        self.reset()

    def _generate_state_defs(self):
        self.state.add_definition('day',
                                  ('day', {}))

    def _generate_reward_defs(self):
        self.reward.add_definition(
            'no_reward', lambda _: 0.0, 'day')

    def _generate_statistic_defs(self):
        pass

    def _generate_action_defs(self):
        pass

    @classmethod
    def _empty_instance(cls):
        return cls(None, None)  # type: ignore

    @staticmethod
    def generate_dose_values(min_dose: float = 0.0,
                             max_dose: float = 15.0,
                             dose_increment: float = 0.5) -> List[float]:

        return list(min_dose + x * dose_increment
                    for x in range(
                        int((max_dose - min_dose)/dose_increment) + 1))

    @staticmethod
    def generate_interval_values(min_interval: int = 1,
                                 max_interval: int = 28,
                                 interval_increment: int = 1) -> List[int]:

        return list(range(min_interval, max_interval, interval_increment))

    def is_terminated(self, _id: Optional[int] = None) -> bool:
        return self._day >= self._max_day

    def take_effect(self,
                    action: FeatureArray,
                    _id: int = 0) -> None:
        Subject.take_effect(self, action, _id)
        current_dose = float(action.value['dose'])
        current_interval = min(int(action.value['interval']),
                               self._max_day - self._day)

        measurements_temp = self._patient.model(
            dose={
                i: current_dose
                for i in range(self._day, self._day + current_interval)
            },
            measurement_days=list(
                range(self._day + 1, self._day + current_interval + 1)
            )
        )[self._measurement_name]

        self._decision_points_dose_history[self._decision_points_index] = \
            current_dose
        self._decision_points_interval_history[self._decision_points_index] = \
            current_interval
        self._decision_points_index += 1

        day_temp = self._day
        self._day += current_interval

        self._full_dose_history[day_temp:self._day] = \
            [current_dose] * current_interval
        self._full_measurement_history[day_temp +
                                       1:self._day + 1] = measurements_temp

        self._decision_points_measurement_history[
            self._decision_points_index] = \
            self._full_measurement_history[self._day]

    def reset(self) -> None:
        Subject.reset(self)
        self._patient.generate()

        self._day: int = 0
        self._full_measurement_history = [0.0] * self._max_day
        self._full_dose_history = [0.0] * self._max_day
        self._decision_points_measurement_history = [0.0] * (self._max_day + 1)
        self._decision_points_dose_history = [0.0] * self._max_day
        self._decision_points_interval_history: List[int] = [1] * self._max_day
        self._decision_points_index: int = 0

        self._full_measurement_history[0] = self._patient.model(
            measurement_days=[0])[self._measurement_name][-1]
        self._decision_points_measurement_history[0] = \
            self._full_measurement_history[0]

    def _numerical_sub_comp(self, name):
        return self._patient.feature_set[name]

    def _categorical_sub_comp(self, name):
        return self._patient.feature_set[name]

    def _get_history(
            self, list_name: str, length: int) -> Feature:
        if length == 0:
            raise ValueError('length should be a positive integer, or '
                             '-1 for full length output.')

        if list_name == f'{self._measurement_name}_history':
            _list = self._decision_points_measurement_history
            index = self._decision_points_index + 1
            filler = 0.0
        elif list_name == f'daily_{self._measurement_name}_history':
            _list = self._full_measurement_history
            index = self._day + 1
            filler = 0.0
        elif list_name == 'dose_history':
            _list = self._decision_points_dose_history
            index = self._decision_points_index
            filler = 0.0
        elif list_name == 'daily_dose_history':
            _list = self._full_dose_history
            index = self._day
            filler = 0.0
        elif list_name == 'interval_history':
            _list = self._decision_points_interval_history
            index = self._decision_points_index
            filler = 1
        else:
            raise ValueError(f'Unknown list_name: {list_name}.')

        if length == -1:
            result = _list[:index]
        else:
            if length > index:
                i1, i2 = length - index, 0
            else:
                i1, i2 = 0, index-length
            result = [filler] * i1 + _list[i2:index]  # type: ignore

        return self.feature_gen_set[list_name](result)

    def _sub_comp_dose_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Feature:
        return self._get_history('dose_history', length)

    def _sub_comp_measurement_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Feature:
        return self._get_history(f'{self._measurement_name}_history', length)

    def _sub_comp_interval_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Feature:
        return self._get_history('interval_history', length)

    def _sub_comp_day(self, _id: int, **kwargs: Any) -> Feature:
        return self.feature_gen_set['day'](
            value=self._day if 0 <= self._day < self._max_day else None)

    def _sub_comp_daily_dose_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Feature:
        return self._get_history('daily_dose_history', length)

    def _sub_comp_daily_measurement_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Feature:
        return self._get_history(
            f'daily_{self._measurement_name}_history', length)

    def __repr__(self) -> str:
        try:
            temp = ', '.join(''.join(
                (str(k), ': ',
                 ('{:4.2f}' if v.is_numerical else '{}').format(v.value)))
                for k, v in self._patient.feature_set.items())
        except (AttributeError, ValueError, KeyError):
            temp = ''

        return (f'{self.__class__.__qualname__} [{temp}]')
