# -*- coding: utf-8 -*-  pylint: disable=undefined-variable
'''
warfarin class
==============

This `warfarin` class implements a two compartment PK/PD model for warfarin.
'''

import functools
import itertools
from typing import Any, List, Optional, Tuple

from reil.datatypes import FeatureArray
from reil.datatypes.feature import Feature, FeatureGenerator
from reil.subjects import Subject, healthcare
from reil.utils import reil_functions


class Warfarin(Subject):
    '''
    A warfarin subject based on Hamberg's two compartment PK/PD model.
    '''

    def __init__(self,
                 patient: healthcare.Patient,
                 dose_range: Tuple[float, float] = (0.0, 15.0),
                 interval_range: Tuple[int, int] = (1, 28),
                 max_day: int = 90,
                 **kwargs: Any):
        '''
        Arguments
        ---------
        patient:
            A patient object that generates new patients and models
            interaction between dose and INR.

        action_generator:
            An `ActionGenerator` object with 'dose' and
            'interval' components.

        max_day:
            Maximum duration of each trial.

        Raises
        ------
        ValueError
            action_generator should have a "dose" component.

        ValueError
            action_generator should have an "interval" component.
        '''

        super().__init__(max_agent_count=1, **kwargs)

        self._patient = patient
        if not self._patient:
            return

        self._dose_range = dose_range
        self._interval_range = interval_range

        self._max_day = max_day

        self.feature_gen_set = {
            name: FeatureGenerator.numerical(
                name=name, lower=lower, upper=upper)
            for name, lower, upper in (
                ('INR_history', 0.0, 15.0),
                ('daily_INR_history', 0.0, 15.0),
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
        patient_basic = (('age', {}), ('CYP2C9', {}),
                         ('VKORC1', {}), ('sensitivity', {}))
        patient_extra = (('weight', {}), ('height', {}),
                         ('gender', {}), ('tobaco', {}),
                         ('amiodarone', {}), ('fluvastatin', {}))

        self.state.add_definition('day',
                                  ('day', {}))

        self.state.add_definition('patient_basic',
                                  *patient_basic)

        self.state.add_definition('patient',
                                  *patient_basic,
                                  *patient_extra)

        self.state.add_definition('patient_w_dosing',
                                  *patient_basic,
                                  *patient_extra,
                                  ('day', {}),
                                  ('dose_history', {'length': -1}),
                                  ('INR_history', {'length': -1}),
                                  ('interval_history', {'length': -1}))

        for i in (1, 5, 10):
            self.state.add_definition(f'patient_w_dosing_{i:02}',
                                      *patient_basic,
                                      *patient_extra,
                                      ('day', {}),
                                      ('dose_history', {'length': i}),
                                      ('INR_history', {'length': i}),
                                      ('interval_history', {'length': i}))

        self.state.add_definition('patient_w_full_dosing',
                                  *patient_basic,
                                  *patient_extra,
                                  ('day', {}),
                                  ('daily_dose_history', {'length': -1}),
                                  ('daily_INR_history', {'length': -1}),
                                  ('interval_history', {'length': -1}))

        self.state.add_definition('daily_INR',
                                  ('daily_INR_history', {'length': -1}))

        self.state.add_definition('Measured_INR_2',
                                  ('INR_history', {'length': 2}),
                                  ('interval_history', {'length': 1}))

        self.state.add_definition('INR_within_2',
                                  ('daily_INR_history', {'length': -1}))

    def _generate_reward_defs(self):
        reward_sq_dist = reil_functions.NormalizedSquareDistance(
            name='sq_dist', arguments=('daily_INR_history',),  # type: ignore
            length=-1, multiplier=-1.0, retrospective=True, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=True)

        reward_sq_dist_interpolation = reil_functions.NormalizedSquareDistance(
            name='sq_dist_interpolation',
            arguments=('INR_history', 'interval_history'),  # type: ignore
            length=2, multiplier=-1.0, retrospective=True, interpolate=True,
            center=2.5, band_width=1.0, exclude_first=True)

        reward_PTTR = reil_functions.PercentInRange(
            name='PTTR', arguments=('daily_INR_history',),  # type: ignore
            length=-1, multiplier=-1.0, retrospective=True, interpolate=False,
            acceptable_range=(2, 3), exclude_first=True)

        reward_PTTR_interpolation = reil_functions.PercentInRange(
            name='PTTR',
            arguments=('INR_history', 'interval_history'),  # type: ignore
            length=2, multiplier=-1.0, retrospective=True, interpolate=True,
            acceptable_range=(2, 3), exclude_first=True)

        self.reward.add_definition(
            'no_reward', lambda _: 0.0, 'Measured_INR_2')

        self.reward.add_definition(
            'sq_dist_exact', reward_sq_dist, 'INR_within_2')

        self.reward.add_definition(
            'sq_dist_interpolation', reward_sq_dist_interpolation,
            'Measured_INR_2')

        self.reward.add_definition(
            'PTTR_exact', reward_PTTR, 'INR_within_2')

        self.reward.add_definition(
            'PTTR_interpolation', reward_PTTR_interpolation, 'Measured_INR_2')

    def _generate_statistic_defs(self):
        statistic_PTTR = reil_functions.PercentInRange(
            name='PTTR', arguments=('daily_INR_history',),  # type: ignore
            length=-1, multiplier=1.0, retrospective=True, interpolate=False,
            acceptable_range=(2, 3), exclude_first=True)

        self.statistic.add_definition(
            'PTTR_exact_basic', statistic_PTTR, 'daily_INR', 'patient_basic')

        self.statistic.add_definition(
            'PTTR_exact', statistic_PTTR, 'daily_INR', 'patient')

    def _generate_action_defs(self):

        dose_gen = self.feature_gen_set['dose']
        interval_gen = self.feature_gen_set['interval']

        def _actions(dose_values, interval_values):
            actions = itertools.product(
                (dose_gen(vi)
                 for vi in dose_values),
                (interval_gen(vi)
                 for vi in interval_values)
            )

            return tuple(FeatureArray(a) for a in actions)

        caps = (5, 10, 15)
        dose = {cap: tuple(self.generate_dose_values(0.0, cap, 0.5))
                for cap in caps}

        int_fixed = {i: (i,) for i in (1, 2, 3, 7)}
        int_weekly = (7, 14, 21, 28)
        int_free = tuple(range(1, 28))

        dose_int_fixed = {(d[0], i[0]): _actions(d[1], i[1])
                          for d, i in itertools.product(
                              dose.items(), int_fixed.items())
                          }

        dose_int_free = {k: _actions(v, int_free)
                         for k, v in dose.items()}

        dose_int_weekly = {k: _actions(v, int_weekly)
                           for k, v in dose.items()}

        def _237(f: FeatureArray, cap):
            day = f['day'].value
            if day == 0:
                return dose_int_fixed[cap, 2]
            elif day == 2:
                return dose_int_fixed[15, 3]
            elif day >= 5:
                return dose_int_fixed[15, 7]
            else:
                raise ValueError(f'Wrong day: {day}.')

        for cap in caps:
            self.possible_actions.add_definition(
                f'daily_{cap:02}', lambda _: dose_int_fixed[cap, 1], 'day')

            self.possible_actions.add_definition(
                f'237_{cap:02}', functools.partial(_237, cap=cap), 'day')

            self.possible_actions.add_definition(
                f'free_{cap:02}', lambda _: dose_int_free[cap], 'day')

            self.possible_actions.add_definition(
                f'weekly_{cap:02}', lambda _: dose_int_weekly[cap], 'day')

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

    # def possible_actions(
    #         self, _id: Optional[int] = None) -> Tuple[FeatureArray, ...]:
    #     return self._action_generator.possible_actions(
    #         self.state('default', _id))

    def take_effect(self,
                    action: FeatureArray,
                    _id: int = 0) -> None:
        Subject.take_effect(self, action, _id)
        current_dose = float(action.value['dose'])
        current_interval = min(int(action.value['interval']),
                               self._max_day - self._day)

        INRs_temp = self._patient.model(
            dose={i: current_dose
                  for i in range(self._day, self._day + current_interval)},
            measurement_days=list(
                range(self._day + 1, self._day + current_interval + 1)))['INR']

        self._decision_points_dose_history[self._decision_points_index] = \
            current_dose
        self._decision_points_interval_history[self._decision_points_index] = \
            current_interval
        self._decision_points_index += 1

        day_temp = self._day
        self._day += current_interval

        self._full_dose_history[day_temp:self._day] = \
            [current_dose] * current_interval
        self._full_INR_history[day_temp + 1:self._day + 1] = INRs_temp

        self._decision_points_INR_history[self._decision_points_index] = \
            self._full_INR_history[self._day]

    def reset(self) -> None:
        Subject.reset(self)
        self._patient.generate()

        self._day: int = 0
        self._full_INR_history = [0.0] * self._max_day
        self._full_dose_history = [0.0] * self._max_day
        self._decision_points_INR_history = [0.0] * (self._max_day + 1)
        self._decision_points_dose_history = [0.0] * self._max_day
        self._decision_points_interval_history: List[int] = [1] * self._max_day
        self._decision_points_index: int = 0

        self._full_INR_history[0] = self._patient.model(
            measurement_days=[0])['INR'][-1]
        self._decision_points_INR_history[0] = self._full_INR_history[0]

    def _default_state_definition(
            self, _id: Optional[int] = None) -> FeatureArray:
        patient_features = self._patient.feature_set
        return FeatureArray([
            patient_features['age'],
            patient_features['CYP2C9'],
            patient_features['VKORC1']])

    def _numerical_sub_comp(self, name):
        return self._patient.feature_set[name]

    def _categorical_sub_comp(self, name):
        return self._patient.feature_set[name]

    def _sub_comp_age(self, _id: int, **kwargs: Any) -> Feature:
        return self._numerical_sub_comp('age')

    def _sub_comp_weight(self, _id: int, **kwargs: Any) -> Feature:
        return self._numerical_sub_comp('weight')

    def _sub_comp_height(self, _id: int, **kwargs: Any) -> Feature:
        return self._numerical_sub_comp('height')

    def _sub_comp_gender(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('gender')

    def _sub_comp_race(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('race')

    def _sub_comp_tobaco(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('tobaco')

    def _sub_comp_amiodarone(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('amiodarone')

    def _sub_comp_fluvastatin(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('fluvastatin')

    def _sub_comp_CYP2C9(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('CYP2C9')

    def _sub_comp_VKORC1(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('VKORC1')

    def _sub_comp_sensitivity(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('sensitivity')

    def _get_history(
            self, list_name: str, length: int) -> Feature:
        if length == 0:
            raise ValueError('length should be a positive integer, or '
                             '-1 for full length output.')

        if list_name == 'INR_history':
            _list = self._decision_points_INR_history
            index = self._decision_points_index + 1
            filler = 0.0
        elif list_name == 'daily_INR_history':
            _list = self._full_INR_history
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

    def _sub_comp_INR_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Feature:
        return self._get_history('INR_history', length)

    def _sub_comp_interval_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Feature:
        return self._get_history('interval_history', length)

    def _sub_comp_day(self, _id: int, **kwargs: Any) -> Feature:
        return self.feature_gen_set['day'](
            value=self._day if 0 <= self._day < self._max_day else None)

    def _sub_comp_daily_dose_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Feature:
        return self._get_history('daily_dose_history', length)

    def _sub_comp_daily_INR_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Feature:
        return self._get_history('daily_INR_history', length)

    def _sub_comp_INR_within(
            self, _id: int, length: int = 1, **kwargs: Any) -> Feature:
        intervals = self._get_history('interval_history', length).value
        return self._get_history('daily_INR', sum(intervals))  # type: ignore

    # def load(self, filename: str,
    #          path: Optional[Union[str, pathlib.PurePath]] = None) -> None:
    #     '''
    #     Extends super class's method to make sure 'action_generator' resets
    #     if it is part of the 'persistent_attributes'.
    #     '''
    #     super().load(filename, path)
    #     if '_action_generator' in self._persistent_attributes:
    #         self._action_generator.reset()

    def __repr__(self) -> str:
        try:
            temp = ', '.join(''.join(
                (str(k), ': ',
                 ('{:4.2f}' if v.is_numerical else '{}').format(v.value)))
                for k, v in self._patient.feature_set.items())
        except (AttributeError, ValueError, KeyError):
            temp = ''

        return (f'{self.__class__.__qualname__} [{temp}]')
