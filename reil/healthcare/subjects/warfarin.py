# -*- coding: utf-8 -*-  pylint: disable=undefined-variable
'''
warfarin class
==============

This `warfarin` class implements a two compartment PK/PD model for warfarin.
'''

import functools
import itertools
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from reil.datatypes.feature import Feature, FeatureArray
from reil.healthcare.patient import Patient
from reil.healthcare.subjects.health_subject import HealthSubject
from reil.utils import reil_functions


patient_basic: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ('age', {}), ('CYP2C9', {}),
    ('VKORC1', {}), ('sensitivity', {})
)
patient_extra: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ('weight', {}), ('height', {}),
    ('gender', {}), ('race', {}), ('tobaco', {}),
    ('amiodarone', {}), ('fluvastatin', {})
)

state_definitions: Dict[str, Tuple[Tuple[str, Dict[str, Any]], ...]] = {
    'age': (('age', {}),),
    'patient_basic': patient_basic,
    'patient': (*patient_basic, *patient_extra),
    'patient_w_dosing': (
        *patient_basic, *patient_extra,
        ('day', {}),
        ('dose_history', {'length': -1}),
        ('INR_history', {'length': -1}),
        ('interval_history', {'length': -1})),
    'patient_for_baseline': (
        *patient_basic, *patient_extra,
        ('day', {}),
        ('dose_history', {'length': 4}),
        ('INR_history', {'length': 4}),
        ('interval_history', {'length': 4})),

    **{
        f'patient_w_dosing_{i:02}': (
            *patient_basic,
            ('day', {}),
            ('dose_history', {'length': i}),
            ('INR_history', {'length': i}),
            ('interval_history', {'length': i}))
        for i in range(1, 10)},

    'patient_w_full_dosing': (
        *patient_basic, *patient_extra,
        ('day', {}),
        ('daily_dose_history', {'length': -1}),
        ('daily_INR_history', {'length': -1}),
        ('interval_history', {'length': -1})),

    'daily_INR': (('daily_INR_history', {'length': -1}),),

    'Measured_INR_2': (
        ('INR_history', {'length': 2}),
        ('interval_history', {'length': 1})),

    'INR_within_2': (('daily_INR_history', {'length': -1}),),
}

reward_sq_dist = reil_functions.NormalizedSquareDistance(
    name='sq_dist', y_var_name='daily_INR_history',
    length=-1, multiplier=-1.0, retrospective=True, interpolate=False,
    center=2.5, band_width=1.0, exclude_first=True)

reward_sq_dist_interpolation = reil_functions.NormalizedSquareDistance(
    name='sq_dist_interpolation',
    y_var_name='INR_history', x_var_name='interval_history',
    length=2, multiplier=-1.0, retrospective=True, interpolate=True,
    center=2.5, band_width=1.0, exclude_first=True)

reward_PTTR = reil_functions.PercentInRange(
    name='PTTR', y_var_name='daily_INR_history',
    length=-1, multiplier=-1.0, retrospective=True, interpolate=False,
    acceptable_range=(2, 3), exclude_first=True)

reward_PTTR_interpolation = reil_functions.PercentInRange(
    name='PTTR',
    y_var_name='INR_history', x_var_name='interval_history',
    length=2, multiplier=-1.0, retrospective=True, interpolate=True,
    acceptable_range=(2, 3), exclude_first=True)

statistic_PTTR = reil_functions.PercentInRange(
    name='PTTR', y_var_name='daily_INR_history',
    length=-1, multiplier=1.0, retrospective=True, interpolate=False,
    acceptable_range=(2, 3), exclude_first=True)


class Warfarin(HealthSubject):
    '''
    A warfarin subject based on Hamberg's two compartment PK/PD model.
    '''

    def __init__(
            self,
            patient: Patient,
            INR_range: Tuple[float, float] = (0.0, 15.0),
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

        INR_range:
            A tuple that specifies min and max INR.

        dose_range:
            A tuple that specifies min and max dose.

        interval_range:
            A tuple that specifies min and max number of days between two
            measurements.

        max_day:
            Maximum duration of each trial.

        '''

        super().__init__(
            patient=patient,
            measurement_name='INR',
            measurement_range=INR_range,
            dose_range=dose_range,
            interval_range=interval_range,
            max_day=max_day,
            **kwargs)

        Warfarin._generate_state_defs(self)
        Warfarin._generate_action_defs(self)
        Warfarin._generate_reward_defs(self)
        Warfarin._generate_statistic_defs(self)

    def _generate_state_defs(self):
        current_defs = self.state.definitions
        for name, args in state_definitions.items():
            if name not in current_defs:
                self.state.add_definition(name, *args)

        # self.state.add_definition(
        #     'age', ('age', {}))

        # self.state.add_definition(
        #     'patient_basic', *patient_basic)

        # self.state.add_definition(
        #     'patient', *patient_basic, *patient_extra)

        # self.state.add_definition(
        #     'patient_w_dosing',
        #     *patient_basic, *patient_extra,
        #     ('day', {}),
        #     ('dose_history', {'length': -1}),
        #     ('INR_history', {'length': -1}),
        #     ('interval_history', {'length': -1}))

        # self.state.add_definition(
        #     'patient_for_baseline',
        #     *patient_basic, *patient_extra,
        #     ('day', {}),
        #     ('dose_history', {'length': 4}),
        #     ('INR_history', {'length': 4}),
        #     ('interval_history', {'length': 4}))

        # for i in range(1, 10):
        #     self.state.add_definition(
        #         f'patient_w_dosing_{i:02}',
        #         *patient_basic,
        #         ('day', {}),
        #         ('dose_history', {'length': i}),
        #         ('INR_history', {'length': i}),
        #         ('interval_history', {'length': i}))

        # self.state.add_definition(
        #     'patient_w_full_dosing',
        #     *patient_basic, *patient_extra,
        #     ('day', {}),
        #     ('daily_dose_history', {'length': -1}),
        #     ('daily_INR_history', {'length': -1}),
        #     ('interval_history', {'length': -1}))

        # self.state.add_definition(
        #     'daily_INR',
        #     ('daily_INR_history', {'length': -1}))

        # self.state.add_definition(
        #     'Measured_INR_2',
        #     ('INR_history', {'length': 2}),
        #     ('interval_history', {'length': 1}))

        # self.state.add_definition(
        #     'INR_within_2',
        #     ('daily_INR_history', {'length': -1}))

    def _generate_reward_defs(self):
        current_defs = self.reward.definitions

        if 'sq_dist_exact' not in current_defs:
            self.reward.add_definition(
                'sq_dist_exact', reward_sq_dist, 'INR_within_2')

        if 'sq_dist_interpolation' not in current_defs:
            self.reward.add_definition(
                'sq_dist_interpolation', reward_sq_dist_interpolation,
                'Measured_INR_2')

        if 'PTTR_exact' not in current_defs:
            self.reward.add_definition(
                'PTTR_exact', reward_PTTR, 'INR_within_2')

        if 'PTTR_interpolation' not in current_defs:
            self.reward.add_definition(
                'PTTR_interpolation', reward_PTTR_interpolation,
                'Measured_INR_2')

    def _generate_statistic_defs(self):
        if 'PTTR_exact_basic' not in self.statistic.definitions:
            self.statistic.add_definition(
                'PTTR_exact_basic', statistic_PTTR,
                'daily_INR', 'patient_basic')

        if 'PTTR_exact' not in self.statistic.definitions:
            self.statistic.add_definition(
                'PTTR_exact', statistic_PTTR, 'daily_INR', 'patient')

    def _generate_action_defs(self):
        dose_gen = self.feature_gen_set['dose']
        interval_gen = self.feature_gen_set['interval']

        def _actions(
                dose_values: Iterable[float], interval_values: Iterable[int]):
            actions = itertools.product(
                (dose_gen(vi)
                 for vi in dose_values),
                (interval_gen(vi)
                 for vi in interval_values)
            )

            return tuple(FeatureArray(a) for a in actions)

        caps = tuple(
            i for i in (5.0, 10.0, 15.0)
            if self._dose_range[0] <= i <= self._dose_range[1])
        dose = {
            cap: tuple(self.generate_dose_values(0.0, cap, 0.5))
            for cap in caps}

        int_fixed = {
            i: (i,) for i in (1, 2, 3, 7)
            if self._interval_range[0] <= i <= self._interval_range[1]}
        int_weekly = tuple(
            i for i in (7, 14, 21, 28)
            if self._interval_range[0] <= i <= self._interval_range[1])
        int_free = tuple(range(
            self._interval_range[0], self._interval_range[0] + 1))

        dose_int_fixed = {
            (d[0], i[0]): _actions(d[1], i[1])
            for d, i in itertools.product(
                dose.items(), int_fixed.items())
        }

        dose_int_free = {
            k: _actions(v, int_free)
            for k, v in dose.items()}

        dose_int_weekly = {
            k: _actions(v, int_weekly)
            for k, v in dose.items()}

        max_cap = min(caps[-1], self._dose_range[1])

        def _237(f: FeatureArray, cap: float):
            day: int = f['day'].value  # type: ignore
            if day == 0:
                return dose_int_fixed[cap, 2]
            elif day == 2:
                return dose_int_fixed[max_cap, 3]
            elif day >= 5:
                return dose_int_fixed[max_cap, 7]
            else:
                raise ValueError(f'Wrong day: {day}.')

        defs: Dict[str, Tuple[Callable[[Any], Tuple[FeatureArray, ...]], str]]
        defs = {
            **{
                f'daily_{int(cap):02}':
                    (lambda _: dose_int_fixed[cap, 1], 'day')  # type: ignore
                for cap in caps},
            **{
                f'237_{int(cap):02}':
                    (functools.partial(_237, cap=cap), 'day')
                for cap in caps},
            **{
                f'free_{int(cap):02}':
                    (lambda _: dose_int_free[cap], 'day')  # type: ignore
                for cap in caps},
            **{
                f'weekly_{int(cap):02}':
                    (lambda _: dose_int_weekly[cap], 'day')  # type: ignore
                for cap in caps}
        }

        current_defs = self.possible_actions.definitions
        for name, args in defs.items():
            if name not in current_defs:
                self.possible_actions.add_definition(name, *args)

            # self.possible_actions.add_definition(
            #     f'237_{int(cap):02}',
            #     functools.partial(_237, cap=cap), 'day')

            # self.possible_actions.add_definition(
            #     f'free_{int(cap):02}',
            #     lambda _: dose_int_free[cap], 'day')

            # self.possible_actions.add_definition(
            #     f'weekly_{int(cap):02}',
            #     lambda _: dose_int_weekly[cap], 'day')

    def _default_state_definition(
            self, _id: Optional[int] = None) -> FeatureArray:
        patient_features = self._patient.feature_set
        return FeatureArray([
            patient_features['age'],
            patient_features['CYP2C9'],
            patient_features['VKORC1']])

    def _sub_comp_age(self, _id: int, **kwargs: Any) -> Feature[float]:
        return super()._numerical_sub_comp('age')

    def _sub_comp_weight(self, _id: int, **kwargs: Any) -> Feature[float]:
        return self._numerical_sub_comp('weight')

    def _sub_comp_height(self, _id: int, **kwargs: Any) -> Feature[float]:
        return self._numerical_sub_comp('height')

    def _sub_comp_gender(self, _id: int, **kwargs: Any) -> Feature[str]:
        return self._categorical_sub_comp('gender')

    def _sub_comp_race(self, _id: int, **kwargs: Any) -> Feature[str]:
        return self._categorical_sub_comp('race')

    def _sub_comp_tobaco(self, _id: int, **kwargs: Any) -> Feature[str]:
        return self._categorical_sub_comp('tobaco')

    def _sub_comp_amiodarone(self, _id: int, **kwargs: Any) -> Feature[str]:
        return self._categorical_sub_comp('amiodarone')

    def _sub_comp_fluvastatin(self, _id: int, **kwargs: Any) -> Feature[str]:
        return self._categorical_sub_comp('fluvastatin')

    def _sub_comp_CYP2C9(self, _id: int, **kwargs: Any) -> Feature[str]:
        return self._categorical_sub_comp('CYP2C9')

    def _sub_comp_CYP2C9_masked(
            self, _id: int, days: int, **kwargs: Any) -> Feature[str]:
        return self._categorical_sub_comp('CYP2C9', self._day < days)

    def _sub_comp_VKORC1(self, _id: int, **kwargs: Any) -> Feature[str]:
        return self._categorical_sub_comp('VKORC1')

    def _sub_comp_VKORC1_masked(
            self, _id: int, days: int, **kwargs: Any) -> Feature[str]:
        return self._categorical_sub_comp('VKORC1', self._day < days)

    def _sub_comp_sensitivity(self, _id: int, **kwargs: Any) -> Feature[str]:
        return self._categorical_sub_comp('sensitivity')

    def _sub_comp_INR_history(
            self, _id: int, length: int = 1, **kwargs: Any
    ) -> Feature[Tuple[float, ...]]:
        return self._sub_comp_measurement_history(
            _id, length, **kwargs)

    def _sub_comp_daily_INR_history(
            self, _id: int, length: int = 1, **kwargs: Any
    ) -> Feature[Tuple[float, ...]]:
        return self._sub_comp_daily_measurement_history(
            _id, length, **kwargs)

    def _sub_comp_INR_within(
            self, _id: int, length: int = 1, **kwargs: Any
    ) -> Feature[Tuple[float, ...]]:
        intervals = self._get_history('interval_history', length).value
        return self._get_history('daily_INR', sum(intervals))  # type: ignore
