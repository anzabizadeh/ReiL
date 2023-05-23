# -*- coding: utf-8 -*-  pylint: disable=undefined-variable
'''
warfarin class
==============

This `warfarin` class implements a two compartment PK/PD model for warfarin.
'''
from __future__ import annotations

import functools
from typing import Any, Callable, Literal, cast

from reil.datatypes.feature import Feature, FeatureGeneratorType, FeatureSet
from reil.healthcare.patient import Patient
from reil.healthcare.subjects.dosing_subject import DosingSubject
from reil.utils import reil_functions

DefComponents = tuple[tuple[str, dict[str, Any]], ...]

patient_basic: DefComponents = (
    ('age', {}), ('CYP2C9', {}),
    ('VKORC1', {})
)
patient_extra: DefComponents = (
    ('weight', {}), ('height', {}),
    ('gender', {}), ('race', {}), ('tobaco', {}),
    ('amiodarone', {}), ('fluvastatin', {})
)

sensitivity: DefComponents = (('sensitivity', {}),)
patient_w_sensitivity: DefComponents = (
    *patient_basic, *sensitivity, *patient_extra)

state_definitions: dict[str, DefComponents] = {
    'age': (('age', {}),),
    'patient_basic': patient_basic,
    'patient_w_sensitivity_basic': (*patient_basic, *sensitivity),
    'patient_w_sensitivity': patient_w_sensitivity,
    'patient': (*patient_basic, *patient_extra),
    'patient_w_dosing': (
        *patient_basic, *patient_extra,
        # ('day', {}),
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
        f'no_patient_w_dosing_{i:02}': (
            ('dose_history', {'length': i}),
            ('INR_history', {'length': i + 1}),
            ('interval_history', {'length': i}))
        for i in range(1, 4)},
    **{
        f'patient_w_dosing_{i:02}': (
            *patient_basic,
            ('dose_history', {'length': i}),
            ('INR_history', {'length': i + 1}),
            ('interval_history', {'length': i}))
        for i in range(1, 4)},

    **{
        f'patient_w_dosing_w_baseline_{i:02}': (
            *patient_basic, *patient_extra,
            ('day', {}),
            ('dose_history', {'length': i}),
            ('INR_history', {'length': i + 1}),
            ('interval_history', {'length': i}))
        for i in range(1, 4)},

    'patient_w_full_dosing': (
        *patient_w_sensitivity,
        ('day', {}),
        ('daily_dose_history', {'length': -1}),
        ('daily_INR_history', {'length': -1}),
        ('interval_history', {'length': -1})),

    'daily_INR': (('daily_INR_history', {'length': -1}),),

    'recent_daily_INR': (('INR_within', {'length': 1}),),

    'Measured_INR_2': (
        ('INR_history', {'length': 2}),
        ('interval_history', {'length': 1})),
    'measured_dose_2': (('daily_dose_history', {'length': 2}),),
    'day_and_last_dose': (('day', {}), ('daily_dose_history', {'length': 1})),
    'day_and_last_dose_INR': (
        ('day', {}), ('daily_dose_history', {'length': 1}),
        ('daily_INR_history', {'length': 1}))
}

action_definition_names = [
    '237_15', 'daily_15', 'free_15', 'semi_15', 'weekly_15', 'delta', 'percent', 'percent_semi']

reward_definitions: dict[str, tuple[reil_functions.ReilFunction[float, int], str]] = dict(
    sq_dist=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False),
        'recent_daily_INR'
    ),
    sq_dist_modified=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist_modified', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05),
        'recent_daily_INR'
    ),
    sq_dist_modified_w_constant=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist_modified_w_constant', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05, constant=-100.0),
        'recent_daily_INR'
    ),
    average_sq_dist_modified_w_constant=(
        reil_functions.NormalizedSquareDistance(
            name='average_sq_dist_modified_w_constant',
            y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05, average=True, constant=-10.0),
        'recent_daily_INR'
    ),
    dist=(
        reil_functions.NormalizedDistance(
            name='dist', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False),
        'recent_daily_INR'
    ),
    sq_dist_interpolation=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist_interpolation',
            y_var_name='INR_history', x_var_name='interval_history',
            length=2, multiplier=-1.0,  interpolate=True,
            center=2.5, band_width=1.0, exclude_first=True),
        'Measured_INR_2'
    ),
    PTTR_exact=(
        reil_functions.PercentInRange(
            name='PTTR_exact', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            acceptable_range=(2, 3), exclude_first=True),
        'recent_daily_INR'
    ),
    dose_change=(
        reil_functions.NotEqual(
            name='dose_change', y_var_name='daily_dose_history',
            length=2, multiplier=-1.0),
        'measured_dose_2'
    ),
    PTTR_interpolation=(
        reil_functions.PercentInRange(
            name='PTTR_interpolation',
            y_var_name='INR_history', x_var_name='interval_history',
            length=2, multiplier=-1.0,  interpolate=True,
            acceptable_range=(2, 3), exclude_first=True),
        'Measured_INR_2'
    )
)

statistic_definition_names = ['PTTR_exact_basic', 'PTTR_exact']

statistic_PTTR = reil_functions.PercentInRange(
    name='PTTR', y_var_name='daily_INR_history',
    length=-1, multiplier=1.0,  interpolate=False,
    acceptable_range=(2, 3), exclude_first=True)


class Warfarin(DosingSubject):
    '''
    A warfarin subject based on Hamberg's two compartment PK/PD model.
    '''

    def __init__(
            self,
            patient: Patient,
            INR_range: tuple[float, float] = (0.0, 15.0),
            dose_range: tuple[float, float] = (0.0, 15.0),
            dose_step: float = 0.5,
            interval_range: tuple[int, int] = (1, 28),
            interval_step: int = 1,
            max_day: int = 90,
            decision_mode: Literal[
                'dose', 'dose_interval', 'dose_change', 'dose_change_interval',
                'dose_change_p', 'dose_change_interval_p'
            ] = 'dose_interval',
            percent_dose_change: tuple[float, ...] | None = None,
            dose_range_p: tuple[float, float] | None = None,
            round_to_step: bool = True,
            backfill: bool = True,
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
            max_day=max_day,
            backfill=backfill,
            interval_range=interval_range,
            dose_range=dose_range,
            decision_mode=decision_mode,
            interval_step=interval_step,
            dose_step=dose_step,
            percent_dose_change=percent_dose_change,
            dose_range_p=dose_range_p,
            round_to_step=round_to_step,
            **kwargs)

        self.state.definition_reference_function(
            f=self._state_def_reference,
            available_definitions=list(state_definitions))
        self.possible_actions.definition_reference_function(
            f=self._action_def_reference,
            available_definitions=action_definition_names)
        self.reward.definition_reference_function(
            f=self._reward_def_reference,
            available_definitions=list(reward_definitions))
        self.statistic.definition_reference_function(
            f=self._statistic_def_reference,
            available_definitions=list(statistic_definition_names))

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        del config['measurement_name']
        config['INR_range'] = config.pop('measurement_range')

        return config

    def copy(
        self, perturb: bool = False, n: int | None = None
    ) -> 'Warfarin' | list['Warfarin']:
        copied_subjects_temp = super().copy(perturb=False, n=n)

        if perturb:
            if n is None:
                copied_subjects = cast(Warfarin, copied_subjects_temp)
                if copied_subjects._patient is not None:
                    copied_subjects._patient._model.perturb(  # type: ignore
                        day=self._day)
            else:
                copied_subjects = cast(list[Warfarin], copied_subjects_temp)
                for c in copied_subjects:
                    if c._patient is not None:
                        c._patient._model.perturb(day=self._day)  # type: ignore
        else:
            copied_subjects = cast(
                Warfarin | list[Warfarin], copied_subjects_temp)

        return copied_subjects

    def _generate_state_defs(self):
        current_defs = self.state.definitions
        for name, args in state_definitions.items():
            if name not in current_defs:
                self.state.add_definition(name, *args)

    def _generate_reward_defs(self):
        current_defs = self.reward.definitions

        for name, args in reward_definitions.items():
            if name not in current_defs:
                self.reward.add_definition(
                    name, *args)

    def _generate_statistic_defs(self):
        if 'PTTR_exact_basic' not in self.statistic.definitions:
            self.statistic.add_definition(
                'PTTR_exact_basic', statistic_PTTR,
                'daily_INR', 'patient_w_sensitivity_basic')

        if 'PTTR_exact' not in self.statistic.definitions:
            self.statistic.add_definition(
                'PTTR_exact', statistic_PTTR,
                'daily_INR', 'patient_w_sensitivity')

    def _generate_action_defs(self):  # noqa: C901
        current_action_definitions = self.possible_actions.definitions

        def _generate(
            feature: FeatureSet,
            ops: tuple[Callable[[FeatureSet], bool], ...],
            dose_masks: tuple[dict[float, float], ...],
            interval_masks: tuple[dict[int, int], ...]
        ) -> FeatureGeneratorType:
            self.action_gen_set.unmask('dose')
            if not self._interval_mode:
                for op, d_mask in zip(ops, dose_masks):
                    if op(feature):
                        self.action_gen_set.mask('dose', d_mask)

                        return self.action_gen_set.make_generator()

            else:
                self.action_gen_set.unmask('interval')
                for op, d_mask, i_mask in zip(ops, dose_masks, interval_masks):
                    if op(feature):
                        self.action_gen_set.mask('dose', d_mask)
                        self.action_gen_set.mask('interval', i_mask)

                        return self.action_gen_set.make_generator()

                self.action_gen_set.mask('interval', interval_masks[-1])

            self.action_gen_set.mask('dose', dose_masks[-1])

            return self.action_gen_set.make_generator()

        caps = tuple(
            i for i in (5.0, 10.0, 15.0)
            if self._dose_range[0] <= i <= self._dose_range[1])
        max_cap = min(caps[-1], self._dose_range[1])

        dose = {
            cap: {
                d: cap
                for d in self.generate_dose_values(cap, max_cap, 0.5)
                if d > cap}
            for cap in caps}

        min_interval, max_interval = self._interval_range
        int_fixed = {
            d: {
                i: d
                for i in range(
                    min_interval, max_interval + 1, self._interval_step)
                if i != d}
            for d in (1, 2, 3, 7)}
        int_semi_free = {
            i: min_interval
            for i in range(min_interval, max_interval + 1, self._interval_step)
            if i not in (1, 2, 3, 7, 14, 21, 28)}
        int_weekly = {
            i: min_interval
            for i in range(min_interval, max_interval + 1, self._interval_step)
            if i not in (7, 14, 21, 28)}

        name: str
        for cap in caps[:-1]:
            name = f'237_{int(cap):02}'
            if name not in current_action_definitions:
                self.possible_actions.add_definition(
                    name, functools.partial(
                        _generate,
                        ops=(
                            lambda f: f['day'].value >= 5,
                            lambda f: f['day'].value == 2,
                            lambda f: f['day'].value == 0),
                        dose_masks=(
                            dose[max_cap], dose[max_cap], dose[cap]
                        ),
                        interval_masks=(
                            int_fixed[7], int_fixed[3], int_fixed[2])),
                    'day')

            name = f'daily_{int(cap):02}'
            if name not in current_action_definitions:
                self.possible_actions.add_definition(
                    name, functools.partial(
                        _generate,
                        ops=(lambda f: f['day'].value > 0,),
                        dose_masks=(dose[max_cap], dose[cap]),
                        interval_masks=(int_fixed[1], int_fixed[1])),
                    'day')

            name = f'free_{int(cap):02}'
            if name not in current_action_definitions:
                self.possible_actions.add_definition(
                    name, functools.partial(
                        _generate,
                        ops=(lambda f: f['day'].value > 0,),
                        dose_masks=(dose[max_cap], dose[cap]),
                        interval_masks=({}, {})),
                    'day')

            name = f'semi_{int(cap):02}'
            if name not in current_action_definitions:
                self.possible_actions.add_definition(
                    name, functools.partial(
                        _generate,
                        ops=(lambda f: f['day'].value > 0,),
                        dose_masks=(dose[max_cap], dose[cap]),
                        interval_masks=(int_semi_free, int_semi_free)),
                    'day')

            name = f'weekly_{int(cap):02}'
            if name not in current_action_definitions:
                self.possible_actions.add_definition(
                    name, functools.partial(
                        _generate,
                        ops=(lambda f: f['day'].value > 0,),
                        dose_masks=(dose[max_cap], dose[cap]),
                        interval_masks=(int_weekly, int_weekly)),
                    'day')

        name = '237_15'
        if name not in current_action_definitions:
            self.possible_actions.add_definition(
                name, functools.partial(
                    _generate,
                    ops=(
                        lambda f: f['day'].value >= 5,
                        lambda f: f['day'].value == 2,
                        lambda f: f['day'].value == 0),
                    dose_masks=(
                        dose[max_cap], dose[max_cap], dose[max_cap]
                    ),
                    interval_masks=(
                        int_fixed[7], int_fixed[3], int_fixed[2])),
                'day')

        name = 'daily_15'
        if name not in current_action_definitions:
            self.possible_actions.add_definition(
                name, functools.partial(
                    _generate,
                    ops=(),
                    dose_masks=(dose[max_cap],),
                    interval_masks=(int_fixed[1],)),
                'day')

        name = 'free_15'
        if name not in current_action_definitions:
            self.possible_actions.add_definition(
                name, functools.partial(
                    _generate,
                    ops=(),
                    dose_masks=(dose[max_cap],),
                    interval_masks=({},)),
                'day')

        name = 'semi_15'
        if name not in current_action_definitions:
            self.possible_actions.add_definition(
                name, functools.partial(
                    _generate,
                    ops=(),
                    dose_masks=(dose[max_cap],),
                    interval_masks=(int_semi_free,)),
                'day')

        name = 'weekly_15'
        if name not in current_action_definitions:
            self.possible_actions.add_definition(
                name, functools.partial(
                    _generate,
                    ops=(lambda f: f['day'].value > 0,),
                    dose_masks=(dose[max_cap],),
                    interval_masks=(int_weekly,)),
                'day')

        name = 'delta'
        if name not in current_action_definitions:
            def delta_dose(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_change')
                self.action_gen_set.unmask('interval')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                day: int = feature['day'].value  # type: ignore
                min_dose, max_dose = self._dose_range
                min_delta = min_dose - last_dose
                max_delta = max_dose - last_dose
                d_list = self.generate_dose_values(min_dose, max_dose)
                d_list = set((*d_list, *(-x for x in d_list)))
                d_mask = {
                    d: min_delta if d < min_delta else max_delta
                    for d in d_list
                    if not (min_delta <= d <= max_delta)
                }
                self.action_gen_set.mask('dose_change', d_mask)

                if day >= 5:
                    interval_mask = int_fixed[7]
                elif day == 2:
                    interval_mask = int_fixed[3]
                elif day == 0:
                    interval_mask = int_fixed[2]
                else:
                    raise ValueError(f'wrong day: {day}.')

                self.action_gen_set.mask('interval', interval_mask)

                return self.action_gen_set.make_generator()

            self.possible_actions.add_definition(
                name, delta_dose, 'day_and_last_dose')

        name = 'percent'
        if name not in current_action_definitions:
            def percent_dose(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_change_p')
                self.action_gen_set.unmask('interval')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                day: int = feature['day'].value  # type: ignore
                min_dose, max_dose = self._dose_range
                min_delta = min_dose - last_dose
                max_delta = max_dose - last_dose
                d_list = self.generate_dose_values(min_dose, max_dose)
                d_list = set((*d_list, *(-x for x in d_list)))
                d_mask = {
                    d: min_delta if d < min_delta else max_delta
                    for d in d_list
                    if not (min_delta <= d <= max_delta)
                }
                self.action_gen_set.mask('dose_change_p', d_mask)

                if day >= 5:
                    interval_mask = int_fixed[7]
                elif day == 2:
                    interval_mask = int_fixed[3]
                elif day == 0:
                    interval_mask = int_fixed[2]
                else:
                    raise ValueError(f'wrong day: {day}.')

                self.action_gen_set.mask('interval', interval_mask)

                return self.action_gen_set.make_generator()

            self.possible_actions.add_definition(
                name, percent_dose, 'day_and_last_dose')

    def _state_def_reference(
            self, name: str) -> DefComponents | None:
        try:
            return state_definitions[name]
        except KeyError:
            return super()._state_def_reference(name)

    def _action_def_reference(  # noqa: C901
        self, name: str
    ) -> tuple[Callable[..., FeatureGeneratorType], str] | None:
        def _generate(
                feature: FeatureSet,
                ops: tuple[Callable[[FeatureSet], bool], ...],
                dose_masks: tuple[dict[float, float], ...],
                interval_masks: tuple[dict[int, int], ...]
        ) -> FeatureGeneratorType:
            self.action_gen_set.unmask('dose')
            if not self._interval_mode:
                for op, d_mask in zip(ops, dose_masks):
                    if op(feature):
                        self.action_gen_set.mask('dose', d_mask)

                        return self.action_gen_set.make_generator()

            else:
                self.action_gen_set.unmask('interval')
                for op, d_mask, i_mask in zip(ops, dose_masks, interval_masks):
                    if op(feature):
                        self.action_gen_set.mask('dose', d_mask)
                        self.action_gen_set.mask('interval', i_mask)

                        return self.action_gen_set.make_generator()

                self.action_gen_set.mask('interval', interval_masks[-1])

            self.action_gen_set.mask('dose', dose_masks[-1])

            return self.action_gen_set.make_generator()

        caps = tuple(
            i for i in (5.0, 10.0, 15.0)
            if self._dose_range[0] <= i <= self._dose_range[1])
        max_cap = min(caps[-1], self._dose_range[1])

        dose = {
            cap: {
                d: cap
                for d in self.generate_dose_values(cap, max_cap, 0.5)
                if d > cap}
            for cap in caps}

        min_interval, max_interval = self._interval_range
        int_fixed = {
            d: {
                i: d
                for i in range(
                    min_interval, max_interval + 1, self._interval_step)
                if i != d}
            for d in (1, 2, 3, 7)}
        int_semi_free = {
            i: min_interval
            for i in range(min_interval, max_interval + 1, self._interval_step)
            if i not in (1, 2, 3, 7, 14, 21, 28)}
        int_weekly = {
            i: min_interval
            for i in range(min_interval, max_interval + 1, self._interval_step)
            if i not in (7, 14, 21, 28)}

        # for cap in caps[:-1]:
        #     name = f'237_{int(cap):02}'
        #     if name not in current_action_definitions:
        #         self.possible_actions.add_definition(
        #             name, functools.partial(
        #                 _generate,
        #                 ops=(
        #                     lambda f: f['day'].value >= 5,
        #                     lambda f: f['day'].value == 2,
        #                     lambda f: f['day'].value == 0),
        #                 dose_masks=(
        #                     dose[max_cap], dose[max_cap], dose[cap]
        #                 ),
        #                 interval_masks=(
        #                     int_fixed[7], int_fixed[3], int_fixed[2])),
        #             'day')

        #     name = f'daily_{int(cap):02}'
        #     if name not in current_action_definitions:
        #         self.possible_actions.add_definition(
        #             name, functools.partial(
        #                 _generate,
        #                 ops=(lambda f: f['day'].value > 0,),
        #                 dose_masks=(dose[max_cap], dose[cap]),
        #                 interval_masks=(int_fixed[1], int_fixed[1])),
        #             'day')

        #     name = f'free_{int(cap):02}'
        #     if name not in current_action_definitions:
        #         self.possible_actions.add_definition(
        #             name, functools.partial(
        #                 _generate,
        #                 ops=(lambda f: f['day'].value > 0,),
        #                 dose_masks=(dose[max_cap], dose[cap]),
        #                 interval_masks=({}, {})),
        #             'day')

        #     name = f'semi_{int(cap):02}'
        #     if name not in current_action_definitions:
        #         self.possible_actions.add_definition(
        #             name, functools.partial(
        #                 _generate,
        #                 ops=(lambda f: f['day'].value > 0,),
        #                 dose_masks=(dose[max_cap], dose[cap]),
        #                 interval_masks=(int_semi_free, int_semi_free)),
        #             'day')

        #     name = f'weekly_{int(cap):02}'
        #     if name not in current_action_definitions:
        #         self.possible_actions.add_definition(
        #             name, functools.partial(
        #                 _generate,
        #                 ops=(lambda f: f['day'].value > 0,),
        #                 dose_masks=(dose[max_cap], dose[cap]),
        #                 interval_masks=(int_weekly, int_weekly)),
        #             'day')

        if name == '237_15':
            return (
                functools.partial(
                    _generate,
                    ops=(
                        lambda f: f['day'].value >= 5,
                        lambda f: f['day'].value == 2,
                        lambda f: f['day'].value == 0),
                    dose_masks=(
                        dose[max_cap], dose[max_cap], dose[max_cap]
                    ),
                    interval_masks=(
                        int_fixed[7], int_fixed[3], int_fixed[2])),
                'day')

        if name == 'daily_15':
            return (
                functools.partial(
                    _generate, ops=(),
                    dose_masks=(dose[max_cap],),
                    interval_masks=(int_fixed[1],)),
                'day')

        if name == 'free_15':
            return (
                functools.partial(
                    _generate, ops=(),
                    dose_masks=(dose[max_cap],), interval_masks=({},)),
                'day')

        if name == 'semi_15':
            return (
                functools.partial(
                    _generate, ops=(),
                    dose_masks=(dose[max_cap],),
                    interval_masks=(int_semi_free,)),
                'day')

        if name == 'weekly_15':
            return (
                functools.partial(
                    _generate, ops=(lambda f: f['day'].value > 0,),
                    dose_masks=(dose[max_cap],),
                    interval_masks=(int_weekly,)),
                'day')

        if name == 'delta':
            def delta_dose(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_change')
                self.action_gen_set.unmask('interval')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                day: int = feature['day'].value  # type: ignore
                min_dose, max_dose = self._dose_range
                min_delta = min_dose - last_dose
                max_delta = max_dose - last_dose
                d_list = self.generate_dose_values(min_dose, max_dose)
                d_list = set((*d_list, *(-x for x in d_list)))
                d_mask = {
                    d: min_delta if d < min_delta else max_delta
                    for d in d_list
                    if not (min_delta <= d <= max_delta)
                }
                self.action_gen_set.mask('dose_change', d_mask)

                if day >= 5:
                    interval_mask = int_fixed[7]
                elif day == 2:
                    interval_mask = int_fixed[3]
                elif day == 0:
                    interval_mask = int_fixed[2]
                else:
                    raise ValueError(f'wrong day: {day}.')

                self.action_gen_set.mask('interval', interval_mask)

                return self.action_gen_set.make_generator()

            return delta_dose, 'day_and_last_dose'

        if name == 'percent_interval':
            def percent_dose(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_change_p')
                self.action_gen_set.unmask('interval')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                day: int = feature['day'].value  # type: ignore
                min_dose, max_dose = self._dose_range
                all_ps: tuple[float, ...] = \
                    self.feature_gen_set['dose_change_p'].fixed_values  # type: ignore
                permissibles = [
                    p for p in all_ps
                    if (min_dose <= last_dose * (1 + p) <= max_dose)
                ]
                min_p = min(permissibles)
                max_p = max(permissibles)
                p_mask = {
                    p: min_p if p < min_p else max_p
                    for p in all_ps
                    if p not in permissibles
                }
                self.action_gen_set.mask('dose_change_p', p_mask)

                if day >= 5:
                    interval_mask = int_fixed[7]
                    self.action_gen_set.mask('interval', interval_mask)
                # elif day == 2:
                #     interval_mask = int_fixed[3]
                # elif day == 0:
                #     interval_mask = int_fixed[2]
                # else:
                #     raise ValueError(f'wrong day: {day}.')

                # self.action_gen_set.mask('interval', interval_mask)

                return self.action_gen_set.make_generator()

            return percent_dose, 'day_and_last_dose'

        if name == 'percent':
            def percent_dose(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_change_p')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                min_dose, max_dose = self._dose_range
                all_ps: tuple[float, ...] = \
                    self.feature_gen_set['dose_change_p'].fixed_values  # type: ignore
                permissibles = [
                    p for p in all_ps
                    if (min_dose <= last_dose * (1 + p) <= max_dose)
                ]
                min_p = min(permissibles)
                max_p = max(permissibles)
                p_mask = {
                    p: min_p if p < min_p else max_p
                    for p in all_ps
                    if p not in permissibles
                }
                self.action_gen_set.mask('dose_change_p', p_mask)

                return self.action_gen_set.make_generator()

            return percent_dose, 'day_and_last_dose'

        if name == 'percent_guided':
            def percent_dose_guided(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_change_p')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                last_INR: float = \
                    feature['daily_INR_history'].value[-1]  # type: ignore
                min_dose, max_dose = self._dose_range
                all_ps: tuple[float, ...] = \
                    self.feature_gen_set['dose_change_p'].fixed_values  # type: ignore
                permissibles = [
                    p for p in all_ps
                    if (min_dose <= last_dose * (1 + p) <= max_dose)
                ]
                if last_INR > 3.0:
                    permissibles = [
                        p for p in permissibles
                        if p <= 0.0
                    ]
                elif last_INR < 2.0:
                    permissibles = [
                        p for p in permissibles
                        if p >= 0.0
                    ]
                min_p = min(permissibles)
                max_p = max(permissibles)
                p_mask = {
                    p: min_p if p < min_p else max_p
                    for p in all_ps
                    if p not in permissibles
                }
                self.action_gen_set.mask('dose_change_p', p_mask)

                return self.action_gen_set.make_generator()

            return percent_dose_guided, 'day_and_last_dose_INR'

        if name == 'percent_semi':
            def percent_dose(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_change_p')
                self.action_gen_set.unmask('interval')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                # day: int = feature['day'].value  # type: ignore
                min_dose, max_dose = self._dose_range
                all_ps: tuple[float, ...] = \
                    self.feature_gen_set['dose_change_p'].fixed_values  # type: ignore
                permissibles = [
                    p for p in all_ps
                    if (min_dose <= last_dose * (1 + p) <= max_dose)
                ]
                min_p = min(permissibles)
                max_p = max(permissibles)
                p_mask = {
                    p: min_p if p < min_p else max_p
                    for p in all_ps
                    if p not in permissibles
                }
                self.action_gen_set.mask('dose_change_p', p_mask)
                self.action_gen_set.mask('interval', int_semi_free)

                return self.action_gen_set.make_generator()

            return percent_dose, 'day_and_last_dose'

    def _reward_def_reference(
        self, name: str
    ) -> tuple[reil_functions.ReilFunction, str] | None:
        try:
            return reward_definitions[name]
        except KeyError:
            return super()._reward_def_reference(name)

    def _statistic_def_reference(self, name: str):
        if name == 'PTTR_exact_basic':
            return statistic_PTTR, 'daily_INR', 'patient_w_sensitivity_basic'

        if name == 'PTTR_exact':
            return statistic_PTTR, 'daily_INR', 'patient_w_sensitivity'

    def _sub_comp_age(self, _id: int, **kwargs: Any) -> Feature:
        return super()._numerical_sub_comp('age')

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

    def _sub_comp_CYP2C9_masked(
            self, _id: int, days: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('CYP2C9', self._day < days)

    def _sub_comp_VKORC1(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('VKORC1')

    def _sub_comp_VKORC1_masked(
            self, _id: int, days: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('VKORC1', self._day < days)

    def _sub_comp_sensitivity(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('sensitivity')

    def _sub_comp_INR_history(
            self, _id: int, length: int = 1, **kwargs: Any
    ) -> Feature:
        return self._sub_comp_measurement_history(
            _id, length, backfill=self._backfill, **kwargs)

    def _sub_comp_daily_INR_history(
            self, _id: int, length: int = 1, **kwargs: Any
    ) -> Feature:
        return self._sub_comp_daily_measurement_history(
            _id, length, backfill=self._backfill, **kwargs)

    def _sub_comp_INR_within(
            self, _id: int, length: int = 1, **kwargs: Any
    ) -> Feature:
        intervals = self._get_history('interval_history', length).value
        return self._get_history(
            'daily_INR_history', sum(intervals))  # type: ignore
