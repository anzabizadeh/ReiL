# -*- coding: utf-8 -*-  pylint: disable=undefined-variable
'''
warfarin class
==============

This `warfarin` class implements a two compartment PK/PD model for warfarin.
'''

import functools
from collections.abc import Callable
from typing import Any, Literal

from reil.datatypes.feature import (FeatureGenerator, FeatureGeneratorType,
                                    FeatureSet)
from reil.healthcare.patient import Patient
from reil.healthcare.subjects.warfarin import Warfarin


def additive(current_value: float, change: float) -> float:
    return current_value + change


def multiplicative(current_value: float, change: float) -> float:
    return current_value * (1 + change)


class WarfarinIncrementalAction(Warfarin):
    '''
    A warfarin subject based on Hamberg's two compartment PK/PD model.
    The actions in this class is limited to a handful of actions, and the
    object updates the action values at durations until convergence.
    '''

    def __init__(
            self,
            patient: Patient,
            action_values: tuple[float, ...],
            increment_range: tuple[float, float],
            action_increment: Literal['additive', 'multiplicative'] = 'additive',
            INR_range: tuple[float, float] = (0.0, 15.0),
            dose_range: tuple[float, float] = (0.0, 15.0),
            dose_step: float = 0.5,
            duration_range: tuple[int, int] = (1, 28),
            duration_step: int = 1,
            max_day: int = 90,
            dose_only: bool = False,
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

        duration_range:
            A tuple that specifies min and max number of days between two
            measurements.

        max_day:
            Maximum duration of each trial.

        '''

        super(Warfarin, self).__init__(
            patient=patient,
            measurement_name='INR',
            measurement_range=INR_range,
            dose_range=dose_range,
            dose_step=dose_step,
            duration_range=duration_range,
            duration_step=duration_step,
            max_day=max_day,
            **kwargs)

        self._action_values: tuple[float, ...] = (0.,) * len(action_values)
        self._increment_range = increment_range
        self._dose_only = dose_only
        self._backfill = backfill
        self.action_gen_set += FeatureGenerator.numerical_fixed_values(
            name='dose_change',
            lower=self._increment_range[0], upper=self._increment_range[1],
            fixed_values=self._action_values)

        if not dose_only:
            self.action_gen_set += self.feature_gen_set['duration']

        WarfarinIncrementalAction._generate_state_defs(self)
        WarfarinIncrementalAction._generate_action_defs(self)
        WarfarinIncrementalAction._generate_reward_defs(self)
        WarfarinIncrementalAction._generate_statistic_defs(self)
        self.set_action_values(action_values)
        self._last_dose = 0.0
        self._op = (
            additive if action_increment == 'additive' else multiplicative)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        del config['measurement_name']
        config['INR_range'] = config.pop('measurement_range')
        config['action_increment'] = (
            'additive' if config.pop('_op') is additive
            else 'multiplicative')

        return config

    def set_action_values(
            self, action_values: tuple[float, ...]) -> None:
        if len(action_values) != len(self._action_values):
            raise ValueError(
                'The list of action values should be of size '
                f'{len(self._action_values)}, '
                f'received a list of size {len(action_values)}')
        for a in action_values:
            if not (self._increment_range[0] <= a <= self._increment_range[1]):
                raise ValueError(
                    f'dose range is {self._increment_range}. '
                    f'Got {a} as an action.')

        self._action_values = action_values
        self.action_gen_set._generators['dose_change'].__dict__['fixed_values'] = \
            self._action_values

    def _generate_action_defs(self):
        def _generate(
                feature: FeatureSet,
                ops: tuple[Callable[[FeatureSet], bool], ...],
                dose_masks: tuple[dict[float, float], ...],
                duration_masks: tuple[dict[int, int], ...]
        ) -> FeatureGeneratorType:
            self.action_gen_set.unmask('dose_change')
            if self._dose_only:
                for op, d_mask in zip(ops, dose_masks):
                    if op(feature):
                        self.action_gen_set.mask('dose_change', d_mask)

                        return self.action_gen_set.make_generator()

            else:
                self.action_gen_set.unmask('duration')
                for op, d_mask, i_mask in zip(ops, dose_masks, duration_masks):
                    if op(feature):
                        self.action_gen_set.mask('dose_change', d_mask)
                        self.action_gen_set.mask('duration', i_mask)

                        return self.action_gen_set.make_generator()

                self.action_gen_set.mask('duration', duration_masks[-1])

            self.action_gen_set.mask('dose_change', dose_masks[-1])

            return self.action_gen_set.make_generator()

        min_duration, max_duration = self._duration_range
        int_fixed = {
            d: {
                i: d
                for i in range(
                    min_duration, max_duration + 1, self._duration_step)
                if i != d}
            for d in (1, 2, 3, 7)}

        self.possible_actions.add_definition(
            'action', lambda _: self.action_gen_set.make_generator(), 'default'
        )

        self.possible_actions.add_definition(
            '237', functools.partial(
                _generate,
                ops=(
                    lambda f: f['day'].value >= 5,  # type: ignore
                    lambda f: f['day'].value == 2,
                    lambda f: f['day'].value == 0),
                dose_masks=({}, {}, {}),
                duration_masks=(
                    int_fixed[7], int_fixed[3], int_fixed[2])), 'day'
        )

    def _take_effect(
            self, action: FeatureSet, _id: int = 0
    ) -> FeatureSet:
        assert self._patient is not None

        self._actions_taken.append(action)

        action_temp = action.value
        if self._day == 0:
            current_dose = float(action_temp['dose'])  # type: ignore
        else:
            current_dose = self._op(
                self._last_dose, float(action_temp['dose_change']))  # type: ignore
        current_dose = min(max(current_dose, 0.0), self._dose_range[1])
        self._last_dose = current_dose
        if not (self._dose_range[0] <= current_dose <= self._dose_range[1]):
            raise ValueError('Dose out of range.')

        current_duration = min(
            int(action_temp['duration']),  # type: ignore
            self._max_day - self._day)

        measurements_temp = self._patient.model(
            dose={
                i: current_dose
                for i in range(self._day, self._day + current_duration)
            },
            measurement_days=list(
                range(self._day + 1, self._day + current_duration + 1)
            )
        )[self._measurement_name]

        self._decision_points_dose_history[self._decision_points_index] = \
            current_dose
        self._decision_points_duration_history[self._decision_points_index] = \
            current_duration
        self._decision_points_index += 1

        day_temp = self._day
        self._day += current_duration

        self._full_dose_history[day_temp:self._day] = \
            [current_dose] * current_duration
        self._full_measurement_history[day_temp +
                                       1:self._day + 1] = measurements_temp

        self._decision_points_measurement_history[
            self._decision_points_index] = \
            self._full_measurement_history[self._day]

        return action
