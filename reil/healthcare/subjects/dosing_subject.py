# -*- coding: utf-8 -*-  pylint: disable=undefined-variable
'''
DosingSubject class
===================

This `DosingSubject` class implements interaction with patients.
'''

from typing import Any, Dict, List, Literal, Optional, Tuple

from reil.datatypes.feature import (Feature, FeatureGenerator,
                                    FeatureGeneratorSet, FeatureSet)
from reil.healthcare.patient import Patient
from reil.healthcare.subjects.health_subject import HealthSubject
from reil.serialization import deserialize, serialize


class DosingSubject(HealthSubject):
    '''
    A DosingSubject subject class with a patient.
    '''
    def __init__(  # noqa: C901
            self,
            patient: Patient,
            measurement_name: str,
            measurement_range: Tuple[float, float],
            max_day: int,
            interval_range: Tuple[int, int],
            dose_range: Tuple[float, float],
            decision_mode: Literal[
                'dose', 'dose_interval', 'dose_change', 'dose_change_interval',
                'dose_change_p', 'dose_change_interval_p'
            ],
            backfill: bool,
            interval_step: int = 1,
            dose_step: Optional[float] = None,
            percent_dose_change: Optional[Tuple[float, ...]] = None,
            dose_range_p: Optional[Tuple[float, float]] = None,
            round_to_step: bool = True,
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

        super().__init__(
            patient=patient,
            measurement_name=measurement_name,
            measurement_range=measurement_range,
            max_day=max_day,
            backfill=backfill,
            interval_range=interval_range,
            interval_step=interval_step, **kwargs)

        self._dose_range = dose_range
        self._dose_range_p = dose_range_p
        self._dose_step = dose_step
        self._percent_dose_change = percent_dose_change
        self._round_to_step = round_to_step

        self._decision_mode = decision_mode

        self._actions_taken: List[FeatureSet] = []
        self._full_dose_history = [0.0] * self._max_day
        self._decision_points_dose_history = [0.0] * self._max_day

        if '_p' in decision_mode:
            self._dose_mode = 'dose_change_p'
        elif 'change' in decision_mode:
            self._dose_mode = 'dose_change'
        else:
            self._dose_mode = 'dose'

        self._interval_mode = 'interval' in decision_mode

        if self._dose_mode in ('dose', 'dose_change'):
            if not self._dose_step:
                raise ValueError(
                    f'`dose_step` missing for `decision_mode`={decision_mode}.')
            self.feature_gen_set += FeatureGeneratorSet(
                FeatureGenerator.discrete(
                    name=name, lower=lower, upper=upper, step=step)
                for name, lower, upper, step in (
                    ('dose', *self._dose_range, self._dose_step),
                    ('dose_change', -max(self._dose_range),
                     max(self._dose_range), self._dose_step),
                    ('dose_history', *self._dose_range, self._dose_step),
                    ('daily_dose_history', *self._dose_range, self._dose_step),
                )
            )

        elif self._dose_mode == 'dose_change_p':
            if self._dose_range_p is None:
                raise ValueError(
                    f'`dose_range_p` missing for `decision_mode`={decision_mode}.')
            if self._percent_dose_change is None:
                raise ValueError(
                    f'`_percent_dose_change` missing for `decision_mode`={decision_mode}.')

            self.feature_gen_set += FeatureGeneratorSet(
                FeatureGenerator.numerical_fixed_values(
                    name='dose_change_p', fixed_values=self._percent_dose_change,
                    lower=self._dose_range_p[0], upper=self._dose_range_p[1])
            )

            if self._round_to_step:
                if self._dose_step is None:
                    raise ValueError(
                        '`round_to_step` is `True`. `dose_step` is required.')
                self.feature_gen_set += FeatureGeneratorSet(
                    FeatureGenerator.discrete(
                        name=name, lower=lower, upper=upper, step=step)
                    for name, lower, upper, step in (
                        ('dose_history', *self._dose_range, self._dose_step),
                        ('daily_dose_history', *self._dose_range, self._dose_step),
                    )
                )
            else:
                self.feature_gen_set += FeatureGeneratorSet(
                    FeatureGenerator.continuous(
                        name=name, lower=lower, upper=upper)
                    for name, lower, upper in (
                        ('dose_history', *self._dose_range),
                        ('daily_dose_history', *self._dose_range),
                    )
                )

        self.action_gen_set += self.feature_gen_set[self._dose_mode]

        if self._interval_mode:
            self.action_gen_set += self.feature_gen_set['interval']

        DosingSubject._generate_state_defs(self)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        actions_taken = config['internal_states'].pop(
            '_actions_taken', [])

        instance = super().from_config(config)
        for action in actions_taken:
            a: FeatureSet = deserialize(action)  # type: ignore
            instance.take_effect(a)

        return instance

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(dict(
            dose_range=self._dose_range,
            dose_step=self._dose_step,
        ))
        # Since we need to take actions again, dose list of the model should
        # be cleared.
        if config['patient']:
            config['patient']['config']['model']['config']['dose'] = {}
        config['internal_states']['_actions_taken'] = [
            serialize(action) for action in self._actions_taken]

        return config

    @classmethod
    def _empty_instance(cls):
        return cls(None, None)  # type: ignore

    @staticmethod
    def generate_dose_values(
            min_dose: float = 0.0,
            max_dose: float = 15.0,
            dose_increment: float = 0.5) -> List[float]:

        return list(min_dose + x * dose_increment
                    for x in range(
                        int((max_dose - min_dose) / dose_increment) + 1))

    def _take_effect(
            self, action: FeatureSet, _id: int = 0
    ) -> FeatureSet:
        if self._patient is None:
            raise RuntimeError('Patient is not set.')

        self._actions_taken.append(action)

        action_temp = action.value
        if 'dose' in action_temp:
            current_dose = float(action_temp['dose'])  # type: ignore
        elif 'dose_change' in action_temp:
            current_dose = (
                self._decision_points_dose_history[self._decision_points_index - 1] +
                action_temp['dose_change'])  # type: ignore
        elif 'dose_change_p' in action_temp:
            current_dose = (
                self._decision_points_dose_history[self._decision_points_index - 1] *
                (1 + action_temp['dose_change_p']))  # type: ignore
            if self._round_to_step:
                current_dose = \
                    self._dose_range[0] + self._dose_step * round(  # type: ignore
                        (current_dose - self._dose_range[0]) / self._dose_step)
        else:
            raise ValueError('dose/ dose_change/ dose_change_p not found.')

        interval = action_temp.get('interval', self._default_interval)
        current_interval = min(
            interval, self._max_day - self._day)  # type: ignore

        measurements_temp = self._patient.model(
            dose={
                i: current_dose
                for i in range(self._day, self._day + current_interval)
            },
            measurement_days=list(
                range(self._day, self._day + current_interval + 1)
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

        return action

    def reset(self) -> None:
        HealthSubject.reset(self)
        self._full_dose_history = [0.0] * self._max_day
        self._decision_points_dose_history = [0.0] * self._max_day
        self._actions_taken = []

    def _get_history(
            self, list_name: str, length: int, backfill: bool = False
    ) -> Feature:
        if length == 0:
            raise ValueError(
                'length should be a positive integer, or '
                '-1 for full length output.')

        filler: float
        _list: List[float]
        if list_name == 'dose_history':
            _list = self._decision_points_dose_history
            index = self._decision_points_index
            filler = 0.0
        elif list_name == 'daily_dose_history':
            _list = self._full_dose_history
            index = self._day
            filler = 0.0
        else:
            return super()._get_history(
                list_name=list_name, length=length, backfill=backfill)

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

    def _sub_comp_dose_history(
            self, _id: int, length: int = 1, backfill: bool = False,
            **kwargs: Any
    ) -> Feature:
        return self._get_history(  # type: ignore
            'dose_history', length, backfill)

    def _sub_comp_daily_dose_history(
            self, _id: int, length: int = 1, backfill: bool = False,
            **kwargs: Any
    ) -> Feature:
        return self._get_history(  # type: ignore
            'daily_dose_history', length, backfill)
