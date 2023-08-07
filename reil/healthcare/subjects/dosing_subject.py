# -*- coding: utf-8 -*-  pylint: disable=undefined-variable
'''
DosingSubject class
===================

This `DosingSubject` class implements interaction with patients.
'''

from typing import Any, Literal

from reil.datatypes.feature import (Feature, FeatureGenerator,
                                    FeatureGeneratorSet, FeatureSet)
from reil.healthcare.subjects.health_subject import HealthSubject
from reil.serialization import deserialize, serialize


class DosingSubject(HealthSubject):
    '''
    A DosingSubject subject class with a patient.
    '''
    def __init__(  # noqa: C901
            self,
            dose_range: tuple[float, float],
            decision_mode: Literal[
                'dose', 'dose_change', 'dose_percent_change',
                'dose_duration', 'dose_change_duration', 'dose_percent_change_duration',
                'dose_duration_joint', 'dose_change_duration_joint',
                'dose_percent_change_duration_joint',
            ],
            dose_step: float | None = None,
            decision_values: tuple[float, ...] | tuple[
                tuple[float, int], ...] | None = None,
            decision_range: tuple[float, float] | tuple[
                tuple[float, float], tuple[int, int]] | None = None,
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

        duration_range:
            A tuple that shows the minimum and maximum duration of each dosing
            decision.

        measurement_range:
            A tuple that shows the minimum and maximum possible values of each
            measurement.

        max_day:
            Maximum duration of each trial.
        '''

        super().__init__(**kwargs)

        self._dose_range = dose_range
        self._decision_range = decision_range
        self._dose_step = dose_step
        self._decision_values = decision_values
        self._round_to_step = round_to_step

        self._decision_mode = decision_mode
        self._duration_mode = 'duration' in decision_mode
        self._joint = 'joint' in decision_mode

        self._actions_taken: list[FeatureSet] = []
        self._full_dose_history = [0.0] * self._max_day
        self._decision_points_dose_history = [0.0] * self._max_day

        if self._joint:
            if 'dose_percent_change' in decision_mode:
                self._dose_mode = 'dose_percent_change_joint'

                if self._decision_values is None:
                    raise ValueError(
                        '`decision_values` missing for '
                        '`decision_mode`=dose_percent_change_joint.')

                if not isinstance(self._decision_values[0], tuple):
                    raise ValueError(
                        '`decision_values` should be of type '
                        '`tuple[tuple[float, int], ...]` for '
                        '`decision_mode`=dose_percent_change_joint.')

                # if self._decision_range is None:
                #     raise ValueError(
                #         '`decision_range` missing for '
                #         '`decision_mode`=dose_percent_change.')

                # if isinstance(self._decision_range[0], tuple):
                #     raise ValueError(
                #         '`decision_range` should be a tuple of floats for '
                #         '`decision_mode`=dose_percent_change.')

                # self.feature_gen_set += FeatureGeneratorSet(
                #     FeatureGenerator.numerical_fixed_values(
                #         name='dose_percent_change',
                #         fixed_values=self._decision_values,
                #         lower=self._decision_range[0],
                #         upper=self._decision_range[1])
                # )
                self.feature_gen_set += FeatureGeneratorSet(
                    FeatureGenerator.categorical(
                        name='dose_percent_change_joint',
                        categories=self._decision_values
                    )
                )

                if self._round_to_step:
                    if self._dose_step is None:
                        raise ValueError(
                            '`round_to_step` is `True`. `dose_step` is required.')
                    self.feature_gen_set += FeatureGeneratorSet(
                        FeatureGenerator.discrete(
                            name=name, lower=lower, upper=upper, step=self._dose_step)
                        for name, lower, upper in (
                            ('dose_history', *self._dose_range),
                            ('daily_dose_history', *self._dose_range),
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

            # elif 'dose_change' in decision_mode:
            #     self._dose_mode = 'dose_change'
            #     if not self._dose_step:
            #         raise ValueError(
            #             '`dose_step` missing for `decision_mode`=dose_change.')
            #     self.feature_gen_set += FeatureGeneratorSet(
            #         FeatureGenerator.discrete(
            #             name=name, lower=lower, upper=upper, step=self._dose_step)
            #         for name, lower, upper in (
            #             ('dose_change', -max(self._dose_range),
            #              max(self._dose_range)),
            #             ('dose_history', *self._dose_range),
            #             ('daily_dose_history', *self._dose_range),
            #         )
            #     )
            # else:
            #     self._dose_mode = 'dose'
            #     if not self._dose_step:
            #         raise ValueError(
            #             '`dose_step` missing for `decision_mode`=dose.')
            #     self.feature_gen_set += FeatureGeneratorSet(
            #         FeatureGenerator.discrete(
            #             name=name, lower=lower, upper=upper, step=self._dose_step)
            #         for name, lower, upper in (
            #             ('dose', *self._dose_range),
            #             ('dose_history', *self._dose_range),
            #             ('daily_dose_history', *self._dose_range),
            #         )
            #     )

            self.action_gen_set += self.feature_gen_set['dose_percent_change_joint']

            # if self._duration_mode:
            #     self.action_gen_set += self.feature_gen_set['duration']

        else:
            if 'dose_percent_change' in decision_mode:
                self._dose_mode = 'dose_percent_change'

                if self._decision_values is None:
                    raise ValueError(
                        '`decision_values` missing for '
                        '`decision_mode`=dose_percent_change.')

                if isinstance(self._decision_values[0], tuple):
                    raise ValueError(
                        '`decision_values` should be a tuple of floats for '
                        '`decision_mode`=dose_percent_change.')

                if self._decision_range is None:
                    raise ValueError(
                        '`decision_range` missing for '
                        '`decision_mode`=dose_percent_change.')

                if isinstance(self._decision_range[0], tuple):
                    raise ValueError(
                        '`decision_range` should be a tuple of floats for '
                        '`decision_mode`=dose_percent_change.')

                # TODO: This needs to be investigated! I am not sure if tuple
                # can be in fixed values and in ranges.
                self.feature_gen_set += FeatureGeneratorSet(
                    FeatureGenerator.numerical_fixed_values(
                        name='dose_percent_change',
                        fixed_values=self._decision_values,  # type: ignore
                        lower=self._decision_range[0],
                        upper=self._decision_range[1])  # type: ignore
                )

                if self._round_to_step:
                    if self._dose_step is None:
                        raise ValueError(
                            '`round_to_step` is `True`. `dose_step` is required.')
                    self.feature_gen_set += FeatureGeneratorSet(
                        FeatureGenerator.discrete(
                            name=name, lower=lower, upper=upper, step=self._dose_step)
                        for name, lower, upper in (
                            ('dose_history', *self._dose_range),
                            ('daily_dose_history', *self._dose_range),
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

            elif 'dose_change' in decision_mode:
                self._dose_mode = 'dose_change'
                if not self._dose_step:
                    raise ValueError(
                        '`dose_step` missing for `decision_mode`=dose_change.')
                self.feature_gen_set += FeatureGeneratorSet(
                    FeatureGenerator.discrete(
                        name=name, lower=lower, upper=upper, step=self._dose_step)
                    for name, lower, upper in (
                        ('dose_change', -max(self._dose_range), max(self._dose_range)),
                        ('dose_history', *self._dose_range),
                        ('daily_dose_history', *self._dose_range),
                    )
                )
            else:
                self._dose_mode = 'dose'
                if not self._dose_step:
                    raise ValueError('`dose_step` missing for `decision_mode`=dose.')
                self.feature_gen_set += FeatureGeneratorSet(
                    FeatureGenerator.discrete(
                        name=name, lower=lower, upper=upper, step=self._dose_step)
                    for name, lower, upper in (
                        ('dose', *self._dose_range),
                        ('dose_history', *self._dose_range),
                        ('daily_dose_history', *self._dose_range),
                    )
                )

            self.action_gen_set += self.feature_gen_set[self._dose_mode]

            if self._duration_mode:
                self.action_gen_set += self.feature_gen_set['duration']

        DosingSubject._generate_state_defs(self)

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        actions_taken = config['internal_states'].pop(
            '_actions_taken', [])

        instance = super().from_config(config)
        for action in actions_taken:
            a: FeatureSet = deserialize(action)  # type: ignore
            instance.take_effect(a)

        return instance

    def get_config(self) -> dict[str, Any]:
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
            dose_increment: float = 0.5) -> list[float]:

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
        duration = None
        if 'dose' in action_temp:
            current_dose = float(action_temp['dose'])  # type: ignore
        elif 'dose_change' in action_temp:
            current_dose = (
                self._decision_points_dose_history[self._decision_points_index - 1] +
                action_temp['dose_change'])  # type: ignore
        elif 'dose_percent_change' in action_temp:
            current_dose = (
                self._decision_points_dose_history[self._decision_points_index - 1] *
                (1 + action_temp['dose_percent_change']))  # type: ignore
            if self._round_to_step:
                current_dose = \
                    self._dose_range[0] + self._dose_step * round(  # type: ignore
                        (current_dose - self._dose_range[0]) / self._dose_step)
        elif 'dose_percent_change_joint' in action_temp:
            current_dose = (
                self._decision_points_dose_history[self._decision_points_index - 1] *
                (1 + action_temp['dose_percent_change_joint'][0]))  # type: ignore
            if self._round_to_step:
                current_dose = \
                    self._dose_range[0] + self._dose_step * round(  # type: ignore
                        (current_dose - self._dose_range[0]) / self._dose_step)
            duration = action_temp['dose_percent_change_joint'][1]  # type: ignore
        else:
            raise ValueError(
                'dose/ dose_change/ dose_percent_change/ dose_percent_change_joint not found.')

        if duration is None:
            duration = action_temp.get('duration', self._default_duration)
        current_duration = min(
            duration, self._max_day - self._day)  # type: ignore

        measurements_temp = self._patient.model(
            dose={
                i: current_dose
                for i in range(self._day, self._day + current_duration)
            },
            measurement_days=list(
                range(self._day, self._day + current_duration + 1)
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
        _list: list[float]
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
