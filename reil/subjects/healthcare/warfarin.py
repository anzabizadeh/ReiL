# -*- coding: utf-8 -*-  pylint: disable=undefined-variable
'''
warfarin class
==============

This `warfarin` class implements a two compartment PK/PD model for warfarin.
'''

import pathlib
from typing import Any, Dict, List, Optional, Tuple

from reil.datatypes import ReilData
from reil.subjects import Subject, healthcare
from reil.utils import action_generator, reil_functions


class Warfarin(Subject):
    '''
    A warfarin subject based on Hamberg's two compartment PK/PD model.
    '''

    def __init__(self,
                 patient: healthcare.Patient,
                 action_generator: action_generator.ActionGenerator,
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

        if 'dose' not in action_generator.components:
            raise ValueError(
                'action_generator should have a "dose" component.')
        if 'interval' not in action_generator.components:
            raise ValueError(
                'action_generator should have an "interval" component.')

        self._action_generator = action_generator
        self._max_day = max_day
        patient_basic = (('age', {}), ('CYP2C9', {}),
                         ('VKORC1', {}), ('sensitivity', {}))
        patient_extra = (('weight', {}), ('height', {}),
                         ('gender', {}), ('tobaco', {}),
                         ('amiodarone', {}), ('fluvastatin', {}))

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

        statistic_PTTR = reil_functions.PercentInRange(
            name='PTTR', arguments=('daily_INR_history',),  # type: ignore
            length=-1, multiplier=1.0, retrospective=True, interpolate=False,
            acceptable_range=(2, 3), exclude_first=True)

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

        self.statistic.add_definition(
            'PTTR_exact_basic', statistic_PTTR, 'daily_INR', 'patient_basic')

        self.statistic.add_definition(
            'PTTR_exact', statistic_PTTR, 'daily_INR', 'patient')

        self.reset()

    @classmethod
    def _empty_instance(cls):
        return cls(None, None)  # type: ignore

    @staticmethod
    def generate_dose_actions(min_dose: float = 0.0,
                              max_dose: float = 15.0,
                              dose_increment: float = 0.5) -> List[float]:

        return list(min_dose + x * dose_increment
                    for x in range(
                        int((max_dose - min_dose)/dose_increment) + 1))

    @staticmethod
    def generate_interval_actions(min_interval: int = 1,
                                  max_interval: int = 28,
                                  interval_increment: int = 1) -> List[int]:

        return list(range(min_interval, max_interval, interval_increment))

    def is_terminated(self, _id: Optional[int] = None) -> bool:
        return self._day >= self._max_day

    def possible_actions(
            self, _id: Optional[int] = None) -> Tuple[ReilData, ...]:
        return self._action_generator.possible_actions(
            self.state('default', _id))

    def take_effect(self,
                    action: ReilData,
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
            self, _id: Optional[int] = None) -> ReilData:
        patient_features = self._patient.feature_set
        return ReilData([
            {'name': 'age',
             'value': patient_features['age'].value,
             'lower': patient_features['age'].lower,
             'upper': patient_features['age'].upper},
            {'name': 'CYP2C9',
             'value': patient_features['CYP2C9'].value,
             'categories': patient_features['CYP2C9'].categories},
            {'name': 'VKORC1',
             'value': patient_features['VKORC1'].value,
             'categories': patient_features['VKORC1'].categories}],
            lazy_evaluation=True)

    def _numerical_sub_comp(self, name):
        temp = self._patient.feature_set[name]
        return {'name': name,
                'value': temp.value,
                'lower': temp.lower,
                'upper': temp.upper}

    def _categorical_sub_comp(self, name):
        temp = self._patient.feature_set[name]
        return {'name': name,
                'value': temp.value,
                'categories': temp.categories}

    def _sub_comp_age(self, _id: int, **kwargs: Any) -> Dict[str, Any]:
        return self._numerical_sub_comp('age')

    def _sub_comp_weight(self, _id: int, **kwargs: Any) -> Dict[str, Any]:
        return self._numerical_sub_comp('weight')

    def _sub_comp_height(self, _id: int, **kwargs: Any) -> Dict[str, Any]:
        return self._numerical_sub_comp('height')

    def _sub_comp_gender(self, _id: int, **kwargs: Any) -> Dict[str, Any]:
        return self._categorical_sub_comp('gender')

    def _sub_comp_race(self, _id: int, **kwargs: Any) -> Dict[str, Any]:
        return self._categorical_sub_comp('race')

    def _sub_comp_tobaco(self, _id: int, **kwargs: Any) -> Dict[str, Any]:
        return self._categorical_sub_comp('tobaco')

    def _sub_comp_amiodarone(self, _id: int, **kwargs: Any) -> Dict[str, Any]:
        return self._categorical_sub_comp('amiodarone')

    def _sub_comp_fluvastatin(self, _id: int, **kwargs: Any) -> Dict[str, Any]:
        return self._categorical_sub_comp('fluvastatin')

    def _sub_comp_CYP2C9(self, _id: int, **kwargs: Any) -> Dict[str, Any]:
        return self._categorical_sub_comp('CYP2C9')

    def _sub_comp_VKORC1(self, _id: int, **kwargs: Any) -> Dict[str, Any]:
        return self._categorical_sub_comp('VKORC1')

    def _sub_comp_sensitivity(self, _id: int, **kwargs: Any) -> Dict[str, Any]:
        return self._categorical_sub_comp('sensitivity')

    def _get_history(
            self, list_name: str, length: int) -> Tuple[List[Any], Any, Any]:
        if length == 0:
            raise ValueError('length should be a positive integer, or '
                             '-1 for full length output.')

        if list_name == 'INR_history':
            _list = self._decision_points_INR_history
            index = self._decision_points_index + 1
            filler = 0.0
            lower, upper = 0.0, 15.0
        elif list_name == 'daily_INR_history':
            _list = self._full_INR_history
            index = self._day + 1
            filler = 0.0
            lower, upper = 0.0, 15.0
        elif list_name == 'dose_history':
            _list = self._decision_points_dose_history
            index = self._decision_points_index
            filler = 0.0
            lower = self._action_generator.lower['dose']
            upper = self._action_generator.upper['dose']
        elif list_name == 'daily_dose_history':
            _list = self._full_dose_history
            index = self._day
            filler = 0.0
            lower = self._action_generator.lower['dose']
            upper = self._action_generator.upper['dose']
        elif list_name == 'interval_history':
            _list = self._decision_points_interval_history
            index = self._decision_points_index
            filler = 1
            lower = self._action_generator.lower['interval']
            upper = self._action_generator.upper['interval']
        else:
            return [], None, None

        if length == -1:
            result = _list[:index]
        else:
            if length > index:
                i1, i2 = length - index, 0
            else:
                i1, i2 = 0, index-length
            result = [filler] * i1 + _list[i2:index]  # type: ignore

        return result, lower, upper

    def _sub_comp_dose_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Dict[str, Any]:
        name = 'dose_history'
        value, lower, upper = self._get_history(name, length)
        return {'name': name,
                'value': tuple(value),
                'lower': lower,
                'upper': upper}

    def _sub_comp_INR_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Dict[str, Any]:
        name = 'INR_history'
        value, lower, upper = self._get_history(name, length)
        return {'name': name,
                'value': tuple(value),
                'lower': lower,
                'upper': upper}

    def _sub_comp_interval_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Dict[str, Any]:
        name = 'interval_history'
        value, lower, upper = self._get_history(name, length)
        return {'name': name,
                'value': tuple(value),
                'lower': lower,
                'upper': upper}

    def _sub_comp_day(self, _id: int, **kwargs: Any) -> Dict[str, Any]:
        return {'name': 'day',
                'value': self._day if 0 <= self._day < self._max_day else None,
                'lower': 0,
                'upper': self._max_day - 1}

    def _sub_comp_daily_dose_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Dict[str, Any]:
        name = 'daily_dose_history'
        value, lower, upper = self._get_history(name, length)
        return {'name': name,
                'value': tuple(value),
                'lower': lower,
                'upper': upper}

    def _sub_comp_daily_INR_history(
            self, _id: int, length: int = 1, **kwargs: Any) -> Dict[str, Any]:
        name = 'daily_INR_history'
        value, lower, upper = self._get_history(name, length)
        return {'name': name,
                'value': tuple(value),
                'lower': lower,
                'upper': upper}

    def _sub_comp_INR_within(
            self, _id: int, length: int = 1, **kwargs: Any) -> Dict[str, Any]:
        name = 'daily_INR'
        l, _, _ = self._get_history('interval_history', length)
        value, lower, upper = self._get_history(name, sum(l))

        return {'name': name,
                'value': tuple(value),
                'lower': lower,
                'upper': upper}

    def load(self, filename: str, path: Optional[pathlib.Path] = None) -> None:
        '''
        Extends super class's method to make sure 'action_generator' resets if
        it is part of the 'persistent_attributes'.
        '''
        super().load(filename, path)
        if '_action_generator' in self._persistent_attributes:
            self._action_generator.reset()

    def __repr__(self) -> str:
        try:
            temp = ', '.join(''.join(
                (str(k), ': ',
                 ('{:4.2f}' if v.is_numerical else '{}').format(v.value)))
                for k, v in self._patient.feature_set.items())
        except (AttributeError, ValueError, KeyError):
            temp = ''

        return (f'{self.__class__.__qualname__} [{temp}]')

    # if self._action_type == 'dose' and self._current_dose < self._max_dose:
    #     return self._possible_actions[
    #         :int(self._current_dose/self._dose_steps) + 1]
    # elif (self._action_type == 'interval' and
    #         self._current_interval < self._max_interval):
    #     return self._possible_actions[
    #         :int(self._current_interval/self._interval_steps)]
    # elif (self._current_dose < self._max_dose or
    #         self._current_interval < self._max_interval):
    #     self._generate_possible_actions(
    #         dose_cap=self._current_dose, interval_cap=self._current_interval)

    # return self._possible_actions

    # def default_state(self, _id: Optional[int] = None) -> reildata.ReilData:
    #     index = self._decision_points_index - 1
    #     return self._state_normal_fixed + reildata.ReilData([
    #         {'name': 'dose',
    #          'value': (self._decision_points_dose_history[index],),
    #          'lower': self._action_generator.lower['dose'],
    #          'upper': self._action_generator.upper['dose']},
    #         {'name': 'INR',
    #          'value': (self._decision_points_INR_history[index],),
    #          'lower': 0.0,
    #          'upper': 15.0},
    #         {'name': 'interval',
    #          'value': (self._decision_points_interval_history[index],),
    #          'lower': self._action_generator.lower['interval'],
    #          'upper': self._action_generator.upper['interval']}],
    #         lazy_evaluation=True)


# def stats(self, stats_list: Tuple[str, ...]) -> Dict[str, Any]:
#     results = {}
#     for s in stats_list:
#         if s == 'TTR':
#             temp = reil_functions.Functions.TTR(self._full_INR_history)
#             # INR = self._full_INR_history
#             # sum(
#             #     (1 if 2.0 <= INRi <= 3.0 else 0 for INRi in INR)
#             #       ) / len(INR)
#         elif s == 'dose_change':
#             temp = reil_functions.Functions.dose_change_count(
#                 self._full_dose_history)
#             # temp = sum(x != self._full_dose_history[i+1]
#             #            for i, x in enumerate(self._full_dose_history[:-1]))
#         elif s == 'delta_dose':
#             temp = reil_functions.Functions.delta_dose(
#                 self._full_dose_history)
#             # temp = sum(abs(x-self._full_dose_history[i+1])
#             #            for i, x in enumerate(self._full_dose_history[:-1]))
#         else:
#             self._logger.warning(
#                 f'WARNING! {s} is not one of the available stats!')
#             continue

#         results[s] = temp

#     results['ID'] = reildata.ReilData([
#         {'name': 'age',
#          'value': self._patient.feature_set['age'].value,
#          'lower': self._patient.feature_set['age'].lower,
#          'upper': self._patient.feature_set['age'].upper},
#         {'name': 'CYP2C9',
#          'value': self._patient.feature_set['CYP2C9'].value,
#          'categories': self._patient.feature_set['CYP2C9'].categories},
#         {'name': 'VKORC1',
#          'value': self._patient.feature_set['VKORC1'].value,
#          'categories': self._patient.feature_set['VKORC1'].categories}],
#         lazy_evaluation=True)

#     return results

# @property
# def _INR_history(self) -> List[float]:
#     # INR has one more value (initial INR) compared to dose.
#     return [0.0]*(self._INR_history_length - self._decision_points_index) \
#         + self._decision_points_INR_history[
#             :self._decision_points_index + 1][-self._INR_history_length - 1:]

# @property
# def _dose_history(self) -> List[float]:
#     return [0.0]*(self._dose_history_length - self._decision_points_index) \
#         + self._decision_points_dose_history[
#             :self._decision_points_index][-self._dose_history_length:] \
#         + ([self._current_dose]
#            if self._action_type == 'interval_only' else [])

# @property
# def _interval_history(self) -> List[int]:
#     return [0]*(
#           self._interval_history_length - self._decision_points_index) \
#         + self._decision_points_interval_history[
#             :self._decision_points_index][-self._interval_history_length:] \
#         + ([self._current_interval]
#            if self._action_type == 'dose_only' else [])

# @property
# def _interval_history_length(self) -> int:
#     return max(self._dose_history_length, self._INR_history_length)

# @property
# def state(self) -> reildata.ReilData:
#     if self._ex_protocol_current['state'] == 'extended':
#         return self._state_extended()
#     else:
#         return self._state_normal()

# def _prepare_reward_arguments(self,
#                               arguments: Tuple[str, ...],
#                               observation_length: int,
#                               retrospective: bool,
#                               interpolate: bool) -> Dict[str, Any]:
#     output = {'y': [], 'x': []}
#     if retrospective:
#         if interpolate:
#             if observation_length == -1:
#                 start = 0
#             else:
#                 start = self._decision_points_index - observation_length + 1
#             end = self._decision_points_index + 1
#             if 'INR' in arguments:
#                 output = {
#                     'y': self._decision_points_INR_history[start:end],
#                     'x': self._decision_points_interval_history[start:end-1]}
#             elif 'Doses' in arguments:
#                 output = {
#                     'y': self._decision_points_dose_history[start:end],
#                     'x': self._decision_points_interval_history[start:end-1]}
#         else:
#             if observation_length == -1:
#                 start = 0
#             else:
#                 start = self._day - observation_length + 1
#             end = self._day + 1
#             if 'INR' in arguments:
#                 output = {'y': self._full_INR_history[start:end]}
#             elif 'Doses' in arguments:
#                 output = {'y': self._full_dose_history[start:end]}

#     else:
#         start = self._day + 1
#         if observation_length == -1:
#             end = self._max_day
#         else:
#             end = min(self._day + observation_length + 1, self._max_day)

#         current_dose = self._decision_points_dose_history[
#             self._decision_points_index]

#         if interpolate:
#             if 'INR' in arguments:
#                 temp_patient = copy.deepcopy(self._patient)
#                 INRs_temp = temp_patient.model(
#                     dose=dict((i, current_dose)
#                               for i in range(start, end)),
#                     measurement_days=[start, end])['INR']
#                 output = {'y': INRs_temp,
#                           'x': [start, end]}
#             elif 'Doses' in arguments:
#                 output = {'y': [current_dose] * 2,
#                           'x': [start, end]}
#         else:
#             if 'INR' in arguments:
#                 temp_patient = copy.deepcopy(self._patient)
#                 INRs_temp = temp_patient.model(
#                     dose=dict((i, current_dose)
#                               for i in range(start, end)),
#                     measurement_days=list(range(start, end)))['INR']
#                 output = {'y': INRs_temp}
#             elif 'Doses' in arguments:
#                 output = {'y': [current_dose] * (end - start)}

#     return output

# def _state_extended(self) -> reildata.ReilData:
#     return self._state_extended_fixed + reildata.ReilData([
#         {'name': 'day',
#          'value': self._day if 0 <= self._day < self._max_day else None,
#          'lower': 0,
#          'upper': self._max_day - 1},
#         {'name': 'Doses',
#          'value': tuple(self._dose_history),
#          'lower': self._action_generator.lower['dose'],
#          'upper': self._action_generator.upper['dose']},
#         {'name': 'INR',
#          'value': tuple(self._INR_history),
#          'lower': 0.0,
#          'upper': 15.0},
#         {'name': 'Intervals',
#          'value': tuple(self._interval_history),
#          'lower': self._action_generator.lower['interval'],
#          'upper': self._action_generator.upper['interval']}],
#         lazy_evaluation=True)

# def _state_normal(self) -> reildata.ReilData:
#     return self._state_normal_fixed + reildata.ReilData([
#         {'name': 'Doses',
#          'value': tuple(self._dose_history),
#          'lower': self._action_generator.lower['dose'],
#          'upper': self._action_generator.upper['dose']},
#         {'name': 'INR',
#          'value': tuple(self._INR_history),
#          'lower': 0.0,
#          'upper': 15.0},
#         {'name': 'Intervals',
#          'value': tuple(self._interval_history),
#          'lower': self._action_generator.lower['interval'],
#          'upper': self._action_generator.upper['interval']}],
#         lazy_evaluation=True)
