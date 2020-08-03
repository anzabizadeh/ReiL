# -*- coding: utf-8 -*-  pylint: disable=undefined-variable
'''
warfarin_model class
==================

This `warfarin_model` class implements a two compartment PK/PD model for warfarin. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import collections
import copy
import numbers
import os
import pathlib
import random
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from rl import rldata, subjects, utils


class WarfarinModel_v5(subjects.Subject):
    '''
    A warfarin subject based on Hamberg's two compartment PK/PD model for wafarin.

    Attributes
    ----------
        state: the state of the subject as a ValueSet.
        is_terminated: whether the subject is finished or not.
        possible_actions: a list of possible actions.

    Methods
    -------
        register: register a new agent and return its ID or return ID of an existing agent.
        take_effect: get an action and change the state accordingly.
        reset: reset the state and is_terminated.
    '''

    def __init__(self,
                 stats_list: Sequence[str] = ['TTR', 'dose_change', 'delta_dose'],

                 max_day: int = 90,
                 therapeutic_range: Tuple[int, int] = (2, 3),
                 action_type: str = 'dose only',
                 INR_history_length: int = 1,
                 INR_penalty_coef: float = 1.0,
                 dose_history_length: int = 1,
                 max_dose: float = 15.0,
                 dose_steps: float = 0.5,
                 dose_change_penalty_coef: float = 1.0,
                 dose_change_penalty_func: Callable[[Sequence[float]], float] =
                 lambda x: int(x[-2] != x[-1]),  # -0.2 * abs(x[-2]-x[-1]),
                 interval: Sequence[int] = [1],
                 max_interval: Optional[int] = None,
                 interval_max_dose: Union[Sequence[float], float] = [15.0],
                 lookahead_duration: int = 0,
                 lookahead_penalty_coef: float = 0,
                 characteristics: Dict[str, Any] = {'age': 71, 'weight': 199.24, 'height': 66.78,
                                                    'CYP2C9': '*1/*1', 'VKORC1': 'A/A',
                                                    'gender': 'Male', 'race': 'White',
                                                    'tobaco': 'No',
                                                    'amiodarone': 'No', 'fluvastatin': 'No'},
                 patient_selection: str = 'random',
                 randomized: bool = True,
                 save_patients: bool = False,
                 patients_save_path: Union[str, pathlib.Path] = './patients',
                 patients_save_prefix: str = 'warfv5',
                 patient_save_overwrite: bool = False,
                 patient_use_existing: bool = True,
                 patient_counter_start: int = 0,
                 **kwargs):
        '''
        Create a warfarin model.
        \nArguments:
        \n  max_day: maximum number of days of simulation (Default: 90)
        \n  therapeutic_range: acceptable range of INR values (Default: (2, 3))
        \n  action_type: type of action among 'dose only', 'interval only', and 'both' (Default = 'dose only')
        \n  INR_history_length: number of days of history for INRs (Default: 1)
        \n  INR_penalty_coef: Coefficient of INR penalty in reward calculation (Default = 1.0)
        \n  dose_history_length: number of days of history for doses (Default: 1)
        \n  max_dose: maximum possible dose (Default: 15.0 mg/day)
        \n  dose_steps: minimum possible change in dose (Default: 0.5)
        \n  dose_change_penalty_coef: Coefficient of dose change penalty in reward calculation. (Default: 1.0)
        \n  dose_change_penalty_func: a function that computes penalty of changing dose from its previous value. The
        \n      argument for this function is the list of previous doses and should return a scalar. (Default: lambda x: int(x[-2] != x[-1]))
        \n  interval: a list consisting of INR measurement intervals. If the list covers less than max_day, the last element is repeated.
        \n      If a negative value is provided, the positive value of that interval is repeated until INR gets in the therapeutic_range. (Default = [1])
        \n  max_interval: maximum number of days between two INR testing/ dosing. Default value use max_day. (Default = None)
        \n  interval_max_dose: a list consisting of maximum dose allowed per interval. (Default = [15.0])
        \n  lookahead_duration: (Default = 0)
        \n  lookahead_penalty_coef: (Default = 0)
        \n  stats_list: a list of statistics to compute for each patient (Default = ['TTR', 'dose_change', 'delta_dose'])
        \n  characteristics: a dictionary describing the patient. (Default: {'age': 71, 'weight': 199.24, 'height': 66.78, 'gender': 'Male',
        \n      'race': 'White', 'tobaco': 'No', 'amiodarone': 'No', 'fluvastatin': 'No', 'CYP2C9': '*1/*1', 'VKORC1': 'A/A'})
        \n  patient_selection: how to generate patients. One of 'random', 'ravvaz' and 'fixed'. (Default: 'random')
        \n  randomized: whether to have random effect in the PK/PD model (Default: True)
        \n  save_patients: should the generated patients be saved? (Default: False)
        \n  patients_save_path: where to save patient files (Default: './patients')
        \n  patients_save_prefix: prefix to use when saving patients (Default: 'warfv5')
        \n  patient_save_overwrite: overwrite currently saved patients? (Default: False)
        \n  patient_use_existing: use currently saved patients if they exist? (Default: True)
        \n  patient_counter_start: the starting value for patient filename counter (Default: 0)
        '''

        kwargs['name'] = kwargs.get('name', __name__)
        kwargs['logger_name'] = kwargs.get('logger_name', __name__)

        super().__init__(stats_list=stats_list, **kwargs)

        self._list_of_characteristics = {
            'age': (18, 100),  # similar to the range of 10k sample from Ravvaz
            # (lb) similar to the range of 10k sample from Ravvaz
            'weight': (70, 500),
            # (in) similar to the range of 10k sample from Ravvaz
            'height': (45, 85),
            'gender': ('Female', 'Male'),
            'race': ('White', 'Black', 'Asian', 'American Indian', 'Pacific Islander'),
            'tobaco': ('No', 'Yes'),
            'amiodarone': ('No', 'Yes'),
            'fluvastatin': ('No', 'Yes'),
            'CYP2C9': ('*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3'),
            'VKORC1': ('G/G', 'G/A', 'A/A')}

        self._list_of_probabilities = {
            'age': (67.3, 14.43),  # lb  - Aurora population
            # in - Aurora population
            'weight': (199.24, 54.71),
            # Aurora population
            'height': (66.78, 4.31),
            # Aurora population
            'gender': (0.5314, 0.4686),
            # Aurora Avatar Population
            'race': (0.9522, 0.0419, 0.0040, 0.0018, 1e-4),
            # Aurora Avatar Population
            'tobaco': (0.9067, 0.0933),
            # Aurora Avatar Population
            'amiodarone': (0.8849, 0.1151),
            # Aurora Avatar Population
            'fluvastatin': (0.9998, 0.0002),
            # Aurora Avatar Population
            'CYP2C9': (0.6739, 0.1486, 0.0925, 0.0651, 0.0197, 2e-4),
            'VKORC1': (0.3837, 0.4418, 0.1745)}  # Aurora Avatar Population

        self._max_day = max_day + lookahead_duration
        self._max_time = (max_day + 1) * 24
        self._therapeutic_range = therapeutic_range
        self._action_type = action_type

        self._INR_history_length = INR_history_length
        self._INR_penalty_coef = INR_penalty_coef

        self._dose_history_length = dose_history_length
        self._max_dose = max_dose
        self._dose_steps = dose_steps
        self._dose_change_penalty_coef = dose_change_penalty_coef
        self._dose_change_penalty_func = dose_change_penalty_func

        self._interval = [x if x != 0 else 1 for x in interval]
        if self._interval != interval:
            self._logger.warning('Replaced zero-day intervals with 1.')

        self._max_interval = max_day if max_interval is None else max_interval
        self._interval_max_dose = interval_max_dose

        if isinstance(self._interval_max_dose, numbers.Number):
            self._interval_max_dose = [
                self._interval_max_dose] * len(self._interval)
        elif len(self._interval_max_dose) == 1:
            self._interval_max_dose = self._interval_max_dose * \
                len(self._interval)
        elif len(self._interval_max_dose) != len(self._interval):
            self._logger.warning(
                'interval_max_dose does not match "interval" in length. max_dose will be used for intervals without interval_max_dose values.')

        self._lookahead_duration = lookahead_duration
        self._lookahead_penalty_coef = lookahead_penalty_coef

        self._stats_list = stats_list
        self._characteristics = characteristics
        self._patient_selection = patient_selection
        if self._patient_selection in ('ravvaz', 'ravvaz 2017', 'ravvaz_2017', 'ravvaz2017'):
            if self._list_of_characteristics['CYP2C9'] != ('*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3') or \
                    self._list_of_characteristics['VKORC1'] != ('G/G', 'G/A', 'A/A'):
                raise ValueError(
                    'For Ravvaz patient generation, CYP2C9 and VKORC1 should not be changed!')

        self._randomized = randomized
        self._save_patients = save_patients
        self._patients_save_path = patients_save_path
        self._patients_save_prefix = patients_save_prefix
        self._patient_save_overwrite = patient_save_overwrite
        self._patient_use_existing = patient_use_existing
        self._patient_counter_start = patient_counter_start
        self._filename_counter = self._patient_counter_start

        # this makes sure that if some elements of dictionaries
        # (e.g. characteristics, list_of_...) changes by the user,
        # other elements of the dictionary remain intact.
        # for key, value in kwargs.items():
        #     if isinstance(value, dict):
        #         try:
        #             temp = self._defaults[key]
        #             temp.update(kwargs[key])
        #             kwargs[key] = temp
        #         except KeyError:
        #             pass

        if self._save_patients:
            if not self._patient_save_overwrite and not self._patient_use_existing:
                while os.path.exists(os.path.join(self._patients_save_path,
                                                  f'{self._patients_save_prefix}{self._filename_counter:06}')):
                    self._filename_counter += 1

        self.reset()

        self._INR_mid = (
            self._therapeutic_range[1] + self._therapeutic_range[0]) / 2
        self._INR_range = self._therapeutic_range[1] - self._therapeutic_range[0]

        if self._action_type.lower() in ('dose', 'dose only', 'dose_only', 'only dose', 'only_dose'):
            self._possible_actions = rldata.RLData(({
                'name': f'dose: {x * self._dose_steps:.1f}',
                'value': x * self._dose_steps,
                'lower': 0, 'upper': self._max_dose}
                for x in range(int(self._max_dose/self._dose_steps) + 1))).split()

        elif self._action_type.lower() in ('interval', 'interval only', 'interval_only', 'only interval', 'only_interval'):
            self._possible_actions = rldata.RLData(({
                'name': f'interval: {x}',
                'value': x,
                'lower': 1, 'upper': self._max_interval}
                for x in range(1, self._max_interval + 1))).split()
                
        else:
            self._possible_actions = rldata.RLData(({
                'name': f'dose: {x * self._dose_steps:.1f}\tinterval: {i}',
                'value': (x*self._dose_steps, i),
                'lower': (0, 1), 'upper': (self._max_dose, self._max_interval)}
                for x in range(int(self._max_dose/self._dose_steps) + 1)
                for i in range(1, self._max_interval + 1))).split()  # TODO: needs a normalizer

    @property
    def _INR_history(self):
        # INR has one more value (initial INR) compared to dose.
        return [0]*(self._INR_history_length - self._decision_points_index) \
            + self._decision_points_INR_history[:self._decision_points_index + 1][-self._INR_history_length - 1:]

    @property
    def _dose_history(self):
        return [0]*(self._dose_history_length-self._decision_points_index) \
            + self._decision_points_dose_history[:self._decision_points_index][-self._dose_history_length:]

    @property
    def _interval_history(self):
        return [0]*(self._interval_history_length-self._decision_points_index) \
            + self._decision_points_interval_history[:self._decision_points_index][-self._interval_history_length:]

    @property
    def state(self) -> rldata.RLData:
        if self._ex_protocol_current['state'] == 'extended':
            return self._state_extended()
        else:
            return self._state_normal()

    @property
    def complete_state(self) -> rldata.RLData:
        return rldata.RLData([
            {'name': 'age',
             'value': self._characteristics['age'],
             'lower': self._list_of_characteristics['age'][0],
             'upper': self._list_of_characteristics['age'][-1]},
            {'name': 'weight',
             'value': self._characteristics['weight'],
             'lower': self._list_of_characteristics['weight'][0],
             'upper': self._list_of_characteristics['weight'][-1]},
            {'name': 'height',
             'value': self._characteristics['height'],
             'lower': self._list_of_characteristics['height'][0],
             'upper': self._list_of_characteristics['height'][-1]},
            {'name': 'gender',
             'value': self._characteristics['gender'],
             'categories': self._list_of_characteristics['gender']},
            {'name': 'race',
             'value': self._characteristics['race'],
             'categories': self._list_of_characteristics['race']},
            {'name': 'tobaco',
             'value': self._characteristics['tobaco'],
             'categories': self._list_of_characteristics['tobaco']},
            {'name': 'amiodarone',
             'value': self._characteristics['amiodarone'],
             'categories': self._list_of_characteristics['amiodarone']},
            {'name': 'fluvastatin',
             'value': self._characteristics['fluvastatin'],
             'categories': self._list_of_characteristics['fluvastatin']},
            {'name': 'CYP2C9',
             'value': self._characteristics['CYP2C9'],
             'categories': self._list_of_characteristics['CYP2C9']},
            {'name': 'VKORC1',
             'value': self._characteristics['VKORC1'],
             'categories': self._list_of_characteristics['VKORC1']},
            {'name': 'day',
             'value': self._day,
             'lower': 0,
             'upper': self._max_day - 1},
            {'name': 'Doses',
             'value': tuple(self._full_dose_history),
             'lower': 0.0,
             'upper': self._max_dose},
            {'name': 'INRs',
             'value': tuple(self._full_INR_history),
             'lower': 0.0,
             'upper': 15.0},
            {'name': 'Intervals',
             'value': tuple(self._decision_points_interval_history[:self._decision_points_index]),
             'lower': 0,
             'upper': self._max_interval}],
            lazy_evaluation=True)

    @property
    def is_terminated(self) -> bool:
        return self._day >= self._max_day - self._lookahead_duration

    @property
    # only considers the dose
    def possible_actions(self) -> rldata.RLData:
        try:
            if self._interval_max_dose[self._interval_index] < self._max_dose:
                return rldata.RLData(({
                    'name': f'dose: {x * self._dose_steps:.1f}',
                    'value': x * self._dose_steps,
                    'lower': 0, 'upper': self._max_dose}
                    for x in range(int(self._interval_max_dose[self._interval_index]/self._dose_steps) + 1)),
                    lazy_evaluation=True).split()
        except IndexError:
            # When self._interval_index >= len(self._interval_max_dose), use self._max_dose instead of self._interval_max_dose[self._interval_index]
            pass

        return self._possible_actions

    def register(self, agent_name) -> int:
        '''
        Registers an agent and returns its ID. If the agent is new, a new ID is generated and the agent_name is added to agent_list.
        \nArguments:
        \n    agent_name: the name of the agent to be registered.
        '''
        try:
            return self._agent_list[agent_name]
        except KeyError:
            if len(self._agent_list) == 1:
                raise ValueError('Only one drug is allowed.')
            self._agent_list[agent_name] = 1
            return 1

    def take_effect(self, action: rldata.RLData, _id: Optional[int] = None) -> rldata.RLData:
        current_dose = action[0].value
        try:  # use the provided action, otherwise the interval
            current_interval = min(action[1].value, self._max_day - self._day)
        except IndexError:
            current_interval = min(
                self._get_next_interval(), self._max_day - self._day)

        self._patient.dose = dict(
            tuple((i + self._day, current_dose) for i in range(current_interval)))

        self._decision_points_dose_history[self._decision_points_index] = current_dose
        self._decision_points_interval_history[self._decision_points_index] = current_interval
        self._decision_points_index += 1

        day_temp = self._day
        self._day += current_interval

        self._full_dose_history[day_temp:self._day] = [
            current_dose] * current_interval
        self._full_INR_history[day_temp + 1:self._day +
                               1] = self._patient.INR(list(range(day_temp + 1, self._day + 1)))

        self._decision_points_INR_history[self._decision_points_index] = self._full_INR_history[self._day]

        try:
            if self.exchange_protocol['take_effect'] == 'standard':
                if self._lookahead_duration > 0:
                    temp_patient = copy.deepcopy(self._patient)
                    temp_patient.dose = dict(tuple(
                        (i + day_temp, current_dose) for i in range(1, self._lookahead_duration + 1)))
                    lookahead_penalty = -sum(((2 / self._INR_range * (self._INR_mid - INRi)) ** 2
                                              for INRi in temp_patient.INR(list(range(day_temp + 1, day_temp + self._lookahead_duration + 1)))))
                else:
                    lookahead_penalty = 0

                last_two_INRs = self._decision_points_INR_history[-2:]
                INR_penalty = -sum(((2 / self._INR_range * (self._INR_mid - last_two_INRs[-2] - (last_two_INRs[-1]-last_two_INRs[-2])/current_interval*j)) ** 2
                                    for j in range(1, current_interval + 1)))  # negative squared distance as reward (used *2/range to normalize)
                dose_change_penalty = - \
                    self._dose_change_penalty_func(self._dose_history)
                reward = self._INR_penalty_coef * INR_penalty \
                    + self._dose_change_penalty_coef * dose_change_penalty \
                    + self._lookahead_penalty_coef * lookahead_penalty
            elif self.exchange_protocol['take_effect'] == 'no_reward':
                reward = 0
            else:
                raise ValueError(
                    f'exchange_protocol for "take_effect" should be "standard" or "no_reward". Got {self.exchange_protocol["take_effect"]}.')

        except TypeError:
            reward = 0

        return rldata.RLData({'name': 'reward', 'value': reward})  # , normalizer=lambda x: x)

    def stats(self, stats_list: Union[Sequence[str], str]) -> Dict[str, Union[rldata.RLData, numbers.Number]]:
        if isinstance(stats_list, str):
            stats_list = [stats_list]
        results = {}
        for s in stats_list:
            if s == 'TTR':
                INRs = self._full_INR_history
                temp = sum(
                    (1 if 2.0 <= INRi <= 3.0 else 0 for INRi in INRs)) / len(INRs)
            elif s == 'dose_change':
                temp = sum(x != self._full_dose_history[i+1] for i, x in enumerate(self._full_dose_history[:-1]))
            elif s == 'delta_dose':
                temp = sum(abs(x-self._full_dose_history[i+1]) for i, x in enumerate(self._full_dose_history[:-1]))
            else:
                print(f'WARNING! {s} is not one of the available stats!')
                continue

            results[s] = temp

        results['ID'] = rldata.RLData([
            {'name': 'age',
             'value': self._characteristics['age'],
             'lower': self._list_of_characteristics['age'][0],
             'upper': self._list_of_characteristics['age'][-1]},
            {'name': 'CYP2C9',
             'value': self._characteristics['CYP2C9'],
             'categories': self._list_of_characteristics['CYP2C9']},
            {'name': 'VKORC1',
             'value': self._characteristics['VKORC1'],
             'categories': self._list_of_characteristics['VKORC1']}],
            lazy_evaluation=True)

        return results

    def reset(self) -> None:
        if self._patient_selection == 'random':
            self._generate_random_patient()
        elif self._patient_selection.lower() in ['ravvaz', 'ravvaz 2017', 'ravvaz_2017', 'ravvaz2017']:
            self._generate_ravvaz_patient()

        self._patient = utils.Patient(age=self._characteristics['age'],
                                      # Not in the patient model
                                      weight=self._characteristics['weight'],
                                      # Not in the patient model
                                      height=self._characteristics['height'],
                                      # Not in the patient model
                                      gender=self._characteristics['gender'],
                                      # Not in the patient model
                                      race=self._characteristics['race'],
                                      # Not in the patient model
                                      tobaco=self._characteristics['tobaco'],
                                      # Not in the patient model
                                      amiodarone=self._characteristics['amiodarone'],
                                      # Not in the patient model
                                      fluvastatin=self._characteristics['fluvastatin'],
                                      CYP2C9=self._characteristics['CYP2C9'],
                                      VKORC1=self._characteristics['VKORC1'],
                                      randomized=self._randomized, max_time=self._max_time)

        current_patient = ''.join(
            (self._patients_save_prefix, f'{self._filename_counter:06}'))
        if self._save_patients:
            if self._patient_save_overwrite:
                self._patient.save(
                    path=self._patients_save_path, filename=current_patient)
            else:
                try:
                    self._load_patient(current_patient)
                except FileNotFoundError:
                    self._patient.save(
                        path=self._patients_save_path, filename=current_patient)

        self._filename_counter += 1

        self._day = 0
        self._interval_history_length = max(self._dose_history_length, self._INR_history_length)
        self._full_INR_history = [0.0] * self._max_day
        self._full_dose_history = [0.0] * self._max_day
        self._decision_points_INR_history = [0.0] * (self._max_day + 1)
        self._decision_points_dose_history = [0.0] * self._max_day
        self._decision_points_interval_history = [0.0] * self._max_day
        self._decision_points_index = 0

        # The latest INR is also stored in self._INR_history
        # self._INR_history = collections.deque(
        #     [0.0]*(self._INR_history_length + 1))
        self._full_INR_history[0] = self._patient.INR([0])[-1]
        self._decision_points_INR_history[0] = self._full_INR_history[0]
        self._interval_index = 0

    def _state_extended(self) -> rldata.RLData:
        return rldata.RLData([
            {'name': 'age',
             'value': self._characteristics['age'],
             'lower': self._list_of_characteristics['age'][0],
             'upper': self._list_of_characteristics['age'][-1]},
            {'name': 'weight',
             'value': self._characteristics['weight'],
             'lower': self._list_of_characteristics['weight'][0],
             'upper': self._list_of_characteristics['weight'][-1]},
            {'name': 'height',
             'value': self._characteristics['height'],
             'lower': self._list_of_characteristics['height'][0],
             'upper': self._list_of_characteristics['height'][-1]},
            {'name': 'gender',
             'value': self._characteristics['gender'],
             'categories': self._list_of_characteristics['gender']},
            {'name': 'race',
             'value': self._characteristics['race'],
             'categories': self._list_of_characteristics['race']},
            {'name': 'tobaco',
             'value': self._characteristics['tobaco'],
             'categories': self._list_of_characteristics['tobaco']},
            {'name': 'amiodarone',
             'value': self._characteristics['amiodarone'],
             'categories': self._list_of_characteristics['amiodarone']},
            {'name': 'fluvastatin',
             'value': self._characteristics['fluvastatin'],
             'categories': self._list_of_characteristics['fluvastatin']},
            {'name': 'CYP2C9',
             'value': self._characteristics['CYP2C9'],
             'categories': self._list_of_characteristics['CYP2C9']},
            {'name': 'VKORC1',
             'value': self._characteristics['VKORC1'],
             'categories': self._list_of_characteristics['VKORC1']},
            {'name': 'day',
             'value': self._day,
             'lower': 0,
             'upper': self._max_day - 1},
            {'name': 'Doses',
             'value': tuple(self._dose_history),
             'lower': 0.0,
             'upper': self._max_dose},
            {'name': 'INRs',
             'value': tuple(self._INR_history),
             'lower': 0.0,
             'upper': 15.0},
            {'name': 'Intervals',
             'value': tuple(self._interval_history),
             'lower': 0,
             'upper': self._max_interval}],
            lazy_evaluation=True) 

    def _state_normal(self) -> rldata.RLData:
        return rldata.RLData([
            {'name': 'age',
             'value': self._characteristics['age'],
             'lower': self._list_of_characteristics['age'][0],
             'upper': self._list_of_characteristics['age'][-1]},
            {'name': 'CYP2C9',
             'value': self._characteristics['CYP2C9'],
             'categories': self._list_of_characteristics['CYP2C9']},
            {'name': 'VKORC1',
             'value': self._characteristics['VKORC1'],
             'categories': self._list_of_characteristics['VKORC1']},
            {'name': 'Doses',
             'value': tuple(self._dose_history),
             'lower': 0.0,
             'upper': self._max_dose},
            {'name': 'INRs',
             'value': tuple(self._INR_history),
             'lower': 0.0,
             'upper': 15.0},
            {'name': 'Intervals',
             'value': tuple(self._interval_history),
             'lower': 0,
             'upper': self._max_interval}],
            lazy_evaluation=True)

    def _get_next_interval(self) -> int:
        if self._interval[self._interval_index] < 0:
            if not (self._therapeutic_range[0] <= self._decision_points_INR_history[self._decision_points_index] <= self._therapeutic_range[1]):
                self._interval_index -= 1

        try:
            interval = self._interval[self._interval_index + 1]
            self._interval_index += 1
        except IndexError:
            interval = self._interval[self._interval_index]

        return abs(interval)

    def _load_patient(self, current_patient: str) -> None:
        self._patient.load(path=self._patients_save_path,
                           filename=current_patient)
        self._characteristics['age'] = self._patient._age
        self._characteristics['weight'] = self._patient._weight  # pylint: disable=no-member
        self._characteristics['height'] = self._patient._height  # pylint: disable=no-member
        self._characteristics['gender'] = self._patient._gender  # pylint: disable=no-member
        self._characteristics['race'] = self._patient._race  # pylint: disable=no-member
        self._characteristics['tobaco'] = self._patient._tobaco  # pylint: disable=no-member
        self._characteristics['amiodarone'] = self._patient._amiodarone  # pylint: disable=no-member
        self._characteristics['fluvastatin'] = self._patient._fluvastatin  # pylint: disable=no-member
        self._characteristics['CYP2C9'] = self._patient._CYP2C9
        self._characteristics['VKORC1'] = self._patient._VKORC1
        self._randomized = self._patient._randomized
        self._max_time = self._patient._max_time

    def _generate_ravvaz_patient(self) -> None:
        self._characteristics['age'] = min([max([np.random.normal(self._list_of_probabilities['age'][0],
                                                                  self._list_of_probabilities['age'][1]),
                                                 self._list_of_characteristics['age'][0]]),
                                            self._list_of_characteristics['age'][1]])  # truncated normal distribution with 3-sigma bound
        self._characteristics['weight'] = min([max([np.random.normal(self._list_of_probabilities['weight'][0],
                                                                     self._list_of_probabilities['weight'][1]),
                                                    self._list_of_characteristics['weight'][0]]),
                                               self._list_of_characteristics['weight'][1]])  # truncated normal distribution with 3-sigma bound
        self._characteristics['height'] = min([max([np.random.normal(self._list_of_probabilities['height'][0],
                                                                     self._list_of_probabilities['height'][1]),
                                                    self._list_of_characteristics['height'][0]]),
                                               self._list_of_characteristics['height'][1]])  # truncated normal distribution with 3-sigma bound

        self._characteristics['gender'] = np.random.choice(self._list_of_characteristics['gender'], 1,
                                                           p=self._list_of_probabilities['gender'])[0]
        self._characteristics['race'] = np.random.choice(self._list_of_characteristics['race'], 1,
                                                         p=self._list_of_probabilities['race'])[0]
        self._characteristics['tobaco'] = np.random.choice(self._list_of_characteristics['tobaco'], 1,
                                                           p=self._list_of_probabilities['tobaco'])[0]
        self._characteristics['amiodarone'] = np.random.choice(self._list_of_characteristics['amiodarone'], 1,
                                                               p=self._list_of_probabilities['amiodarone'])[0]
        self._characteristics['fluvastatin'] = np.random.choice(self._list_of_characteristics['fluvastatin'], 1,
                                                                p=self._list_of_probabilities['fluvastatin'])[0]

        self._characteristics['CYP2C9'] = np.random.choice(self._list_of_characteristics['CYP2C9'], 1,
                                                           p=self._list_of_probabilities['CYP2C9'])[0]
        self._characteristics['VKORC1'] = np.random.choice(self._list_of_characteristics['VKORC1'], 1,
                                                           p=self._list_of_probabilities['VKORC1'])[0]

    def _generate_random_patient(self) -> None:
        self._characteristics['age'] = np.random.uniform(self._list_of_characteristics['age'][0],
                                                         self._list_of_characteristics['age'][1])
        self._characteristics['weight'] = np.random.uniform(self._list_of_characteristics['weight'][0],
                                                            self._list_of_characteristics['weight'][1])
        self._characteristics['height'] = np.random.uniform(self._list_of_characteristics['height'][0],
                                                            self._list_of_characteristics['height'][1])
        self._characteristics['gender'] = random.choice(
            self._list_of_characteristics['gender'])
        self._characteristics['race'] = random.choice(
            self._list_of_characteristics['race'])
        self._characteristics['tobaco'] = random.choice(
            self._list_of_characteristics['tobaco'])
        self._characteristics['amiodarone'] = random.choice(
            self._list_of_characteristics['amiodarone'])
        self._characteristics['fluvastatin'] = random.choice(
            self._list_of_characteristics['fluvastatin'])
        self._characteristics['CYP2C9'] = random.choice(
            self._list_of_characteristics['CYP2C9'])
        self._characteristics['VKORC1'] = random.choice(
            self._list_of_characteristics['VKORC1'])

    def __repr__(self) -> str:
        try:
            return f"WarfarinModel: {[' '.join((str(k), ':', str(v))) for k, v in self._characteristics.items()]}, " \
                   f"INR: {self._INR_history}, intervals: {self._interval_history}, doses: {self._dose_history}"
        except:
            return 'WarfarinModel'
