# -*- coding: utf-8 -*-
'''
warfarin_model class
==================

This `warfarin_model` class implements a two compartment PK/PD model for warfarin. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from copy import deepcopy
from numbers import Number
import os

import numpy as np
import pandas as pd

from random import choice, seed, shuffle
from math import ceil
from collections import deque

from ..rlbase import RLBase
from ..rldata import RLData
from ..subjects import Subject
from ..utils import Patient

class WarfarinModel_v5(Subject):
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

    def __init__(self, **kwargs):
        '''
        Create a warfarin model:
        \nArguments:
        \n   max_day: maximum number of days of simulation (Default: 90 days)
        \n   max_dose: maximum possible dose (Default: 15 mg/day)
        \n   dose_steps: minimum possible change in dose (Default: 0.5)
        \n   dose_change_penalty_func: a function that computes penalty of changing dose from its previous value. The
        \n     argument for this function is the list of previous doses and should return a scalar. (Default: lambda x: int(x[-2] != x[-1]))
        \n   dose_change_penalty_coef: the magnigute of impact of dose change penalty.
        \n      (Total penalty = INR penalty + coef. * dose change penalty) (Default: 1)

        \n   dose_history_length: number of days of history for doses (Default: 5 days)
        \n   INR_history_length: number of days of history for INRs (Default: 5 days)

        \n   therapeutic_range: acceptable range of INR values (Default: (2, 3))
        \n   patient_selection: how to generate patients. One of 'random', 'ravvaz' and 'fixed'. (Default: 'random')
        \n   characteristics: a dictionary describing the patient. (Default: {'age': 71, 'weight': 199.24, 'height': 66.78, 'gender': 'Male',
        \n     'race': 'White', 'tobaco': 'No', 'amiodarone': 'No', 'fluvastatin': 'No', 'CYP2C9': '*1/*1', 'VKORC1': 'A/A'})
        \n   randomized: whether to have random effect in the PK/PD model (Default: True)

        \n   save_patients: should the generated patients be saved? (Default: False)
        \n   patients_save_path: where to save patient files (Default: './patients')
        \n   patients_save_prefix: prefix to use when saving patients (Default: 'warfv5')
        \n   patient_save_overwrite: overwrite currently saved patients? (Default: False)
        \n   patient_use_existing: use currently saved patients if they exist? (Default: True)
        \n   patient_counter_start: the starting value for patient filename counter (Default: 0)

        \n   action_type: what type of actions are allowed: 'dose only', 'interval only', 'both' (Default: 'dose only')
        ''' 


        self._INR_penalty_coef = kwargs.get('INR_penalty_coef', 1)
        self._lookahead_penalty_coef = kwargs.get('lookahead_penalty_coef', 0)


        self.set_defaults(ex_protocol_options={'state': ['standard', 'extended'], 'possible_actions': ['standard'], 'take_effect': ['standard', 'no_reward']},
                            ex_protocol_current={'state': 'standard', 'possible_actions': 'standard', 'take_effect': 'standard'},
                            stats_list={'TTR', 'dose_change', 'delta_dose'},
                            patient_selection='random',
                            list_of_characteristics={'age': (18, 100),  # similar to the range of 10k sample from Ravvaz
                                                    'weight': (70, 500),  # (lb) similar to the range of 10k sample from Ravvaz
                                                    'height': (45, 85),  # (in) similar to the range of 10k sample from Ravvaz
                                                    'gender': ('Female', 'Male'),
                                                    'race': ('White', 'Black', 'Asian', 'American Indian', 'Pacific Islander'),
                                                    'tobaco': ('No', 'Yes'),
                                                    'amiodarone': ('No', 'Yes'),
                                                    'fluvastatin': ('No', 'Yes'),
                                                    'CYP2C9': ('*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3'),
                                                    'VKORC1': ('G/G', 'G/A', 'A/A')},
                            list_of_probabilities={'age': (67.3, 14.43),  # lb  - Aurora population
                                                'weight': (199.24, 54.71),  # in - Aurora population
                                                'height': (66.78, 4.31),  # Aurora population
                                                'gender': (0.5314, 0.4686),  # Aurora population
                                                'race': (0.9522, 0.0419, 0.0040, 0.0018, 1e-4),  # Aurora Avatar Population
                                                'tobaco': (0.9067, 0.0933),  # Aurora Avatar Population
                                                'amiodarone': (0.8849, 0.1151),  # Aurora Avatar Population
                                                'fluvastatin': (0.9998, 0.0002),  # Aurora Avatar Population
                                                'CYP2C9': (0.6739, 0.1486, 0.0925, 0.0651, 0.0197, 2e-4),  # Aurora Avatar Population
                                                'VKORC1': (0.3837, 0.4418, 0.1745)},  # Aurora Avatar Population
                            characteristics={'age': 71, 'weight': 199.24, 'height': 66.78, 'gender': 'Male',
                                            'race': 'White', 'tobaco': 'No', 'amiodarone': 'No', 'fluvastatin': 'No',
                                            'CYP2C9': '*1/*1', 'VKORC1': 'A/A'},
                            randomized=True,
                            action_type='dose only',

                            day=1, max_day=90, max_time=24*90,
                            dose_history_length=1, dose_history=[0], max_dose=15.0, dose_steps=0.5,
                            INR_history_length=1, INR_history=[0, 0], intervals_history=[1],
                            therapeutic_range=(2, 3),
                            INR_penalty_coef=1,
                            dose_change_penalty_coef=1,
                            dose_change_penalty_func=lambda x: int(x[-2] != x[-1]),  # -0.2 * abs(x[-2]-x[-1]),
                            save_patients=False,
                            patients_save_path='./patients',
                            patients_save_prefix='warfv5',
                            patient_save_overwrite=False,
                            patient_use_existing=True,
                            patient_counter_start=0,
                            lookahead_duration=0, 
                            interval=[1], interval_max_dose=[15], interval_index=0, max_interval = 28 # everyday dosing
                        )

        # this makes sure that if some elements of dictionaries
        # (e.g. characteristics, list_of_...) changes by the user,
        # other elements of the dictionary remain intact.
        for key, value in kwargs.items():
            if isinstance(value, dict):
                try:
                    temp = self._defaults[key]
                    temp.update(kwargs[key])
                    kwargs[key] = temp
                except KeyError:
                    pass

        self.set_params(**kwargs)
        self._max_day += self._lookahead_duration
        super().__init__(**kwargs)

        if False:
            self._ex_protocol_options = {}
            self._ex_protocol_current = {}
            self._max_time = 0
            self._day = 1
            self._max_day = 0
            self._max_dose = 0
            self._dose_change_penalty_func = lambda x: 0
            self._dose_change_penalty_coef = 0
            self._dose_steps = 0
            self._dose_history_length = 0
            self._INR_history_length = 0

            self._patient_selection = ''
            self._list_of_characteristics = {}
            self._list_of_probabilities = {}
            self._characteristics = {}
            self._INR_history = []

            self._intervals_history = []
            self._therapeutic_range = ()
            self._randomized = True
            self._save_patients = False
            self._patients_save_path = './patients'
            self._patients_save_prefix = 'warfv5'
            self._patient_save_overwrite = False
            self._patient_use_existing = True
            self._patient_counter_start = 0
            self._interval = [1]
            self._interval_index = 0
            self._interval_max_dose = [15]
            self._max_interval = 28
            self._action_type = ''

            self._lookahead_duration = 0

        if self._patient_selection in ('ravvaz', 'ravvaz 2017', 'ravvaz_2017', 'ravvaz2017'):
            if self._list_of_characteristics['CYP2C9'] != ('*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3') or \
                    self._list_of_characteristics['VKORC1'] != ('G/G', 'G/A', 'A/A'):
                raise ValueError(
                    'For Ravvaz patient generation, CYP2C9 and VKORC1 should not be changed!')

        self._max_time = (self._max_day + 1)*24  # until the end of max_day

        self._filename_counter = self._patient_counter_start
        if self._save_patients:
            if not self._patient_save_overwrite and not self._patient_use_existing:
                while os.path.exists(os.path.join(self._patients_save_path,
                                                  ''.join((self._patients_save_prefix, f'{self._filename_counter:06}')))):
                    self._filename_counter += 1

        if isinstance(self._interval_max_dose, Number):
            self._interval_max_dose = [self._interval_max_dose] * len(self._interval)
        elif len(self._interval_max_dose) == 1:
            self._interval_max_dose = self._interval_max_dose * len(self._interval)
        elif len(self._interval_max_dose) != len(self._interval):
            self._logger.warning('interval_max_dose does not match "interval" in length. max_dose will be used for intervals without interval_max_dose values.')

        self.reset()

        self._INR_mid = (self._therapeutic_range[1] + self._therapeutic_range[0]) / 2
        self._INR_range = self._therapeutic_range[1] - self._therapeutic_range[0]

        if self._action_type.lower() in ('dose', 'dose only', 'dose_only', 'only dose', 'only_dose'):
            self._possible_actions = RLData([x*self._dose_steps
                                            for x in range(int(self._max_dose/self._dose_steps), -1, -1)],
                                            lower=0, upper=self._max_dose).as_rldata_array()
        elif self._action_type.lower() in ('interval', 'interval only', 'interval_only', 'only interval', 'only_interval'):
            self._possible_actions = RLData(list(range(1, self._max_interval + 1)),
                                            lower=1, upper=self._max_interval).as_rldata_array()
        else:
             self._possible_actions = RLData([(x*self._dose_steps, i)
                                            for x in range(int(self._max_dose/self._dose_steps), -1, -1)
                                            for i in range(1, self._max_interval + 1)],
                                            lower=(0, 1), upper=(self._max_dose, self._max_interval)).as_rldata_array()

    @property
    def state(self):
        if self._ex_protocol_current['state'] == 'extended':
            return self._state_extended()
        else:
            return self._state_normal()

    @property
    def complete_state(self):
        return self._state_extended()

    @property
    def is_terminated(self):
        return self._day >= self._max_day - self._lookahead_duration

    @property
    # only considers the dose
    def possible_actions(self):
        try:
            if self._interval_max_dose[self._interval_index] < self._max_dose:
                return RLData([x*self._dose_steps
                                for x in range(int(self._interval_max_dose[self._interval_index]/self._dose_steps), -1, -1)],
                                lower=0, upper=self._max_dose).as_rldata_array()
        except IndexError:
            pass  # When self._interval_index >= len(self._interval_max_dose), use self._max_dose instead of self._interval_max_dose[self._interval_index]

        return self._possible_actions

    def register(self, agent_name):
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

    def take_effect(self, action, _id=None):
        current_dose = action[0]
        try:  # use the provided action, otherwise the interval
            current_interval = min(action[1], self._max_day - self._day)
        except IndexError:
            current_interval = min(self._get_next_interval(), self._max_day - self._day)

        self._patient.dose = dict(tuple((i + self._day, current_dose) for i in range(current_interval)))

        self._dose_history.append(current_dose)
        self._dose_history.popleft()
        self._intervals_history.append(current_interval)
        self._intervals_history.popleft()

        day_temp = self._day
        self._day += current_interval

        self._INR_history.append(self._patient.INR(self._day)[-1])
        self._INR_history.popleft()

        try:
            if self.exchange_protocol['take_effect'] == 'standard':
                if self._lookahead_duration > 0:
                    temp_patient = deepcopy(self._patient)
                    temp_patient.dose = dict(tuple((i + day_temp, current_dose) for i in range(1, self._lookahead_duration + 1)))
                    lookahead_penalty = -sum(((2 / self._INR_range * (self._INR_mid - INRi)) ** 2
                                            for INRi in temp_patient.INR(list(range(day_temp + 1, day_temp + self._lookahead_duration + 1)))))
                else:
                    lookahead_penalty = 0

                INR_penalty = -sum(((2 / self._INR_range * (self._INR_mid - self._INR_history[-2] - (self._INR_history[-1]-self._INR_history[-2])/current_interval*j)) ** 2
                                    for j in range(1, current_interval + 1)))  # negative squared distance as reward (used *2/range to normalize)
                dose_change_penalty = - self._dose_change_penalty_func(self._dose_history)
                reward = self._INR_penalty_coef * INR_penalty \
                        + self._dose_change_penalty_coef * dose_change_penalty \
                        + self._lookahead_penalty_coef * lookahead_penalty
            elif self.exchange_protocol['take_effect'] == 'no_reward':
                reward = 0

        except TypeError:
            reward = 0

        return RLData(reward, normalizer=lambda x: x)

    def stats(self, stats_list):
        if isinstance(stats_list, str):
            stats_list = [stats_list]
        results = {}
        for s in stats_list:
            if s == 'TTR':
                INRs = self._patient.INR(list(range(self._day+1)))
                temp = sum((1 if 2.0<=INRi<=3.0 else 0 for INRi in INRs)) / len(INRs)
            elif s == 'dose_change':
                temp = np.sum(np.abs(np.diff(self._patient.dose))>0)
            elif s == 'delta_dose':
                temp = np.sum(np.abs(np.diff(self._patient.dose)))
            else:
                print(f'WARNING! {s} is not one of the available stats!')
                continue

            results[s] = temp

        results['ID'] = RLData({'age': self._characteristics['age'],
                                'CYP2C9': self._characteristics['CYP2C9'],
                                'VKORC1': self._characteristics['VKORC1']},
                                lower={'age': self._list_of_characteristics['age'][0]},
                                upper={'age': self._list_of_characteristics['age'][-1]},
                                categories={'CYP2C9': self._list_of_characteristics['CYP2C9'],
                                            'VKORC1': self._list_of_characteristics['VKORC1']}, lazy_evaluation=True)

        return results

    def reset(self):
        if self._patient_selection == 'random':
            self._generate_random_patient()
        elif self._patient_selection.lower() in ['ravvaz', 'ravvaz 2017', 'ravvaz_2017', 'ravvaz2017']:
            self._generate_ravvaz_patient()

        self._patient = Patient(age=self._characteristics['age'],
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

        current_patient = ''.join((self._patients_save_prefix, f'{self._filename_counter:06}'))
        if self._save_patients:
            if self._patient_save_overwrite:
                self._patient.save(path=self._patients_save_path, filename=current_patient)
            else:
                try:
                    self._load_patient(current_patient)
                except FileNotFoundError:
                    self._patient.save(path=self._patients_save_path, filename=current_patient)

        self._filename_counter += 1

        self._day = 1
        self._dose_history = deque([0.0]*self._dose_history_length)
        self._intervals_history = deque([1]*self._dose_history_length)

        self._INR_history = deque([0.0]*(self._INR_history_length + 1))  # The latest INR is also stored in self._INR_history
        self._INR_history[-1] = self._patient.INR([0])[-1]
        self._interval_index = 0

    def _state_extended(self):
        return RLData({'age': self._characteristics['age'],
                        'weight': self._characteristics['weight'],
                        'height': self._characteristics['height'],
                        'gender': self._characteristics['gender'],
                        'race': self._characteristics['race'],
                        'tobaco': self._characteristics['tobaco'],
                        'amiodarone': self._characteristics['amiodarone'],
                        'fluvastatin': self._characteristics['fluvastatin'],
                        'CYP2C9': self._characteristics['CYP2C9'],
                        'VKORC1': self._characteristics['VKORC1'],
                        'day': self._day,
                        'Doses': tuple(self._dose_history),
                        'INRs': tuple(self._INR_history)},
                        lower={'age': self._list_of_characteristics['age'][0],
                                'weight': self._list_of_characteristics['weight'][0],
                                'height': self._list_of_characteristics['height'][0],
                                'day': 0,
                                'Doses': 0.0,
                                'INRs': 0.0},
                        upper={'age': self._list_of_characteristics['age'][-1],
                                'weight': self._list_of_characteristics['weight'][-1],
                                'height': self._list_of_characteristics['height'][-1],
                                'day': self._max_day,
                                'Doses': self._max_dose,
                                'INRs': 15.0},
                        categories={'CYP2C9': self._list_of_characteristics['CYP2C9'],
                                    'VKORC1': self._list_of_characteristics['VKORC1'],
                                    'gender': self._list_of_characteristics['gender'],
                                    'race': self._list_of_characteristics['race'],
                                    'tobaco': self._list_of_characteristics['tobaco'],
                                    'amiodarone': self._list_of_characteristics['amiodarone'],
                                    'fluvastatin': self._list_of_characteristics['fluvastatin']})

    def _state_normal(self):
        return RLData({'age': self._characteristics['age'],
                       'CYP2C9': self._characteristics['CYP2C9'],
                       'VKORC1': self._characteristics['VKORC1'],
                       'Doses': tuple(self._dose_history),
                       'INRs': tuple(self._INR_history),
                       'Intervals': tuple(self._intervals_history)},
                      lower={'age': self._list_of_characteristics['age'][0],
                             'Doses': 0.0,
                             'INRs': 0.0,
                             'Intervals': 1},
                      upper={'age': self._list_of_characteristics['age'][-1],
                             'Doses': self._max_dose,
                             'INRs': 15.0,
                             'Intervals': self._max_day},
                      categories={'CYP2C9': self._list_of_characteristics['CYP2C9'],
                                  'VKORC1': self._list_of_characteristics['VKORC1']})

    def _get_next_interval(self):
        if self._interval[self._interval_index] < 0:
            if not (self._therapeutic_range[0] <= self._INR_history[-1] <= self._therapeutic_range[1]):
                self._interval_index -= 1

        try:
            interval = self._interval[self._interval_index + 1]
            self._interval_index += 1
        except IndexError:
            interval = self._interval[self._interval_index]

        return abs(interval)

    def _load_patient(self, current_patient):
        self._patient.load(path=self._patients_save_path, filename=current_patient)
        self._characteristics['age'] = self._patient._age
        self._characteristics['weight'] = self._patient._weight
        self._characteristics['height'] = self._patient._height
        self._characteristics['gender'] = self._patient._gender
        self._characteristics['race'] = self._patient._race
        self._characteristics['tobaco'] = self._patient._tobaco
        self._characteristics['amiodarone'] = self._patient._amiodarone
        self._characteristics['fluvastatin'] = self._patient._fluvastatin
        self._characteristics['CYP2C9'] = self._patient._CYP2C9
        self._characteristics['VKORC1'] = self._patient._VKORC1
        self._randomized = self._patient._randomized
        self._max_time = self._patient._max_time

    def _generate_ravvaz_patient(self):
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

    def _generate_random_patient(self):
        self._characteristics['age'] = np.random.uniform(self._list_of_characteristics['age'][0],
                                                            self._list_of_characteristics['age'][1])
        self._characteristics['weight'] = np.random.uniform(self._list_of_characteristics['weight'][0],
                                                            self._list_of_characteristics['weight'][1])
        self._characteristics['height'] = np.random.uniform(self._list_of_characteristics['height'][0],
                                                            self._list_of_characteristics['height'][1])
        self._characteristics['gender'] = choice(
            self._list_of_characteristics['gender'])
        self._characteristics['race'] = choice(
            self._list_of_characteristics['race'])
        self._characteristics['tobaco'] = choice(
            self._list_of_characteristics['tobaco'])
        self._characteristics['amiodarone'] = choice(
            self._list_of_characteristics['amiodarone'])
        self._characteristics['fluvastatin'] = choice(
            self._list_of_characteristics['fluvastatin'])
        self._characteristics['CYP2C9'] = choice(
            self._list_of_characteristics['CYP2C9'])
        self._characteristics['VKORC1'] = choice(
            self._list_of_characteristics['VKORC1'])

    def __repr__(self):
        try:
            return f"WarfarinModel: {[' '.join((str(k), ':', str(v))) for k, v in self._characteristics.items()]}, " \
                   f"INR: {self._INR_history}, intervals: {self._intervals_history}, doses: {self._dose_history}"
        except:
            return 'WarfarinModel'
