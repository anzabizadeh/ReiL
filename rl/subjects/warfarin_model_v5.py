# -*- coding: utf-8 -*-
'''
warfarin_model class
==================

This `warfarin_model` class implements a two compartment PK/PD model for warfarin. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

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
        \n   initial_phase_duration: number of days initial phase (INR measured every day) lasts. If it is set to -1, it automatically
        \n       switches from initial to maintenance as soon as INR is in the range. (Default: -1)
        \n   maintenance_day_interval: how many days between two INR measurements in the maintenance phase (Default: 1)
        \n   max_day_1_dose: the maximum dose the experiment can start with. (Default: 15 mg/day)
        \n   max_initial_dose_change: maximum change of the dose in the initial phase. (Default: 15 mg/day)
        \n   max_maintenance_dose_change: maximum change of the dose in the maintenance phase. (Default: 15 mg/day)

        \n   max_dose: maximum possible dose (Default: 15 mg/day)
        \n   dose_steps: minimum possible change in dose (Default: 0.5)
        \n   dose_change_penalty_func: a function that computes penalty of changing dose from its previous value. The
        \n     argument for this function is the list of previous doses and should return a scalar. (Default: lambda x: int(x[-2] != x[-1]))
        \n   dose_change_penalty_coef: the magnigute of impact of dose change penalty.
        \n      (Total penalty = INR penalty + coef. * dose change penalty) (Default: 1)

        \n   dose_history: number of days of history for doses (Default: 5 days)
        \n   INR_history: number of days of history for INRs (Default: 5 days)

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
        ''' 
        
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
                            SS=0, max_time=24*90,
                            day=1, max_day=90, INR=[0], INR_current=0,
                            dosing_intervals=[None]*5, d_current=1, d_max=30,
                            current_dose=0, max_dose=15, dose_steps=0.5, therapeutic_range=(2, 3),
                            dose_history=5, INR_history=5, pill_per_day=1, randomized=True,
                            dose_list=[None]*5,
                            dose_change_penalty_coef=1,
                            dose_change_penalty_func=lambda x: int(
                                x[-2] != x[-1]),  # -0.2 * abs(x[-2]-x[-1]),
                            save_patients=False,
                            patients_save_path='./patients',
                            patients_save_prefix='warfv5',
                            patient_save_overwrite=False,
                            patient_use_existing=True,
                            patient_counter_start=0,
                            initial_phase_duration=-1,
                            phase='initial',
                            maintenance_day_interval=1,
                            max_day_1_dose=15,
                            max_initial_dose_change=15,
                            max_maintenance_dose_change=15
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
        super().__init__(**kwargs)

        if False:
            self._ex_protocol_options = {}
            self._ex_protocol_current = {}
            self._max_time = 0
            self._day = 1
            self._max_day = 0
            self._current_dose = 0
            self._max_dose = 0
            self._dose_change_penalty_func = lambda x: 0
            self._dose_change_penalty_coef = 0
            self._dose_steps = 0
            self._dose_history = 0
            self._INR_history = 0

            self._patient_selection = ''
            self._list_of_characteristics = {}
            self._list_of_probabilities = {}
            self._characteristics = {}
            self._INR = []

            self._dosing_intervals = 0
            self._d_current = 0
            self._therapeutic_range = ()
            self._randomized = True
            self._save_patients = False
            self._patients_save_path = './patients'
            self._patients_save_prefix = 'warfv5'
            self._patient_save_overwrite = False
            self._patient_use_existing = True
            self._patient_counter_start = 0

            self._initial_phase_duration = -1
            self._phase = 'initial'
            self._maintenance_day_interval = 1
            self._max_day_1_dose = 15
            self._max_initial_dose_change = 15
            self._max_maintenance_dose_change = 15

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

        self.reset()

        self._INR_mid = (self._therapeutic_range[1] + self._therapeutic_range[0]) / 2
        self._INR_range = self._therapeutic_range[1] - self._therapeutic_range[0]


        self._possible_actions = RLData([x*self._dose_steps
                                         for x in range(int(self._max_dose/self._dose_steps), -1, -1)],
                                        lower=0, upper=self._max_dose).as_rldata_array()
        self._day_1_possible_actions = RLData([x*self._dose_steps
                                                for x in range(int(self._max_day_1_dose/self._dose_steps), -1, -1)],
                                                lower=0, upper=self._max_dose).as_rldata_array()

    @property
    def state(self):
        if self._ex_protocol_current['state'] == 'extended':
            return self._state_extended()
        else:
            return self._state_normal()

    @property
    def is_terminated(self):
        return self._day >= self._max_day

    @property
    # only considers the dose
    def possible_actions(self):
        if self._day == 1:
            return self._day_1_possible_actions

        if self._phase == 'initial' and self._current_dose + self._max_initial_dose_change < self._max_dose:
            return self._possible_actions[ceil((self._max_dose - self._current_dose - self._max_initial_dose_change)/self._dose_steps):]

        if self._phase == 'maintenance' and self._current_dose + self._max_maintenance_dose_change < self._max_dose:
            return self._possible_actions[ceil((self._max_dose - self._current_dose - self._max_maintenance_dose_change)/self._dose_steps):]

        # if self._phase == 'initial' and self._max_initial_dose_change != self._max_dose:
        #     return RLData([x*self._dose_steps
        #                    for x in range(int(self._max_dose/self._dose_steps), -1, -1)
        #                    if abs(x*self._dose_steps - self._current_dose) <= self._max_initial_dose_change],
        #                    lower=0, upper=self._max_dose).as_rldata_array()

        # if self._phase == 'maintenance' and self._max_maintenance_dose_change != self._max_dose:
        #     return RLData([x*self._dose_steps
        #                    for x in range(int(self._max_dose/self._dose_steps), -1, -1)
        #                    if abs(x*self._dose_steps - self._current_dose) <= self._max_maintenance_dose_change],
        #                    lower=0, upper=self._max_dose).as_rldata_array()

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
        self._current_dose = action[0]
        self._dosing_intervals.append(self._d_current)
        self._dosing_intervals.popleft()
        self._dose_list.append(self._current_dose)
        self._dose_list.popleft()

        if (self._initial_phase_duration == -1 and self._therapeutic_range[0] <= self._INR[-1] <= self._therapeutic_range[1]) \
            or self._day == self._initial_phase_duration:
                self._phase = 'maintenance'
        if self._phase == 'initial':
            self._d_current = 1  # action.value[1]
            self._patient.dose = {self._day: self._current_dose}
        else:
            self._d_current = min(self._maintenance_day_interval, self._max_day-self._day)
            self._patient.dose = dict(tuple((i + self._day, self._current_dose) for i in range(self._d_current)))

        self._day += self._d_current

        self._INR.append(self._patient.INR(self._day)[-1])
        self._INR.popleft()

        try:
            # reward = 1 if self._therapeutic_range[0] <= self._INR[-1] <= self._therapeutic_range[1] else 0
            # TTR = sum((self._therapeutic_range[0] <= self._INR[-2] + (self._INR[-1]-self._INR[-2])/self._dosing_intervals*j <= self._therapeutic_range[1]
            #            for j in range(self._dosing_intervals)))
            # considering day=1 in INR_penalty calculation
            # INR_penalty = -sum(((2 / INR_range * (INR_mid - self._INR[-2] - (self._INR[-1]-self._INR[-2])/self._d_current*j)) ** 2
            #                     for j in range(0 if self._day==1 else 1, self._d_current + 1)))  # negative squared distance as reward (used *2/range to normalize)
            INR_penalty = -sum(((2 / self._INR_range * (self._INR_mid - self._INR[-2] - (self._INR[-1]-self._INR[-2])/self._d_current*j)) ** 2
                                for j in range(1, self._d_current + 1)))  # negative squared distance as reward (used *2/range to normalize)
            dose_change_penalty = - self._dose_change_penalty_func(self._dose_list)
            reward = INR_penalty + self._dose_change_penalty_coef * dose_change_penalty
        except TypeError:
            # TTR = 0  # assuming that for the first use of the pill, we don't have INR and TTR=0
            reward = 0

        # return TTR*self._d_current
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
        if self._save_patients and not self._patient_save_overwrite:
            try:
                self._load_patient(current_patient)
            except FileNotFoundError:
                pass
        self._patient.save(path=self._patients_save_path, filename=current_patient)
        self._filename_counter += 1

        self._day = 1
        self._current_dose = 0
        self._phase = 'initial'
        self._dose_list = deque([0.0]*self._dose_history)
        self._dosing_intervals = deque([1]*self._dose_history)

        self._INR = deque([0.0]*(self._INR_history + 1))
        self._INR[-1] = self._patient.INR([0])[-1]
        self._d_current = 1

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
                        'Doses': tuple(self._dose_list),
                        'INRs': tuple(self._INR)},
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
                       'Doses': tuple(self._dose_list),
                       'INRs': tuple(self._INR),
                       'Intervals': tuple(self._dosing_intervals)},
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
                   f"INR: {self._INR}, d_prev: {self._dosing_intervals}, d: {self._d_current}"
        except:
            return 'WarfarinModel'
