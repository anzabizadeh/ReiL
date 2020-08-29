# -*- coding: utf-8 -*-
'''
warfarin_model class
==================

This `warfarin_model` class implements a two compartment PK/PD model for warfarin. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import os
from collections import deque
from math import exp, log, sqrt
from random import choice, gauss, sample, seed, shuffle

import numpy as np
import pandas as pd
from dill import HIGHEST_PROTOCOL, dump, load
from scipy.stats import lognorm

from ..rlbase import RLBase
from ..rldata import RLData
from ..subjects import Subject

class WarfarinModel_v4(Subject):
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
        self.set_defaults(patient_selection='random',
                             list_of_characteristics={'age': (71, 86),
                                                      # lb
                                                      'weight': (35, 370),
                                                      'height': (53, 80),  # in
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
                             d_previous=0, d_current=1, d_max=30,
                             current_dose=0, max_dose=15, dose_steps=0.5, TTR_range=(2, 3),
                             dose_history=5, INR_history=5, pill_per_day=1, randomized=True,
                             dose_list=[None]*5,
                             dose_change_penalty_coef=1,
                             dose_change_penalty_func=lambda x: int(
                                 x[-2] != x[-1]),  # -0.2 * abs(x[-2]-x[-1]),
                             save_patients=False,
                             patients_save_path='./',
                             patients_save_prefix='warfv4',
                             patient_save_overwrite=False,
                             patient_use_existing=True,
                             extended_state=False
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
            self._model_filename = ''
            self._Cs_super = 0
            self._SS = 0
            self._max_time = 0
            self._rseed = 0
            self._day = 1
            self._max_day = 0
            self._current_dose = 0
            self._max_dose = 0
            self._dose_change_penalty_func = lambda x: 0
            self._dose_change_penalty_coef = 0
            self._dose_steps = 0
            self._dose_history = 0
            self._INR_history = 0
            self._pill_per_day = 0

            self._patient_selection = ''
            self._list_of_characteristics = {}
            self._list_of_probabilities = {}
            self._characteristics = {}
            self._INR = []

            self._d_previous = 0
            self._d_current = 0
            self._d_max = 0
            self._TTR_range = ()
            self._randomized = True
            self._extended_state = False
            self._save_patients = False
            self._patients_save_path = './'
            self._patients_save_prefix = 'warfv4'
            self._patient_save_overwrite = False
            self._patient_use_existing = True
            self._extended_state = False

        if self._patient_selection in ('ravvaz', 'ravvaz 2017', 'ravvaz_2017', 'ravvaz2017'):
            if self._list_of_characteristics['CYP2C9'] != ('*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3') or \
                    self._list_of_characteristics['VKORC1'] != ('G/G', 'G/A', 'A/A'):
                raise ValueError(
                    'For Ravvaz patient generation, CYP2C9 and VKORC1 should not be changed!')

        self._max_time = (self._max_day + 1)*24  # until the end of max_day

        self._filename_counter = 0
        if self._save_patients:
            if not self._patient_save_overwrite and not self._patient_use_existing:
                while os.path.exists(os.path.join(self._patients_save_path,
                                                  ''.join((self._patients_save_prefix, f'{self._filename_counter:06}')))):
                    self._filename_counter += 1

        self.reset()

        self._possible_actions = RLData([x*self._dose_steps
                                         for x in range(int(self._max_dose/self._dose_steps), -1, -1)],
                                        lower=0, upper=self._max_dose).as_rldata_array()

    @property
    def state(self):
        if self._extended_state:
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

        return RLData({'age': self._characteristics['age'],
                       'CYP2C9': self._characteristics['CYP2C9'],
                       'VKORC1': self._characteristics['VKORC1'],
                       'Doses': tuple(self._dose_list),
                       'INRs': tuple(self._INR)},
                      lower={'age': self._list_of_characteristics['age'][0],
                             'Doses': 0.0,
                             'INRs': 0.0},
                      upper={'age': self._list_of_characteristics['age'][-1],
                             'Doses': self._max_dose,
                             'INRs': 15.0},
                      categories={'CYP2C9': self._list_of_characteristics['CYP2C9'],
                                  'VKORC1': self._list_of_characteristics['VKORC1']})

    @property
    def is_terminated(self):
        return self._day > self._max_day

    @property
    # only considers the dose
    def possible_actions(self):
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
        self._current_dose = action.value
        self._d_previous = self._d_current
        self._dose_list.append(self._current_dose)
        self._dose_list.popleft()

        self._patient.dose = {self._day: self._current_dose}
        self._d_current = 1  # action.value[1]
        self._day += self._d_current

        self._INR.append(self._patient.INR(self._day)[-1])
        self._INR.popleft()

        try:
            # reward = 1 if self._TTR_range[0] <= self._INR[-1] <= self._TTR_range[1] else 0
            # TTR = sum((self._TTR_range[0] <= self._INR[-2] + (self._INR[-1]-self._INR[-2])/self._d_previous*j <= self._TTR_range[1]
            #            for j in range(self._d_previous)))
            INR_mid = (self._TTR_range[1] + self._TTR_range[0]) / 2
            INR_range = self._TTR_range[1] - self._TTR_range[0]
            INR_penalty = -sum(((2 * (INR_mid - self._INR[-2] + (self._INR[-1]-self._INR[-2])/self._d_previous*j)) ** 2
                                for j in range(self._d_previous))) / INR_range  # negative squared distance as reward (used *2/range to normalize)
            dose_change_penalty = - \
                self._dose_change_penalty_func(self._dose_list)
            reward = INR_penalty + self._dose_change_penalty_coef * dose_change_penalty
        except TypeError:
            # TTR = 0  # assuming that for the first use of the pill, we don't have INR and TTR=0
            reward = 0

        # return TTR*self._d_current
        return reward

    def reset(self):
        if self._patient_selection == 'random':
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

        elif self._patient_selection.lower() in ['ravvaz', 'ravvaz 2017', 'ravvaz_2017', 'ravvaz2017']:
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

        current_patient = os.path.join(self._patients_save_path,
                                        ''.join((self._patients_save_prefix, f'{self._filename_counter:06}')))
        if self._save_patients and not self._patient_save_overwrite:
            try:
                # self._patient = Patient()
                self._patient.load(filename=current_patient)

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

                done_resetting = True
            except FileNotFoundError:
                pass

        self._patient.save(filename=current_patient)

        self._filename_counter += 1

        self._day = 1
        self._current_dose = 0
        self._dose_list = deque([0.0]*self._dose_history)

        self._INR = deque([0.0]*(self._INR_history + 1))
        self._INR[-1] = self._patient.INR([0])[-1]

        self._d_previous = 0
        self._d_current = 1

    def __repr__(self):
        try:
            return f"WarfarinModel: {[' '.join((str(k), ':', str(v))) for k, v in self._characteristics.items()]}, " \
                f"INR: {self._INR}, d_prev: {self._d_previous}, d: {self._d_current}"
        except:
            return 'WarfarinModel'


def rlnormRestricted(meanVal, stdev):
    # capture 50% of the data.  This restricts the log values to a "reasonable" range
    quartileRange = (0.25, 0.75)
    lnorm = lognorm(stdev, scale=exp(meanVal))
    qValues = lnorm.ppf(quartileRange)
    values = list(v for v in lnorm.rvs(size=1000)
                  if (v > qValues[0]) & (v < qValues[1]))
    return sample(values, 1)[0]


class Patient(RLBase):
    '''
    Two compartment PK/PD model for wafarin.

    Attributes
    ----------
        dose: a dictionary containing with day as key and dose as value.

    Methods
    -------
        INR: returns a list of INR values corresponding to the given list of days.
    '''

    def __init__(self, age=50, CYP2C9='*1/*1', VKORC1='G/A', randomized=True, max_time=24,
                 dose_interval=24, dose={}, lazy=False, **kwargs):
        self.set_defaults(**kwargs)
        self.set_params(**kwargs)
        super().__init__(**kwargs)

        self._age = age
        self._CYP2C9 = CYP2C9
        self._VKORC1 = VKORC1
        self._randomized = randomized
        self._MTT_1 = kwargs.get('MTT_1', rlnormRestricted(
            log(11.6), sqrt(0.141)) if randomized else 11.6)
        self._MTT_2 = kwargs.get('MTT_2', rlnormRestricted(
            log(120), sqrt(1.02)) if randomized else 120)
        # EC_50 in mg/L
        self._EC_50 = kwargs.get('EC_50', None)
        if self._EC_50 is None:
            if VKORC1 == "G/G":  # Order of genotypes changed
                self._EC_50 = rlnormRestricted(
                    log(4.61), sqrt(0.409)) if randomized else 4.61
            elif VKORC1 in ["G/A", "A/G"]:
                self._EC_50 = rlnormRestricted(
                    log(3.02), sqrt(0.409)) if randomized else 3.02
            elif VKORC1 == "A/A":
                self._EC_50 = rlnormRestricted(
                    log(2.20), sqrt(0.409)) if randomized else 2.20
            else:
                raise ValueError('The VKORC1 genotype is not supported!')

        self._cyp_1_1 = kwargs.get('cyp_1_1', rlnormRestricted(
            log(0.314), sqrt(0.31)) if randomized else 0.314)
        self._V1 = kwargs.get('V1', rlnormRestricted(
            log(13.8), sqrt(0.262)) if randomized else 13.8)
        self._V2 = kwargs.get('V2', rlnormRestricted(
            log(6.59), sqrt(0.991)) if randomized else 6.59)
        self._Q = 0.131    # (L/h)
        self._lambda = 3.61

        self._gamma = 0.424  # no units

        # bioavilability fraction 0-1 (from: "Applied Pharmacokinetics & Pharmacodynamics 4th edition, p.717", some other references)
        self._F = 0.9

        self._ka = 2  # absorption rate (1/hr)

        self._ktr1 = 6/self._MTT_1					# 1/hours; changed from 1/MTT_1
        self._ktr2 = 1/self._MTT_2					# 1/hours
        self._E_MAX = 1					        	# no units

        self._CL_s = 1
        if self._age > 71:
            self._CL_s = 1 - 0.0091 * (self._age - 71)

        if self._CYP2C9 == "*1/*1":
            self._CL_s = self._CL_s * self._cyp_1_1
        elif self._CYP2C9 == "*1/*2":
            self._CL_s = self._CL_s * self._cyp_1_1 * (1 - 0.315)
        elif self._CYP2C9 == "*1/*3":
            self._CL_s = self._CL_s * self._cyp_1_1 * (1 - 0.453)
        elif self._CYP2C9 == "*2/*2":
            self._CL_s = self._CL_s * self._cyp_1_1 * (1 - 0.722)
        elif self._CYP2C9 == "*2/*3":
            self._CL_s = self._CL_s * self._cyp_1_1 * (1 - 0.69)
        elif self._CYP2C9 == "*3/*3":
            self._CL_s = self._CL_s * self._cyp_1_1 * (1 - 0.852)
        else:
            raise ValueError('The CYP2C9 genotype not recognized fool!')

        self._max_time = max_time  # The last hour of experiment
        self._dose_interval = dose_interval
        self._lazy = lazy
        self._last_computed_day = 0

        # prepend time 0 to the list of times for deSolve initial conditions (remove when returning list of times)
        times = list(range(self._max_time+1))
        # times also equals the time-step for deSolve

        k12 = self._Q / self._V1
        k21 = self._Q / self._V2
        k10 = self._CL_s / self._V1
        b = k10 + k21 + k12
        c = k10 * k21
        alpha = (b + sqrt(b ** 2 - 4*c)) / 2
        beta = (b - sqrt(b ** 2 - 4*c)) / 2

        # 2-compartment model
        part_1 = np.array(list(((k21 - alpha) / ((self._ka - alpha)*(beta - alpha)))
                               * temp for temp in (exp(-alpha * t) for t in times)))
        part_2 = np.array(list(((k21 - beta) / ((self._ka - beta)*(alpha - beta)))
                               * temp for temp in (exp(-beta * t) for t in times)))
        part_3 = np.array(list(((k21 - self._ka) / ((self._ka - alpha)*(self._ka - beta)))
                               * temp for temp in (exp(-self._ka * t) for t in times)))
        # self._multiplication_term = part_1 + part_2 + part_3

        self._multiplication_term = ((self._ka * self._F / 2) /
                                     self._V1) * (part_1 + part_2 + part_3)

        if self._lazy:
            self._data = pd.DataFrame(columns=['dose']+times)
            self.dose = {0: 0.0}
        else:
            self._data = pd.DataFrame(columns=['dose'])
            self._total_Cs = np.zeros(self._max_time + 1)

    @property
    def dose(self):
        return self._data.dose

    @dose.setter
    def dose(self, dose):
        if self._lazy:
            for d, v in dose.items():
                if dose != 0.0:
                    self._data.loc[d] = np.insert(
                        self._Cs(dose=v, t0=d*self._dose_interval), 0, v)
        else:
            for d, v in dose.items():
                if dose != 0.0:
                    self._data.loc[d] = v
                    self._total_Cs = np.add(self._total_Cs, self._Cs(
                        dose=v, t0=d*self._dose_interval))

    def _Cs(self, dose, t0):
        # C_s_pred = dose * self._multiplication_term

        # if t0 == 0:  # non steady-state
        #     C_s_error = np.array([exp(gauss(0, 0.09)) for _ in range(len(C_s_pred))]) if self._randomized \
        #         else np.ones(len(C_s_pred))  # Sadjad
        # else:  # steady-state
        #     C_s_error = np.array([exp(gauss(0, 0.30)) for _ in range(len(C_s_pred))]) if self._randomized \
        #         else np.ones(len(C_s_pred))

        # C_s = np.multiply(C_s_pred, C_s_error).clip(min=0)

        # return np.pad(C_s, (t0, 0), 'constant', constant_values=(0,))[:self._max_time+1]

        C_s_pred = dose * self._multiplication_term

        if t0 == 0:  # non steady-state
            C_s_error = np.exp(np.random.normal(0, 0.09, len(C_s_pred))) if self._randomized \
                else np.ones(len(C_s_pred))  # Sadjad
        else:  # steady-state
            C_s_error = np.exp(np.random.normal(0, 0.30, len(C_s_pred))) if self._randomized \
                else np.ones(len(C_s_pred))

        C_s = np.multiply(C_s_pred, C_s_error).clip(min=0)

        return np.pad(C_s, (t0, 0), 'constant', constant_values=(0,))[:self._max_time+1]

    def INR(self, days):
        if isinstance(days, int):
            days = [days]

        if self._lazy:
            Cs_gamma = np.power(np.sum(self._data[list(
                range(int(max(days)*self._dose_interval)+1))], axis=0), self._gamma)
        else:
            Cs_gamma = np.power(self._total_Cs[0:int(
                max(days)*self._dose_interval)+1], self._gamma)

        start_days = sorted(
            [0 if days[0] < self._last_computed_day else self._last_computed_day] + days[:-1])
        end_days = sorted(days)

        if start_days[0] == 0:
            self._A = [1]*7
            self._dA = [0]*7

        INR_max = 20
        baseINR = 1
        INR = []
        for d1, d2 in zip(start_days, end_days):
            for i in range(int(d1*self._dose_interval), int(d2*self._dose_interval)):
                self._dA[0] = self._ktr1 * (1 - self._E_MAX * Cs_gamma[i] /
                                            (self._EC_50 ** self._gamma + Cs_gamma[i])) - self._ktr1*self._A[0]
                for j in range(1, 6):
                    self._dA[j] = self._ktr1 * (self._A[j-1] - self._A[j])

                self._dA[6] = self._ktr2 * (1 - self._E_MAX * Cs_gamma[i] /
                                            (self._EC_50 ** self._gamma + Cs_gamma[i])) - self._ktr2*self._A[6]
                for j in range(7):
                    self._A[j] += self._dA[j]

            e_INR = gauss(0, 0.0325) if self._randomized else 0
            INR.append(
                (baseINR + (INR_max*(1-self._A[5]*self._A[6]) ** self._lambda)) * exp(e_INR))

        self._last_computed_day = end_days[-1]

        return INR


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time

    max_day = 100
    p_count = range(20)
    p = [Patient(randomized=False, max_time=24*max_day + 1) for _ in p_count]
    t = time()
    for j in p_count:
        p[j].dose = {i: 15 for i in range(max_day)}
        # plt.plot(p.INR(list(i/24 for i in range(1, max_day*24 + 1))))
        plt.plot(list(range(24, (max_day+1)*24, 240)),
                 p[j].INR(list(i for i in range(1, max_day+1, 10))), 'x')
    plt.show()

    # p = [Patient(randomized=True, max_time=24*max_day + 1) for _ in p_count]
    # t = time()
    # for j in p_count:
    #     p[j].dose = {i: 15 for i in range(max_day)}
    #     p[j].INR(list(i for i in range(1, max_day+1, 10)))
    #     # plt.plot(p.INR(list(i/24 for i in range(1, max_day*24 + 1))))
    # #     plt.plot(list(range(24, (max_day+1)*24, 240)),
    # #              p[j].INR(list(i for i in range(1, max_day+1, 10))), 'x')
    # # plt.show()
    # print(time() - t)

    # p = [Patient(randomized=True, max_time=24*max_day + 1) for _ in p_count]
    # t = time()
    # for j in p_count:
    #     p[j].dose = {i: 15 for i in range(max_day)}
    #     for i in range(max_day, 1, -10):
    #         p[j].INR(i)

    # print(time() - t)
