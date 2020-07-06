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
from collections import deque

from ..rlbase import RLBase
from ..rldata import RLData
from ..subjects import WarfarinModel_v5
from ..utils import Patient

class WarfarinModelFixedInterval(WarfarinModel_v5):
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
        \n   switches from initial to maintenance as soon as INR is in the range. (Default: -1)
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
        \n   interval: a list of days that INR measurement and dose change happens
        \n   interval_index: a pointer to the current index of interval
        ''' 

        interval=[3]*2 + [7]*11 + [6, 1]  # had to use 6-days in the final round to have one last state value in stats computation
        self.set_defaults(interval=interval, interval_index=0, d_current=interval[0])
        self.set_params(**kwargs)
        super().__init__(**kwargs)

        if False:
            self._interval = []

    def take_effect(self, action, _id=None):
        self._current_dose = action[0]
        self._d_current = min(self._interval[self._interval_index], self._max_day - self._day)
        self._interval_index += 1
        self._patient.dose = dict(tuple((i + self._day, self._current_dose) for i in range(self._d_current)))

        self._dose_list.append(self._current_dose)
        self._dose_list.popleft()
        self._dosing_intervals.append(self._d_current)
        self._dosing_intervals.popleft()

        self._day += self._d_current

        self._INR.append(self._patient.INR(self._day)[-1])
        self._INR.popleft()

        try:
            INR_penalty = -sum(((2 / self._INR_range * (self._INR_mid - self._INR[-2] - (self._INR[-1]-self._INR[-2])/self._d_current*j)) ** 2
                                for j in range(1, self._d_current + 1)))  # negative squared distance as reward (used *2/range to normalize)
            dose_change_penalty = - self._dose_change_penalty_func(self._dose_list)
            reward = INR_penalty + self._dose_change_penalty_coef * dose_change_penalty
        except TypeError:
            reward = 0

        return RLData(reward, normalizer=lambda x: x)

    def reset(self):
        super().reset()
        self._interval_index = 0
        # self._d_current = self._interval[self._interval_index]

    def __repr__(self):
        try:
            return f"WarfarinModel: {[' '.join((str(k), ':', str(v))) for k, v in self._characteristics.items()]}, " \
                   f"INR: {self._INR}, d_prev: {self._dosing_intervals}, d: {self._d_current}"
        except:
            return 'WarfarinModel'
