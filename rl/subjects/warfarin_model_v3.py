# -*- coding: utf-8 -*-
'''
warfarin_model class
==================

This `warfarin_model` class implements a two compartment PK/PD model for warfarin. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from collections import deque
from math import exp, log, sqrt
from random import choice, normalvariate, sample, seed, shuffle

import numpy as np
import pandas as pd
from scipy.stats import lognorm

# from ..valueset import ValueSet
from ..rldata import RLData
from .subject import Subject


class WarfarinModel(Subject):
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
        Subject.__init__(self, **kwargs)

        Subject.set_defaults(self, patient_selection='random',
                             age_list=list(range(70, 86)),
                             CYP2C9_list=['*1/*1', '*1/*2',
                                          '*1/*3', '*2/*2', '*2/*3', '*3/*3'],
                             VKORC1_list=['G/G', 'G/A', 'A/A'],
                             age=60, CYP2C9='*1/*1', VKORC1='A/A', SS=0, max_time=24*90,
                             day=1, max_day=90, INR=[0], INR_current=0,
                             d_previous=0, d_current=1, d_max=30,
                             current_dose=0, max_dose=15, dose_steps=0.5, TTR_range=(2, 3),
                             dose_history=5, pill_per_day=1, randomized=True,
                             dose_list=[None]*5
                             )

        Subject.set_params(self, **kwargs)

        self._max_time = self._max_day*24
        self._patient = Patient(age=self._age, CYP2C9=self._CYP2C9, VKORC1=self._VKORC1,
                                randomized=self._randomized, max_time=self._max_time)

        self._INR = deque([0.0]*self._dose_history)
        self._INR[-1] = self._patient.INR([0])[-1]
        self._dose_list = deque([0.0]*self._dose_history)
        self._possible_actions = RLData([x*self._dose_steps
                         for x in range(int(self._max_dose/self._dose_steps), -1, -1)],
                        lower=0, upper=self._max_dose).as_rldata_array()
        if False:
            self._model_filename = ''
            self._Cs_super = 0
            self._age = 0
            self._CYP2C9 = ''
            self._VKORC1 = ''
            self._SS = 0
            self._max_time = 0
            self._rseed = 0
            self._day = 0
            self._max_day = 0
            self._current_dose = 0
            self._max_dose = 0
            self._dose_steps = 0
            self._dose_history = 0
            self._pill_per_day = 0

            self._patient_selection = ''
            self._age_list = []
            self._CYP2C9_list = []
            self._VKORC1_list = []
            self._INR = []

            self._d_previous = 0
            self._d_current = 0
            self._d_max = 0
            self._TTR_range = ()

    @property
    def state(self):
        return RLData({'Age': self._age,
                       'CYP2C9': self._CYP2C9,
                       'VKORC1': self._VKORC1,
                       'Doses': tuple(self._dose_list),
                       'INRs': tuple(self._INR)},
                      lower={'Age': 65,
                             'Doses': 0.0,
                             'INRs': 0.0},
                      upper={'Age': 85,
                             'Doses': 15.0,
                             'INRs': 10.0},
                      categories={'CYP2C9': self._CYP2C9_list,
                                  'VKORC1': self._VKORC1_list})

    @property
    def is_terminated(self):
        return self._day >= self._max_day

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

    def take_effect(self, _id, action):
        self._current_dose = action.value[0]
        self._d_previous = self._d_current
        self._dose_list.append(self._current_dose)
        self._dose_list.popleft()

        self._patient.dose = {self._day: self._current_dose}
        self._d_current = 1  # action.value[1]
        self._day += self._d_current

        self._INR.append(self._patient.INR(self._day)[-1])
        self._INR.popleft()

        try:
            # TTR = sum((self._TTR_range[0] <= self._INR_previous + (self._INR_current-self._INR_previous)/self._d_previous*j <= self._TTR_range[1]
            #            for j in range(self._d_previous)))
            INR_mid = (self._TTR_range[1] + self._TTR_range[0]) / 2
            INR_range = self._TTR_range[1] - self._TTR_range[0]
            reward = -sum(((2 * (INR_mid - self._INR[-2] + (self._INR[-1]-self._INR[-2])/self._d_previous*j)) ** 2
                           for j in range(self._d_previous))) / INR_range  # negative squared distance as reward (used *2/range to normalize)
        except TypeError:  # here I have assumed that for the first use of the pill, we don't have INR and TTR=0
            # TTR = 0
            reward = 0

        # return TTR*self._d_current
        return reward

    def reset(self):
        self._day = 1
        self._current_dose = 0
        self._dose_list = deque([0.0]*self._dose_history)
        self._INR = deque([0.0]*self._dose_history)

        if self._patient_selection == 'random':
            self._age = choice(self._age_list)
            self._CYP2C9 = choice(self._CYP2C9_list)
            self._VKORC1 = choice(self._VKORC1_list)

        self._patient = Patient(age=self._age, CYP2C9=self._CYP2C9, VKORC1=self._VKORC1,
                                randomized=self._randomized, max_time=self._max_time)
        self._INR = deque([0.0]*self._dose_history)
        self._INR[-1] = self._patient.INR([0])[-1]

        self._d_previous = 0
        self._d_current = 1

    def __repr__(self):
        try:
            return 'WarfarinModel: [age: {}, CYP2C9: {}, VKORC1: {}, INR: {}, d_prev: {}, d: {}]'.format(
                self._age, self._CYP2C9, self._VKORC1, self._INR, self._d_previous, self._d_current)
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


class Patient:
    '''
    Two compartment PK/PD model for wafarin.

    Attributes
    ----------
        dose: a dictionary containing with day as key and dose as value.

    Methods
    -------
        INR: returns a list of INR values corresponding to the given list of days.
    '''

    def __init__(self, age=50, CYP2C9='*3/*3', VKORC1='G/G', randomized=True, max_time=24,
                 dose_interval=24, dose={}, **kwargs):
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
        self._multiplication_term = part_1 + part_2 + part_3

        self._data = pd.DataFrame(columns=['dose']+times)
        self.dose = {0: 0.0}

    @property
    def dose(self):
        return self._data.dose

    @dose.setter
    def dose(self, dose):
        for d, v in dose.items():
            self._data.loc[d] = np.insert(
                self._Cs(dose=v, t0=d*self._dose_interval), 0, v)

    def _Cs(self, dose, t0):
        C_s_pred = ((self._ka * self._F * dose / 2) /
                    self._V1) * self._multiplication_term

        if t0 == 0:  # non steady-state
            C_s_error = [exp(normalvariate(0, 0.30)) for _ in range(len(C_s_pred))] if self._randomized \
                else [1]*len(C_s_pred)  # Sadjad
        else:  # steady-state
            C_s_error = [exp(normalvariate(0, 0.09)) for _ in range(len(C_s_pred))] if self._randomized \
                else [1]*len(C_s_pred)

        C_s = [C_s_pred[i] * C_s_error[i] for i in range(len(C_s_pred))]

        return np.pad(C_s, (t0, 0), 'constant', constant_values=(0,))[:self._max_time+1]

    def INR(self, days):
        if isinstance(days, int):
            days = [days]

        Cs_gamma = np.nan_to_num(np.power(np.sum(self._data[list(
            range(int(max(days)*self._dose_interval)+1))], axis=0), np.array(self._gamma)))

        A = [1]*7
        dA = [0]*7

        INR_max = 20
        baseINR = 1
        INR = []
        for d1, d2 in zip(sorted([0]+days[:-1]), sorted(days)):
            for i in range(int(d1*self._dose_interval), int(d2*self._dose_interval)):
                dA[0] = self._ktr1 * (1 - self._E_MAX * Cs_gamma[i] /
                                      (self._EC_50 ** self._gamma + Cs_gamma[i])) - self._ktr1*A[0]
                for j in range(1, 6):
                    dA[j] = self._ktr1 * (A[j-1] - A[j])

                dA[6] = self._ktr2 * (1 - self._E_MAX * Cs_gamma[i] /
                                      (self._EC_50 ** self._gamma + Cs_gamma[i])) - self._ktr2*A[6]
                for j in range(7):
                    A[j] += dA[j]

            e_INR = normalvariate(0, 0.0325) if self._randomized else 0
            INR.append(
                (baseINR + (INR_max*(1-A[5]*A[6]) ** self._lambda)) * exp(e_INR))

        return INR


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    max_day = 100
    p_count = range(100)
    p = [Patient(randomized=True, max_time=24*max_day + 1) for _ in p_count]
    for j in p_count:
        p[j].dose = {i: 7.5 for i in range(max_day)}
        # plt.plot(p.INR(list(i/24 for i in range(1, max_day*24 + 1))))
        plt.plot(list(range(24, (max_day+1)*24, 240)),
                 p[j].INR(list(i for i in range(1, max_day+1, 10))), 'x')
    plt.show()
