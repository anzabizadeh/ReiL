# -*- coding: utf-8 -*-
'''
warfarin_model class
==================

This `warfarin_model` class implements a two compartment PK/PD model for warfarin. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from collections import deque
from math import exp
from random import choice, shuffle

from ..legacy import ValueSet
from ..subjects import Subject
from .warfarin_pkpd_model import hamberg_2007


class WarfarinModel_v2(Subject):
    '''
    Two compartment PK/PD model for wafarin.

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
                             age_list=list(range(70, 86)),
                             CYP2C9_list=['*1/*1', '*1/*2',
                                          '*1/*3', '*2/*2', '*2/*3', '*3/*3'],
                             VKORC1_list=['G/G', 'G/A', 'A/A'],
                             Cs_super=0, age=60, CYP2C9='*1/*1', VKORC1='A/A', SS=0, maxTime=24,
                             day=1, max_day=90, INR_previous=0, INR_current=0,
                             d_previous=0, d_current=1, d_max=30,
                             current_dose=0, max_dose=15, dose_steps=0.5, TTR_range=(2, 3),
                             dose_history=10, pill_per_day=1, randomized=True,
                             dose_list=[None]*10
                             )
        self.set_params(**kwargs)
        super().__init__(**kwargs)

        self._maxTime = self._dose_history*24
        result = hamberg_2007(self._current_dose, self._Cs_super, self._age, self._CYP2C9,
                              self._VKORC1, self._SS, self._maxTime, randomized=self._randomized)
        self._Cs_super = None
        self._Cs = deque([result['Cs']])
        self._INR_current = result['INR']
        self._dose_list=deque([0.0]*self._dose_history)

        if False:
            self._model_filename = './rl/subjects/warfarin.pkpd'
            self._Cs_super = 0
            self._age = 60
            self._CYP2C9 = '*1/*1'
            self._VKORC1 = 'A/A'
            self._SS = 1
            self._maxTime = 24
            self._rseed = 12345
            self._day = 1
            self._max_day = 90
            self._current_dose = 0
            self._max_dose = 15
            self._dose_steps = 0.5
            self._dose_history = 10
            self._pill_per_day = 1

            self._patient_selection = 'random'
            self._age_list = list(range(70, 86))
            self._CYP2C9_list = ['*1/*1', '*1/*2',
                                 '*1/*3', '*2/*2', '*2/*3', '*3/*3']
            self._VKORC1_list = ['G/G', 'G/A', 'A/A']
            result = hamberg_2007(self._current_dose, self._Cs_super, self._age,
                                  self._CYP2C9, self._VKORC1, 0, self._maxTime, randomized=self._randomized)
            self._INR_previous = 0
            self._Cs_super = result['Cs']
            self._INR_current = result['INR']

            self._d_previous = 0
            self._d_current = 1
            self._d_max = 30
            self._TTR_range = (2, 3)

    @property  # binary representation is not complete!
    def state(self):
        try:
            Cs = round(self._Cs_super, 1)
        except TypeError:
            Cs = self._Cs_super
        return ValueSet((self._age, self._CYP2C9, self._VKORC1,
                         Cs, tuple(self._dose_list), round(self._INR_current, 1),
                         self._d_previous, self._d_current),
                        min=None, max=None,
                        binary=lambda x: [1 if x == a else 0 for a in self._age_list] if x in self._age_list
                        else [1 if x == a else 0 for a in self._CYP2C9_list] if x in self._CYP2C9_list
                        else [1 if x == a else 0 for a in self._VKORC1_list] if x in self._VKORC1_list
                        else list([d/self._max_dose for d in x]) if isinstance(x, tuple)
                        # for Cs, INR_previous, INR_current, d_previous, d_current I divide by 30 to normalize
                        else [0] if x is None else [x/30]
                        )

        # return ValueSet((self._age, self._CYP2C9, self._VKORC1,
        #                  Cs, tuple(self._dose_list), round(self._INR_current, 1),
        #                  self._d_previous, self._d_current),
        #                 min=None, max=None,
        #                 binary=lambda x: [1 if x == a else 0 for a in self._age_list] if x in self._age_list
        #                 else [1 if x == a else 0 for a in self._CYP2C9_list] if x in self._CYP2C9_list
        #                 else [1 if x == a else 0 for a in self._VKORC1_list] if x in self._VKORC1_list
        #                 else list([1 if a.value==b else 0 for b in x for a in self.possible_actions]) if isinstance(x, tuple)
        #                 # for Cs, INR_previous, INR_current, d_previous, d_current I divide by 30 to normalize
        #                 else [0] if x is None else [x/30]
        #                 )
        # return ValueSet((self._age, self._CYP2C9, self._VKORC1,
        #                  Cs, self._current_dose,
        #                  round(self._INR_previous, 1), round(
        #                      self._INR_current, 1),
        #                  self._d_previous, self._d_current),
        #                 min=None, max=None,
        #                 binary=lambda x: [1 if x == a else 0 for a in self._age_list] if x in self._age_list
        #                 else [1 if x == a else 0 for a in self._CYP2C9_list] if x in self._CYP2C9_list
        #                 else [1 if x == a else 0 for a in self._VKORC1_list] if x in self._VKORC1_list
        #                 # for Cs, current_dose, INR_previous, INR_current, d_previous, d_current I divide by 30 to normalize
        #                 else [0] if x is None else [x/30]
        #                 )

    @property
    def is_terminated(self):  # ready
        return self._day >= self._max_day

    @property
    # only considers the dose
    def possible_actions(self):
        return ValueSet([x*self._dose_steps
                         for x in range(int(self._max_dose/self._dose_steps), -1, -1)],
                        min=0, max=self._max_dose,
                        binary=lambda x: [1 if x == a else 0 for a in range(int(self._max_dose/self._dose_steps)+1)]).as_valueset_array()
        # binary should be implemented for day!!!  action[0]=dose, action[1]=interval
        # return ValueSet([(x*self._dose_steps, d)
        #                  for x in range(int(self._max_dose/self._dose_steps), -1, -1)
        #                  for d in range(1, min(self._d_max, self._max_day-self._day)+1)],
        #                 min=(0, 1), max=(self._max_dose, 30),
        #                 binary=lambda x: [1 if x == a else 0 for a in range(int(self._max_dose/self._dose_steps)+1)]).as_valueset_array()

        # binary=lambda (x, d): (int(x * self._dose_steps // self._max_dose), self._dose_steps+1)).as_valueset_array()

    def register(self, agent_name):  # should work
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
        self._d_current = 1  # action.value[1]
        self._day += self._d_current

        for _ in range(self._d_current):
            while len(self._Cs) > self._dose_history:
                self._Cs.pop()
            n = len(self._Cs)
            l = (self._Cs[i][-int((self._dose_history - (n-i) /
                                   self._pill_per_day)*24+1)] for i in range(n))
            Cs_super_previous = self._Cs_super
            self._Cs_super = sum(l)
            if Cs_super_previous == self._Cs_super:
                self._SS = 1
            else:
                self._SS = 0

            result = hamberg_2007(self._current_dose, self._Cs_super, self._age, self._CYP2C9,
                                  self._VKORC1, self._SS, self._maxTime, randomized=self._randomized)
            self._Cs.append(result['Cs'])

        self._INR_previous = self._INR_current
        # self._Cs_super = result['Cs']
        self._INR_current = result['INRv'][24]

        try:
            # TTR = sum((self._TTR_range[0] <= self._INR_previous + (self._INR_current-self._INR_previous)/self._d_previous*j <= self._TTR_range[1]
            #            for j in range(self._d_previous)))
            INR_mid = (self._TTR_range[1] + self._TTR_range[0]) / 2
            INR_range = self._TTR_range[1] - self._TTR_range[0]
            reward = -sum(((INR_mid - self._INR_previous + (self._INR_current-self._INR_previous)/self._d_previous*j) ** 2
                           for j in range(self._d_previous))) * 2 / INR_range  # negative squared distance as reward (used *2/range to normalize)
        except TypeError:  # here I have assumed that for the first use of the pill, we don't have INR and TTR=0
            # TTR = 0
            reward = 0

        # return TTR*self._d_current
        return reward

    def reset(self):
        self._day = 1
        self._current_dose = 0
        self._SS = 0
        self._dose_list=deque([0.0]*self._dose_history)

        if self._patient_selection == 'random':
            self._age = choice(self._age_list)
            self._CYP2C9 = choice(self._CYP2C9_list)
            self._VKORC1 = choice(self._VKORC1_list)

        result = hamberg_2007(dose=self._current_dose,
                              Cs_super=0.0,
                              AGE=self._age,
                              CYP2C9=self._CYP2C9,
                              VKORC1=self._VKORC1,
                              SS=self._SS,
                              maxTime=self._maxTime, randomized=self._randomized)
        self._Cs_super = None
        self._Cs = [result['Cs']]
        self._INR_current = result['INR']
        self._INR_previous = 0

        self._d_previous = 0
        self._d_current = 1

    def __repr__(self):
        try:
            return f'WarfarinModel: [age: {self._age}, CYP2C9: {self._CYP2C9}, VKORC1: {self._VKORC1}, ' \
                   f'INR_prev: {self._INR_previous}, INR: {self._INR_current}, d_prev: {self._d_previous}, d: {self._d_current}]'
        except:
            return 'WarfarinModel'
