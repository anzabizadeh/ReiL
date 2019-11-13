# -*- coding: utf-8 -*-
'''
warfarin_model class
==================

This `warfarin_model` class implements a two compartment PK/PD model for warfarin. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from math import exp
from random import choice
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

from ..legacy.valueset import ValueSet
from ..subjects import Subject


class WarfarinModel(Subject):
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
        Subject.__init__(self, **kwargs)

        Subject.set_defaults(self, model_filename='./rl/subjects/warfarin.pkpd', patient_selection='random',
                             age_list=list(range(70, 86)),
                             CYP2C9_list=['*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3'],
                             VKORC1_list=['G/G', 'G/A', 'A/A'],
                             Cs_super=0, age=60, CYP2C9='*1/*1', VKORC1='A/A', SS=1, maxTime=24, rseed=12345,
                             day=1, max_day=90, INR_previous=0, INR_current=0,
                             d_previous=0, d_current=1, d_max=30,
                             current_dose=0, max_dose=15, dose_steps=0.5, TTR_range=(2, 3)
                             )

        Subject.set_params(self, **kwargs)

        try:
            with open(self._model_filename, mode='r') as file:
                warfarin_code = file.read()
        except FileNotFoundError:
            raise FileNotFoundError('The PK/PD Model Not Found.')
        utils = importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(StrVector(['deSolve']))
        robjects.r('set.seed({})'.format(self._rseed))
        robjects.r(warfarin_code)
        self._hamberg_2007 = robjects.r['hamberg_2007']

        result = dict(zip(('INR', 'Cs', 'out', 'INRv', 'parameters'),
                          self._hamberg_2007(self._current_dose, self._Cs_super, self._age, self._CYP2C9, self._VKORC1, 0, self._maxTime)))
        self._Cs_super = result['Cs'][-1]
        self._INR_current = result['INR'][-1]

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

            self._patient_selection = 'random'
            self._age_list=list(range(70, 86))
            self._CYP2C9_list=['*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3']
            self._VKORC1_list=['G/G', 'G/A', 'A/A']
            result = dict(zip(('INR', 'Cs', 'out', 'INRv', 'parameters'),
                              self._hamberg_2007(self._current_dose, self._Cs_super, self._age, self._CYP2C9, self._VKORC1, 0, self._maxTime)))
            self._INR_previous = 0
            self._Cs_super = result['Cs'][-1]
            self._INR_current = result['INR'][-1]

            self._d_previous = 0
            self._d_current = 1
            self._d_max = 30
            self._TTR_range = (2, 3)

    @property  # binary representation is not complete!
    def state(self):
        return ValueSet((self._age, self._CYP2C9, self._VKORC1,
                         round(self._Cs_super, 1), self._current_dose,
                         round(self._INR_previous, 1), round(
                             self._INR_current, 1),
                         self._d_previous, self._d_current),
                        min=None, max=None,
                        binary=lambda x: [1 if x==a else 0 for a in self._age_list] if x in self._age_list
                            else [1 if x==a else 0 for a in self._CYP2C9_list] if x in self._CYP2C9_list
                            else [1 if x==a else 0 for a in self._VKORC1_list]
                        )

    @property
    def is_terminated(self):  # ready
        return self._day >= self._max_day

    @property
    # binary should be implemented for day!!!  action[0]=dose, action[1]=interval
    def possible_actions(self):
        return ValueSet([(x*self._dose_steps, d)
                         for x in range(int(self._max_dose/self._dose_steps)+1)
                         for d in range(1, min(self._d_max, self._max_day-self._day)+1)],
                        min=(0, 1), max=(self._max_dose, 30),
                        binary=lambda x: [1 if x==a else 0 for a in range(int(self._max_dose/self._dose_steps)+1)]).as_valueset_array()

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

    def take_effect(self, _id, action):
        self._current_dose = action.value[0]
        self._d_previous = self._d_current
        self._d_current = action.value[1]
        self._day += self._d_current
        Cs = self._Cs_super
        for _ in range(self._d_current):
            result = dict(zip(('INR', 'Cs', 'out', 'INRv', 'parameters'),
                              self._hamberg_2007(self._current_dose, Cs, self._age, self._CYP2C9, self._VKORC1, self._SS, self._maxTime)))
            Cs = result['Cs'][-1]

        self._INR_previous = self._INR_current
        self._Cs_super = result['Cs'][-1]
        self._INR_current = result['INR'][-1]

        try:
            TTR = sum((self._TTR_range[0] <= self._INR_previous + (self._INR_current-self._INR_previous)/self._d_previous*j <= self._TTR_range[1]
                       for j in range(self._d_previous)))
        except TypeError:  # here I have assumed that for the first use of the pill, we don't have INR and TTR=0
            TTR = 0

        # if TTR == 0:
        #     TTR = -0.01

        return TTR*self._d_current

    def reset(self):
        self._Cs_super = 0
        self._day = 1
        self._current_dose = 0

        if self._patient_selection == 'random':
            self._age = choice(self._age_list)
            self._CYP2C9 = choice(self._CYP2C9_list)
            self._VKORC1 = choice(self._VKORC1_list)

        result = dict(zip(('INR', 'Cs', 'out', 'INRv', 'parameters'),
                          self._hamberg_2007(self._current_dose, self._Cs_super, self._age, self._CYP2C9, self._VKORC1, 0, self._maxTime)))
        self._INR_previous = 0
        self._Cs_super = result['Cs'][-1]
        self._INR_current = result['INR'][-1]

        self._d_previous = 0
        self._d_current = 1

    def __repr__(self):
        try:
            return 'WarfarinModel: [{}, {}, {}, INR_prev: {}, INR: {}, d_prev: {}, d: {}]'.format(
                self._age, self._CYP2C9, self._VKORC1, self._INR_previous, self._INR_current, self._d_previous, self._d_current)
        except:
            return 'WarfarinModel'
