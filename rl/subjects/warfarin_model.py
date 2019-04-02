# -*- coding: utf-8 -*-
'''
warfarin_model class
==================

This `warfarin_model` class implements a two compartment PK/PD model for warfarin. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from math import exp

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

from ..valueset import ValueSet
from .subject import Subject


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
        Subject.set_defaults(self, model_filename='./rl/subjects/warfarin.pkpd',
                             Cs_super=0, age=60, CYP2C9='*1/*1', VKORC1='A/A', SS=1, maxTime=24, rseed=12345,
                             day=0, max_day=90, INR_previous=None, INR_current=None,
                             d_previous=None, d_current=1,
                             max_dose=15, dose_steps=0.5, TTR_range=(2, 3)
                             )
        Subject.set_params(self, **kwargs)

        try:
            with open(self._model_filename, mode='r') as file:
                warfarin_code = file.read()
        except FileNotFoundError:
            raise FileNotFoundError('The PK/PD Model Not Found')
        utils = importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(StrVector(['deSolve']))
        robjects.r(warfarin_code)
        self._hamberg_2007 = robjects.r['hamberg_2007']

        if False:
            self._model_filename='./rl/subjects/warfarin.pkpd'
            self._Cs_super=0
            self._age=60
            self._CYP2C9='*1/*1'
            self._VKORC1='A/A'
            self._SS=1
            self._maxTime=24
            self._rseed=12345
            self._day=0
            self._max_day=90
            self._max_dose=15
            self._dose_steps=0.5
            self._INR_previous=None
            self._INR_current=None
            self._d_previous=None
            self._d_current=1
            self._TTR_range = (2, 3)

    @property
    def state(self):
        return ValueSet((self._age, self._CYP2C9, self._VKORC1, self._INR_previous, self._INR_current, self._d_previous, self._d_current), min=None, max=None)


    @property
    def is_terminated(self):  # ready
        return self._day == self._max_day

    @property
    def possible_actions(self):  # ready
        return ValueSet([x*self._dose_steps for x in range(int(self._max_dose/self._dose_steps)+1)], min=0, max=self._max_dose,
                        binary=lambda x: (int(x * self._dose_steps // self._max_dose), self._dose_steps+1)).as_valueset_array()

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
        result = dict(zip(('INR', 'Cs', 'out', 'INRv', 'parameters'),
                          self._hamberg_2007(action.value[0], self._Cs_super, self._age, self._CYP2C9, self._VKORC1, self._SS, self._maxTime, self._rseed)))
        self._INR_previous = self._INR_current
        self._d_previous = self._d_current
        self._day += self._d_previous

        self._Cs_super = result['Cs'][-1]
        self._INR_current = result['INR'][-1]
        self._d_current = 1
        try:
            TTR = sum((self._TTR_range[0] <= (self._INR_current-self._INR_previous)/self._d_previous*j <= self._TTR_range[1]
                for j in range(self._d_previous)))
        except TypeError:  # here I have assumed that for the first use of the pill, we don't have INR and TTR=0
            TTR = 0

        return TTR*self._d_previous

    def reset(self):
        self._Cs_super=0
        self._day=0
        self._INR_previous=None
        self._INR_current=None
        self._d_previous=None
        self._d_current=1

    def __repr__(self):
        try:
            return 'WarfarinModel: [day: {}, N: {}, T: {}, N: {}, C: {}]'.format(
                self._x['day'], self._x['normal_cells'], self._x['tumor_cells'], self._x['immune_cells'], self._x['drug'])
        except:
            return 'WarfarinModel'

