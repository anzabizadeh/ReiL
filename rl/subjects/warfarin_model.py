# -*- coding: utf-8 -*-
'''
warfarin_model class
==================

This `warfarin_model` class implements a two compartment PK/PD model for warfarin. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from math import exp

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
        Subject.set_defaults(self,
            Cs_super=0, AGE=60, CYP2C9='*1/*1', VKORC1='A/A', SS=1, maxTime=24, rseed=12345,
            day=0, state={}, 
            max_dose=15, dose_steps=0.5,
            state_function=lambda x: {'value': 0, 'min':0, 'max':0}, reward_function=lambda new_x, old_x: 1-new_x['cancer']/old_x['cancer'],
            termination_check= lambda x: x['tumor_cells']==0,
            )
        Subject.set_params(self, **kwargs)

    @property
    def state(self):
        return ValueSet(**self._state_function(self._x))
        # e = self._e(self._x)
        # for i, _ in enumerate(self._state_range):
        #     if e <= self._state_range[i]:
        #         return ValueSet(i, min=0, max=len(self._state_range))
        # return ValueSet(len(self._state_range), min=0, max=len(self._state_range))

    @property
    def is_terminated(self):
        return self._termination_check(self._x)

    @property
    def possible_actions(self):
        return ValueSet([self._max_dose*x/self._dose_steps for x in range(0, self._dose_steps+1)], min=0, max=self._max_dose, 
                        binary=lambda x: (int(x * self._dose_steps // self._max_dose), self._dose_steps+1)).as_valueset_array()

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
        self._drug['infusion_rate'] = action.value[0]
        x_dot = {}
        x_dot['day'] = 1
        x_dot['normal_cells'] = self._x['normal_cells'] * (
                                    self._normal_cells['growth_rate'] * (1 - self._normal_cells['carrying_capacity']*self._x['normal_cells'])
                                    - self._competition_term['normal_from_tumor'] * self._x['tumor_cells']
                                    - self._drug['normal_cell_kill_rate'] * (1-exp(-self._x['drug'])))

        x_dot['tumor_cells'] = self._x['tumor_cells'] * (
                                    self._tumor_cells['growth_rate'] * (1 - self._tumor_cells['carrying_capacity']*self._x['tumor_cells'])
                                    - self._competition_term['tumor_from_immune'] * self._x['immune_cells']
                                    - self._competition_term['tumor_from_normal'] * self._x['normal_cells']
                                    - self._drug['tumor_cell_kill_rate'] * (1-exp(-self._x['drug'])))

        x_dot['immune_cells'] = self._x['immune_cells'] * (
                                    self._immune_cells['response_rate']*self._x['tumor_cells']/(self._immune_cells['threshold_rate']+self._x['tumor_cells'])
                                    - self._immune_cells['death_rate']
                                    - self._competition_term['immune_from_tumor'] * self._x['tumor_cells']
                                    - self._drug['immune_cell_kill_rate'] * (1-exp(-self._x['drug']))) + self._immune_cells['influx_rate']

        x_dot['drug'] = self._x['drug'] * (-self._drug['decay_rate']) + self._drug['infusion_rate']
        new_x = {}
        for i in self._x:
            try:
                new_x[i] = max(self._x[i] + x_dot[i], 0)  # I manually enforced >=0 constraint, but it shouldn't be!
            except:
                new_x[i] = self._x[i]

        r = self._reward_function(new_x, self._x)
        self._x = new_x

        return r

    def reset(self):
        self._x = {'drug': self._drug['initial_value'], 'normal_cells': self._normal_cells['initial_value'],
                   'tumor_cells': self._tumor_cells['initial_value'], 'immune_cells': self._immune_cells['initial_value'], 'day': 1}

    def __repr__(self):
        try:
            return 'CancerModel: [day: {}, N: {}, T: {}, N: {}, C: {}]'.format(
                self._x['day'], self._x['normal_cells'], self._x['tumor_cells'], self._x['immune_cells'], self._x['drug'])
        except:
            return 'CancerModel'