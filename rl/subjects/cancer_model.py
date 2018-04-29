# -*- coding: utf-8 -*-
'''
cancer_model class
==================

This `cancer_model` class implements a four-state nonlinear cancer chemotherapy model. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from .subject import Subject
from ..valueset import ValueSet


class CancerModel(Subject):
    '''
    Four-state nonlinear cancer chemotherapy model.
    
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
            drug={'initial_value': 0, 'concentration': 0, 'infusion_rate': 0, 'decay_rate': 0,
                  'normal_cell_kill_rate': 0, 'tumor_cell_kill_rate': 0, 'immune_cell_kill_rate': 0, 'competing_factor': 0},
            normal_cells={'initial_value': 0, 'growth_rate': 0, 'carrying_capacity': 0, 'death_rate': 0, 'competing_factor': 0},
            tumor_cells={'initial_value': 0, 'growth_rate': 0, 'carrying_capacity': 0, 'death_rate': 0, 'competing_factor': 0},
            immune_cells={'initial_value': 0, 'influx_rate': 0, 'threshold_rate': 1, 'response_rate': 0, 'competing_factor': 0},
            x={'drug': 0, 'normal_cells': 0, 'tumor_cells': 0, 'immune_cells': 0},
            e=lambda x: 0, state_range=[0], termination_check= lambda x: x['tumor_cells']==0,
            u_max=10, u_steps=20)
        Subject.set_params(self, **kwargs)

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._agent_list, self._drug, self._normal_cells, self._tumor_cells = {}, {}, {}, {}
            self._immune_cells, self._x, self._e, self._state_range, self._termination_check = {}, {}, lambda x: 0, [0], lambda x: x['tumor_cells']==0
            self._u_max, self._u_steps = 10, 20

    @property
    def state(self):
        e = self._e(self._x)
        for i in enumerate(self._state_range):
            if e <= self._state_range[i]:
                return ValueSet(i, min=0, max=len(self._state_range))
        return ValueSet(len(self._state_range), min=0, max=len(self._state_range))

    @property
    def is_terminated(self):
        return self._termination_check(self._x)

    @property
    def possible_actions(self):
        return ValueSet([self._u_max*x/self._u_steps for x in range(0, self._u_steps+1)], min=0, max=self._u_max).as_valueset_array()

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
        self._drug['infusion_rate'] = action.value()
        x_dot = {}
        x_dot['normal_cells'] = self._x['normal_cells'] * (
                                    self._normal_cells['growth_rate'] * (1 - self._normal_cells['carrying_capacity']*self._x['normal_cells'])
                                    - self._tumor_cells['competing_factor'] * self._x['tumor_cells']
                                    - self._drug['normal_cell_kill_rate'] * self._x['drug'])

        x_dot['tumor_cells'] = self._x['tumor_cells'] * (
                                    self._tumor_cells['growth_rate'] * (1 - self._tumor_cells['carrying_capacity']*self._x['tumor_cells'])
                                    - self._immune_cells['competing_factor'] * self._x['immune_cells']
                                    - self._normal_cells['competing_factor'] * self._x['normal_cells']
                                    - self._drug['tumor_cell_kill_rate'] * self._x['drug'])

        x_dot['immune_cells'] = self._x['immune_cells'] * (
                                    self._immune_cells['response_rate']*self._x['tumor_cells']/(self._immune_cells['threshold_rate']+self._x['tumor_cells'])
                                    - self._immune_cells['death_rate']
                                    - self._tumor_cells['competing_factor'] * self._x['tumor_cells']
                                    - self._drug['immune_cell_kill_rate'] * self._x['drug']) + self._immune_cells['influx_rate']
        
        x_dot['drug'] = self._x['drug'] * (-self._drug['decay_rate']) + self._drug['infusion_rate']
        new_x = {}
        for i in self._x:
            new_x[i] = self._x[i] + x_dot[i]

        r = 1 - self._e(new_x)/self._e(self._x)
        self._x = new_x

        return r

    def reset(self):
        self._x = [self._normal_cells['initial_value'], self._tumor_cells['initial_value'], self._immune_cells['initial_value'], self._drug['initial_value']]

    def __repr__(self):
        return 'CancerModel'