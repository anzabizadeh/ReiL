# -*- coding: utf-8 -*-
'''
cancer_model class
==================

This `cancer_model` class implements a four-state nonlinear cancer
chemotherapy model proposed by Lisette et al. (2003).
'''

from reil.subjects.healthcare import Patient
from reil.datatypes import Feature
from reil.utils import functions


class CancerPatient(Patient):
    feature_set = {
        'age': Feature.numerical(  # Aurora population
            lower=18.0, upper=100.0, mean=67.30, stdev=13.43,
            generator=functions.random_truncated_normal),
        'weight': Feature.numerical(  # lb  - Aurora population
            lower=70.0, upper=500.0, mean=199.24, stdev=54.71,
            generator=functions.random_truncated_normal),
        'height': Feature.numerical(  # in - Aurora population
            lower=45.0, upper=85.0, mean=66.78, stdev=4.31,
            generator=functions.random_truncated_normal),
        'gender': Feature.categorical(  # Aurora population
            categories=('Female', 'Male'),
            probabilities=(0.5314, 0.4686),
            generator=functions.random_categorical)}

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
        self.set_defaults(
            drug={'initial_value': 0, 'infusion_rate': 0, 'decay_rate': 0,
                'normal_cell_kill_rate': 0, 'tumor_cell_kill_rate': 0, 'immune_cell_kill_rate': 0},
            normal_cells={'initial_value': 0, 'growth_rate': 0, 'carrying_capacity': 0},
            tumor_cells={'initial_value': 0, 'growth_rate': 0, 'carrying_capacity': 0},
            immune_cells={'initial_value': 0, 'influx_rate': 0, 'threshold_rate': 1, 'response_rate': 0, 'death_rate': 0},
            competition_term={'normal_from_tumor':0, 'tumor_from_normal': 0, 'tumor_from_immune': 0, 'immune_from_tumor': 0},
            x={'drug': 0, 'normal_cells': 0, 'tumor_cells': 0, 'immune_cells': 0, 'day': 1},
            state_function=lambda x: {'value': 0, 'min':0, 'max':0}, rl_function=lambda new_x, old_x: 1-new_x['cancer']/old_x['cancer'],
            termination_check= lambda x: x['tumor_cells']==0,
            e=lambda x: 0, state_range=[0],
            u_max=10, u_steps=20)
        self.set_params(**kwargs)
        super().__init__(**kwargs)


        self._x = {'drug': self._drug['initial_value'], 'normal_cells': self._normal_cells['initial_value'],
                'tumor_cells': self._tumor_cells['initial_value'], 'immune_cells': self._immune_cells['initial_value'], 'day': 1}

    @property
    def state(self):
        return ValueSet(**self._state_function(self._x))
        e = self._e(self._x)
        for i, _ in enumerate(self._state_range):
            if e <= self._state_range[i]:
                return ValueSet(i, min=0, max=len(self._state_range))
        return ValueSet(len(self._state_range), min=0, max=len(self._state_range))

    @property
    def is_terminated(self):
        return self._termination_check(self._x)

    @property
    def possible_actions(self):
        return ValueSet([self._u_max*x/self._u_steps for x in range(0, self._u_steps+1)], min=0, max=self._u_max, 
                        binary=lambda x: (int(x * self._u_steps // self._u_max), self._u_steps+1)).as_valueset_array()

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

        r = self._rl_function(new_x, self._x)
        self._x = new_x

        return r

    def reset(self):
        self._x = {'drug': self._drug['initial_value'], 'normal_cells': self._normal_cells['initial_value'],
                'tumor_cells': self._tumor_cells['initial_value'], 'immune_cells': self._immune_cells['initial_value'], 'day': 1}

    def __repr__(self):
        try:
            return f"CancerModel: [day: {self._x['day']}, N: {self._x['normal_cells']}, T: {self._x['tumor_cells']}, N: {self._x['immune_cells']}, C: {self._x['drug']}]"
        except:
            return 'CancerModel'
