# -*- coding: utf-8 -*-
'''
cancer_model class
==================

This `cancer_model` class implements a four-state nonlinear cancer chemotherapy model. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from ..valueset import ValueSet
from .cancer_model import CancerModel


class ConstrainedCancerModel(CancerModel):
    '''
    Four-state nonlinear cancer chemotherapy model with constraint on drug dose.
    
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
        CancerModel.set_defaults(self, drug_cap=lambda x: self._u_max)
        CancerModel.__init__(self, **kwargs)
        # CancerModel.set_defaults(self,
        #     drug={'initial_value': 0, 'infusion_rate': 0, 'decay_rate': 0,
        #           'normal_cell_kill_rate': 0, 'tumor_cell_kill_rate': 0, 'immune_cell_kill_rate': 0},
        #     normal_cells={'initial_value': 0, 'growth_rate': 0, 'carrying_capacity': 0},
        #     tumor_cells={'initial_value': 0, 'growth_rate': 0, 'carrying_capacity': 0},
        #     immune_cells={'initial_value': 0, 'influx_rate': 0, 'threshold_rate': 1, 'response_rate': 0, 'death_rate': 0},
        #     competition_term={'normal_from_tumor':0, 'tumor_from_normal': 0, 'tumor_from_immune': 0, 'immune_from_tumor': 0},
        #     x={'drug': 0, 'normal_cells': 0, 'tumor_cells': 0, 'immune_cells': 0},
        #     e=lambda x: 0, state_range=[0], termination_check= lambda x: x['tumor_cells']==0,
        #     u_max=10, u_steps=20, day=1,
        #     drug_cap=lambda x: self._u_max)
        # CancerModel.set_params(self, **kwargs)
        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        # if False:
        #     self._agent_list, self._drug, self._normal_cells, self._tumor_cells = {}, {}, {}, {}
        #     self._immune_cells, self._x, self._e, self._state_range, self._termination_check = {}, {}, lambda x: 0, [0], lambda x: x['tumor_cells']==0
        #     self._competition_term, self._u_max, self._u_steps, self._day = {}, 10, 20, 1
        #     self._drug_cap=lambda x, day: self._u_max

        # self._x = {'drug': self._drug['initial_value'], 'normal_cells': self._normal_cells['initial_value'],
        #            'tumor_cells': self._tumor_cells['initial_value'], 'immune_cells': self._immune_cells['initial_value']}

    @property
    def possible_actions(self):
        return ValueSet([self._drug_cap(self._x)*x/self._u_steps for x in range(0, self._u_steps+1)], min=0, max=self._u_max, 
                        binary=lambda x: (int(x * self._u_steps // self._u_max), self._u_steps+1)).as_valueset_array()

    def __repr__(self):
        try:
            return 'ConstrainedCancerModel: [day: {}, N: {}, T: {}, N: {}, C: {}]'.format(
                self._x['day'], self._x['normal_cells'], self._x['tumor_cells'], self._x['immune_cells'], self._x['drug'])
        except:
            return 'ConstrainedCancerModel'