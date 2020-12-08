# -*- coding: utf-8 -*-
'''
cancer_model class
==================

This `cancer_model` class implements a four-state nonlinear cancer chemotherapy model. 


'''

from ..legacy import ValueSet
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
        self._drug_cap = lambda x: kwargs.get('u_max', 0)
        if 'drug_cap' in kwargs:
            self.set_params(drug_cap=kwargs['drug_cap'])
        super().__init__(**kwargs)

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._drug_cap=lambda x, day: self._u_max

        self._x['drug_cap'] = self._drug_cap(self._x)

    @property
    def possible_actions(self):
        return ValueSet([self._x['drug_cap']*x/self._u_steps for x in range(0, self._u_steps+1)], min=0, max=self._u_max, 
                        binary=lambda x: (int(x * self._u_steps // self._u_max), self._u_steps+1)).as_valueset_array()

    def take_effect(self, action, _id=None):
        r = CancerModel.take_effect(self, action, _id)
        self._x['drug_cap'] = self._drug_cap(self._x)
        return r

    def reset(self):
        CancerModel.reset(self)
        self._x['drug_cap'] = self._drug_cap(self._x)

    def __repr__(self):
        try:
            return f"ConstrainedCancerModel: [day: {self._x['day']}, N: {self._x['normal_cells']}, T: {self._x['tumor_cells']}, " \
                   f"N: {self._x['immune_cells']}, C: {self._x['drug']}, drug cap: {self._x.get('drug_cap','N/A')}]"
        except:
            return 'ConstrainedCancerModel'
