# -*- coding: utf-8 -*-
'''
HealthMathModel class
=====================

The base class of all mathematical models, e.g. ODE, PDE, PK/PD, etc. in
healthcare.
'''
from typing import Any, Dict

from reil.datatypes.feature import Feature


class HealthMathModel:
    '''
    The base class of all mathematical models in healthcare.
    '''

    def setup(self, **arguments: Feature[Any]) -> None:
        '''
        Set up the model.

        Arguments
        ---------
        arguments:
            Any parameter that the model needs to setup initially.
        '''
        raise NotImplementedError

    def run(self, **inputs: Any) -> Dict[str, Any]:
        '''
        Run the model.

        Arguments
        ---------
        inputs:
            Any input arguments that the model needs for the run.

        Returns
        -------
        :
            A dictionary of model's return values.
        '''
        raise NotImplementedError
