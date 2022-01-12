# -*- coding: utf-8 -*-
'''
HealthMathModel class
=====================

The base class of all mathematical models, e.g. ODE, PDE, PK/PD, etc. in
healthcare.
'''
from typing import Any, Dict, Optional

from reil.datatypes.feature import FeatureGeneratorSet, FeatureSet


class HealthMathModel:
    '''
    The base class of all mathematical models in healthcare.
    '''
    _parameter_generators = FeatureGeneratorSet()

    @classmethod
    def generate(
            cls,
            input_features: Optional[FeatureSet] = None,
            **kwargs: Any) -> FeatureSet:
        if input_features:
            temp = input_features.value
            temp.update(kwargs)
            return cls._parameter_generators(temp)

        return cls._parameter_generators(None)

    def setup(self, arguments: FeatureSet) -> None:
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
