# -*- coding: utf-8 -*-
'''
Patient class
=============

This class is the base class to model patients with different characteristics.
'''

from typing import Any, Dict

from reil.datatypes.feature import Feature, FeatureGenerator
from reil.healthcare.mathematical_models import HealthMathModel


class Patient:
    '''
    Base class for patients in healthcare.
    '''
    def __init__(self, model: HealthMathModel, **feature_values: Any) -> None:
        '''
        Parameters
        ----------
        model:
            A `HealthMathModel` to be used to model patient's behavior.

        feature_values:
            Keyword arguments by which some of the `features` of the patient
            can be determined. For example, if "age" is one of the features,
            age=40.0 will set the initial age to 40.0.
        '''
        if not hasattr(self, 'feature_gen_set'):
            self.feature_gen_set: Dict[str, FeatureGenerator] = {}
        if not hasattr(self, 'feature_set'):
            self.feature_set: Dict[str, Feature] = {}

        for k in self.feature_gen_set:
            self.feature_set[k] = self.feature_gen_set[k](
                feature_values.get(k))

        self._model = model
        self._model.setup(**self.feature_set)

    def generate(self) -> None:
        '''
        Generate a new patient.

        This method calls every `feature`, and then sets
        up to `model` using the new values.
        '''
        for k in self.feature_gen_set:
            self.feature_set[k] = self.feature_gen_set[k]()

        self._model.setup(**self.feature_set)

    def model(self, **inputs: Any) -> Dict[str, Any]:
        '''Model patient's behavior.

        Arguments
        ---------
        inputs:
            Keyword arguments that specify inputs to the model. For example, if
            `dose` is a necessary input, `model(dose=10.0)` will provide the
            model with dose of 10.0.

        Returns
        -------
        :
            All the outputs of running the mathematical model, given the input.
        '''
        return self._model.run(**inputs)
