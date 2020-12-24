# -*- coding: utf-8 -*-
'''
Patient class
===========

This class is the base class to model patients with different characteristics.
'''

from typing import Any, Dict

from reil.datatypes import Feature
from reil.subjects.healthcare.mathematical_models import HealthMathModel


class Patient:
    '''
    Base class for patients in healthcare.
    '''
    feature_set: Dict[str, Feature] = {}

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
        for k, v in self.feature_set.items():
            if k in feature_values:
                self.feature_set[k] = v
            else:
                self.feature_set[k].generate()

        self._model = model
        self._model.setup(**self.feature_set)

    def generate(self) -> None:
        '''
        Generate a new patient.

        This method calls `generate` method of every `feature`, and then sets
        up to `model` using the new values.
        '''
        for v in self.feature_set.values():
            v.generate()

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
