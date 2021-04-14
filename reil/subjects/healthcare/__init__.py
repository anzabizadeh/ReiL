# -*- coding: utf-8 -*-
'''
healthcare subjects module for reinforcement learning
=====================================================

This module provides different healthcare subjects in reinforcement learning
context.

Classes
-------
Patient:
    The base class of all healthcare subject classes

WarfarinPatientRavvaz:
    A warfarin patient model with features and parameters of
    Ravvaz et al. 2016.

CancerModel:
    A 4-ordinary differential equation model of cancer.
    (uses legacy ValueSet instead of FeatureArray)

ConstrainedCancerModel:
    A constrained version of CancerModel.
    (uses legacy ValueSet instead of FeatureArray)

Warfarin:
    A `Subject` for warfarin that uses `WarfarinPatientRavvaz` and
    `healthcare.hamberg_pkpd`.
'''

# TODO: update CancerModel
# TODO: update ConstrainedCancerModel

from .patient import Patient  # noqa: W0611
from .warfarin_patient_ravvaz import WarfarinPatientRavvaz  # noqa: W0611
# from .cancer_model import CancerModel  # noqa: W0611
# from .constrained_cancer_model import ConstrainedCancerModel  # noqa: W0611
from .warfarin import Warfarin  # noqa: W0611
