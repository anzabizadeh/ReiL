# -*- coding: utf-8 -*-
'''
healthcare subjects module for reinforcement learning
=====================================================

This module provides different healthcare subjects in reinforcement learning
context.

Classes
-------
    Patient: the base class of all healthcare subject classes
    WarfarinPatientRavvaz: a warfarin patient model with features and parameters
        of Ravvaz et al. 2016.

    CancerModel: a 4-ordinary differential equation model of cancer (uses legacy ValueSet instead of ReilData)
    ConstrainedCancerModel: a constrained version of CancerModel (uses legacy ValueSet instead of ReilData)
    Warfarin: a PK/PD model for warfarin with extended state definition

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

# TODO: update CancerModel
# TODO: update ConstrainedCancerModel

from .patient import Patient
from .warfarin_patient_ravvaz import WarfarinPatientRavvaz
# from .cancer_model import CancerModel
# from .constrained_cancer_model import ConstrainedCancerModel
from .warfarin import Warfarin
