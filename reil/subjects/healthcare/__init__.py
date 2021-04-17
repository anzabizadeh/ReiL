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

PatientWarfarinRavvaz:
    A warfarin patient model with features and parameters of
    Ravvaz et al. 2016.

Warfarin:
    A `Subject` for warfarin that uses `PatientWarfarinRavvaz` and
    `healthcare.hamberg_pkpd`.
'''

# TODO: update CancerModel
# TODO: update ConstrainedCancerModel

from .patient import Patient  # noqa: W0611
from .patient_warfarin_ravvaz import PatientWarfarinRavvaz  # noqa: W0611
from .health_subject import HealthSubject  # noqa: W0611
from .warfarin import Warfarin  # noqa: W0611
