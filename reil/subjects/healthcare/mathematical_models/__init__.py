# -*- coding: utf-8 -*-
'''
healthcare mathematical models module
=====================================

This module provides mathematical models, e.g. PK/PDs, cancer PDEs, etc.

Classes
-------
    HealthMathModel: the base class for healthcare math models.
    HambergPKPD: Hamberg's warfarin PK/PD model.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from .health_math_model import HealthMathModel
from .hamberg_pkpd import HambergPKPD
