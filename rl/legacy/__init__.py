# -*- coding: utf-8 -*-
'''
legacy classes for reinforcement learning
========================================

This module provides classes that are no longer supported.

Classes
-------
    ANNAgent
    ValueSet
    WarfarinModel
    WarfarinModel_v2
    WarfarinModel_v3
    WarfarinModel_v4
    WarfarinLookAhead
    WarfarinModelFixedInterval

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

# import warnings
# import pip

# try:
#     installed_pkgs = [pkg.key for pkg in pip.get_installed_distributions()]
#     if 'rpy2' in installed_pkgs:
#         from .warfarin_model import WarfarinModel
#     else:
#         import warnings
#         warnings.warn('Could not find dependencies of "WarfarinModel" ("rpy2"). Skipped installing the module.')
# except AttributeError:
#     warnings.warn('Could not use pip to check the availability of dependencies of "WarfarinModel" ("rpy2"). Skipped installing the module.')

from .ann_agent import ANNAgent
from .valueset import ValueSet
from .legacyrldata import LegacyRLData

# from .warfarin_model_v2 import WarfarinModel_v2
# from .warfarin_model_v3 import WarfarinModel_v3
# from .warfarin_model_v4 import WarfarinModel_v4
from .warfarin_lookahead import WarfarinLookAhead
from .warfarin_model_fixed_intervals import WarfarinModelFixedInterval
