# -*- coding: utf-8 -*-
'''
subjects module for reinforcement learning
==========================================

This module provides different subjects in reinforcement learning
context.

Classes
-------
    MNKBoard: an m-by-n board in which k similar horizontal, vertical, or
        diagonal sequence is a win.
    Patient: the Warfarin PK/PD model based on Hamberg et al (2006)
    WekaClusterer: a clustering class based on Weka's clustering capabilities

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from .action_generator import ActionGenerator
from .mnkboard import MNKBoard
from .buffers import Buffer, EndlessBuffer, CircularBuffer, VanillaExperienceReplay
from .exploration_strategies import ExplorationStrategy, ConstantEpsilonGreedy, VariableEpsilonGreedy
from .feature import Feature
from .functions import (get_argument, random_categorical, random_truncated_normal,
    random_truncated_lnorm, random_uniform)

import warnings
import pip

try:
    installed_pkgs = [pkg.key for pkg in pip.get_installed_distributions()]

    if 'weka' in installed_pkgs:
        from .weka_clustering import WekaClusterer
    else:
        import warnings
        warnings.warn('Could not find dependencies of "WekaClusterer" ("weka"). Skipped installing the module.')
except AttributeError:
    warnings.warn('Could not use pip to check the availability of dependencies of "WekaClusterer" ("weka"). Skipped installing the module.')
