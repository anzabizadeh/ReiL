# -*- coding: utf-8 -*-
'''
Utils module for reinforcement learning
=======================================

This module provides different utilities used in the `reil` package.

Submodules
----------
buffers: a module that contains different types of `Buffers` used in `reil` package.

exploration_strategies: a module that provides different exploration strategies
for `agents`.

functions: contains different useful functions.

Classes
-------
ActionGenerator: a class that accepts categorical and numerical components, and
generates lists of actions as `ReilData` objects.

InstanceGenerator: accepts any object derived from `ReilBase`, and generates
instances.

MNKBoard: an m-by-n board in which k similar horizontal, vertical, or diagonal
sequence is a win. Used in `subjects` such as `TicTacToe`.

WekaClusterer: a clustering class based on Weka's clustering capabilities (disabled)

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from .action_generator import ActionGenerator, CategoricalComponent, NumericalComponent
from .instance_generator import InstanceGenerator
from .mnkboard import MNKBoard

import reil.utils.buffers
import reil.utils.exploration_strategies
import reil.utils.functions


# import warnings
# import pip

# try:
#     installed_pkgs = [pkg.key for pkg in pip.get_installed_distributions()]

#     if 'weka' in installed_pkgs:
#         from .weka_clustering import WekaClusterer
#     else:
#         import warnings
#         warnings.warn('Could not find dependencies of "WekaClusterer" '
#           '("weka"). Skipped installing the module.')
# except AttributeError:
#     warnings.warn('Could not use pip to check the availability of'
#       ' dependencies of "WekaClusterer" ("weka"). Skipped installing the module.')
