# -*- coding: utf-8 -*-
'''
Utils module for reinforcement learning
=======================================

This module provides different utilities used in the `reil` package.

Submodules
----------
exploration_strategies:
    A module that provides different exploration strategies for `agents`.

functions:
    Contains different useful functions.

Classes
-------
ActionGenerator:
    A class that accepts categorical and numerical components, and
    generates lists of actions as `ReilData` objects.

InstanceGenerator:
    Accepts any object derived from `ReilBase`, and generates instances.

MNKBoard:
    An m-by-n board in which k similar horizontal, vertical, or diagonal
    sequence is a win. Used in `subjects` such as `TicTacToe`.

WekaClusterer:
    A clustering class based on Weka's clustering capabilities (disabled)
'''

import reil.utils.exploration_strategies
import reil.utils.functions  # noqa: W0611

from .action_generator import (ActionGenerator,  # noqa: W0611
                               CategoricalComponent,
                               NumericalComponent)
from .instance_generator import InstanceGenerator  # noqa: W0611
from .mnkboard import MNKBoard  # noqa: W0611
