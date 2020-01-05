# -*- coding: utf-8 -*-
'''
subjects module for reinforcement learning
==========================================

This module provides different subjects in reinforcement learning
context.

Classes
-------
    Subject: the super class of all subject classes
    MNKGame: a simple game consisting of an m-by-n board,
        each player should make a horizontal, vertical, or
        diagonal sequence of size k to win the game.
    FrozenLake: a frozen lake with cracks in it! (uses legacy ValueSet instead of RLData)
    WindyGridworld: a grid with displacement of agent (as if wind blows) (uses legacy ValueSet instead of RLData)

    CancerModel: a 4-ordinary differential equation model of cancer (uses legacy ValueSet instead of RLData)
    ConstrainedCancerModel: a constrained version of CancerModel (uses legacy ValueSet instead of RLData)
    WarfarinModel_v5: a PK/PD model for warfarin with extended state definition

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from .subject import Subject
from .mnkgame import MNKGame
from .windy_gridworld import WindyGridworld
from .cancer_model import CancerModel
from .constrained_cancer_model import ConstrainedCancerModel
from .warfarin_model_v5 import WarfarinModel_v5
from .iterable_subject import IterableSubject
from .warfarin_lookahead import WarfarinLookAhead
from .warfarin_model_fixed_intervals import WarfarinModelFixedInterval