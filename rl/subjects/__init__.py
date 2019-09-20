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
    WindyGridworld: a grid with displacement of agent (as if wind blows) 
    Snake: single player snake game

    CancerModel: a 4-ordinary differential equation model of cancer
    ConstrainedCancerModel: a constrained version of CancerModel
    WarfarinModel: a PK/PD model for warfarin
    WarfarinModel_v4: a PK/PD model for warfarin with extended state definition

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from .subject import Subject
from .mnkgame import MNKGame
from .windy_gridworld import WindyGridworld
# from .snake import Snake
# from .risk import Risk
from .cancer_model import CancerModel
from .constrained_cancer_model import ConstrainedCancerModel
from .warfarin_model_v3 import WarfarinModel
from .warfarin_model_v4 import WarfarinModel_v4
from .warfarin_model_v5 import WarfarinModel_v5
