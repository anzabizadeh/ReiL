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
    Snake: single player snake game

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from .subject import Subject
from .mnkgame import MNKGame
from .windy_gridworld import WindyGridworld
from .risk import Risk
from .cancer_model import CancerModel
from .constrained_cancer_model import ConstrainedCancerModel
# from .snake import Snake