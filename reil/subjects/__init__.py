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
    FrozenLake: a frozen lake with cracks in it! (uses legacy ValueSet instead of ReilData)
    WindyGridworld: a grid with displacement of agent (as if wind blows) (uses legacy ValueSet instead of ReilData)


'''

from .subject import Subject, SubjectType
from .mnkgame import MNKGame
from .tic_tac_toe import TicTacToe
# from .windy_gridworld import WindyGridworld
from .iterable_subject import IterableSubject
