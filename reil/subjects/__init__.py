# -*- coding: utf-8 -*-
'''
subjects module for reinforcement learning
==========================================

This module provides different subjects in reinforcement learning
context.

Classes
-------
Subject:
    The base class of all `subject` classes.

MNKGame:
    A simple game consisting of an m-by-n board.
    Each player should make a horizontal, vertical, or
    diagonal sequence of size k to win the game.

TicTacToe:
    Standard Tic-Tac-Toe game.

FrozenLake:
    (disabled)
    A frozen lake with cracks in it! (uses legacy ValueSet instead of ReilData)

WindyGridworld:
    (disabled)
    A grid with displacement of `agent` (as if wind blows)
    (uses legacy ValueSet instead of ReilData)
'''

from .subject import Subject, SubjectType  # noqa: W0611
from .mnkgame import MNKGame  # noqa: W0611
from .tic_tac_toe import TicTacToe  # noqa: W0611
# from .windy_gridworld import WindyGridworld  # noqa: W0611
# from .frozen_lake import FrozenLake  # noqa: W0611
