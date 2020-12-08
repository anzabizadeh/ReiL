# -*- coding: utf-8 -*-
'''
exploration_strategies module
=============================

Contains classes that mimics exploration strategies in reinforcement learning.

Classes
-------
ExplorationStrategy: the base class for all exploration strategies.

ConstantEpsilonGreedy: an epsilon greedy object with constant epsilon

VariableEpsilonGreedy: an epsilon greedy object that accepts a uni-variate
function to determine epsilon.


'''
import random
from typing import Callable
import warnings

from reil import reilbase


class ExplorationStrategy(reilbase.ReilBase):
    '''
    The base class for all exploration strategies.

    Methods
-----------
    explore: returns True if the `agent` needs to explore.
    '''

    def __init__(self) -> None:
        '''
        initilizes the instance.
        '''
        pass

    def explore(self, epoch: int = 0) -> bool:
        '''
        Returns True if the `agent` needs to explore.

        Arguments
-----------
        epoch: the current epoch number.
        '''
        return True


class ConstantEpsilonGreedy(ExplorationStrategy):
    '''
    An epsilon greedy object with constant epsilon.

    Methods
-----------
    explore: returns True if the `agent` needs to explore.
    '''

    def __init__(self, epsilon: float) -> None:
        '''
        initilizes the instance.

        Arguments
-----------
        epsilon: the value of epsilon

        Note: if epsilon is not in the range of [0, 1], a warning is being issued,
        but it does not raise an exception.
        '''
        if not (0.0 <= epsilon <= 1.0):
            warnings.warn('epsilon is not in the range of [0, 1].')
        self._epsilon = epsilon

    def explore(self, epoch: int = 0) -> bool:
        '''
        Returns True if a randomly generated number is less than `epsilon`.

        Arguments
-----------
        epoch: the current epoch number.
        '''
        return random.random() < self._epsilon


class VariableEpsilonGreedy(ExplorationStrategy):
    '''
    An epsilon greedy object with constant epsilon.

    Methods
-----------
    explore: returns True if the `agent` needs to explore.
    '''

    def __init__(self, epsilon: Callable[[int], float]) -> None:
        '''
        initilizes the instance.

        Arguments
-----------
        epsilon: a uni-variate function that computes `epsilon` based on `epoch`.

        Raises `TypeError` if `epsilon` is not callable.
        '''
        if not callable(epsilon):
            raise TypeError('epsilon should be callable. For constant epsilon, '
                            'use `ConstantEpsilonGreedy` class.')
        self._epsilon = epsilon

    def explore(self, epoch: int) -> bool:
        '''
        Returns True if a randomly generated number is less than `epsilon`.

        Arguments
-----------
        epoch: the current epoch number.
        '''
        return random.random() < self._epsilon(epoch)
