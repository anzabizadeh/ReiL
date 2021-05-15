# -*- coding: utf-8 -*-
'''
DeepQLearning class
===================

A Q-learning `agent` with a Neural Network Q-function approximator.
'''

from typing import Any

from reil.agents import QLearning
from reil.datatypes.buffers import VanillaExperienceReplay
from reil.learners import Dense
from reil.utils.exploration_strategies import ExplorationStrategy


class DeepQLearning(QLearning):
    '''
    A Deep Q-learning `agent`.
    '''

    def __init__(self,
                 learner: Dense,
                 buffer: VanillaExperienceReplay,
                 exploration_strategy: ExplorationStrategy,
                 method: str = 'backward',
                 **kwargs: Any):
        '''
        Arguments
        ---------
        learner:
            The `Learner` of type `Dense` that does the learning.

        buffer:
            A buffer that collects observations for training. Some
            variation of `ExperienceReply` is recommended.

        exploration_strategy:
            An `ExplorationStrategy` object that determines
            whether the `action` should be exploratory or not for a given
            `state` at a given `iteration`.

        method:
            Either 'forward' or 'backward' Q-learning.

        kwargs:
            Keyword arguments to be passed on to the parent class.
        '''
        super().__init__(learner=learner,
                         buffer=buffer,
                         exploration_strategy=exploration_strategy,
                         method=method,
                         **kwargs)
