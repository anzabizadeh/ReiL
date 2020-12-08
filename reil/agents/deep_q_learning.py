# -*- coding: utf-8 -*-
'''
DeepQLearning class
===================

A Q-learning `agent` with a Neural Network Q-function approximator.
'''

from typing import Any

from reil import agents, learners
from reil.utils import buffers, exploration_strategies


class DeepQLearning(agents.QLearning):
    '''
    A Deep Q-learning `agent`.
    '''
    def __init__(self,
                 learner: learners.Dense,
                 buffer: buffers.VanillaExperienceReplay,
                 exploration_strategy: exploration_strategies.ExplorationStrategy,
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
            whether the `action` should be exploratory or not for a given `state` at
            a given `epoch`.

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
