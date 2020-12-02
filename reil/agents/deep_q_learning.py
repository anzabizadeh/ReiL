# -*- coding: utf-8 -*-
'''
DeepQLearning class
===================

A Q-learning agent with Neural Network Q-function approximator

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from typing import Any

from reil import agents, learners
from reil.utils import buffers, exploration_strategies


class DeepQLearning(agents.QLearning):
    '''
    A Deep Q-learning agent.
    '''

    def __init__(self,
                 learner: learners.Dense,
                 buffer: buffers.VanillaExperienceReplay,
                 exploration_strategy: exploration_strategies.ExplorationStrategy,
                 method: str = 'backward',
                 **kwargs: Any):
        '''
        Initialize a Q-Learning agent with deep neural network Q-function approximator.

        ### Arguments
        learner: the `Learner` of type `Dense` that does the learning.

        buffer: a `Buffer` object that collects observations for training. Some
        variation of `ExperienceReply` is recommended.

        exploration_strategy: an `ExplorationStrategy` object that determines
        whether the `action` should be exploratory or not for a given `state` at
        a given `epoch`.

        method: either 'forward' or 'backward' Q-learning.
        '''
        super().__init__(learner=learner,
                         buffer=buffer,
                         exploration_strategy=exploration_strategy,
                         method=method,
                         **kwargs)
