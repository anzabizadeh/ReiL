# -*- coding: utf-8 -*-
'''
DQNAgent class
=================

A Q-learning agent with Neural Network Q-function approximator

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from typing import Any

from reil import agents, learners
from reil.utils import buffers, ExplorationStrategy


class DeepQLearning(agents.QLearning):
    '''
    A Q-learning agent.

    Constructor Arguments
    ---------------------
        * gamma: discount factor in TD equation. (Default = 1)
        * epsilon: exploration probability. (Default = 0)
        * default_actions: list of default actions.
        * lr_initial: learning rate for ANN. (Default = 1e-3)
        * hidden_layer_sizes: tuple containing hidden layer sizes.
        * input_length: size of the input vector. (Default = 1)
        buffer_size: DQN stores buffer_size observations and samples from it for training. (Default = 50)
        batch_size: the number of samples to choose randomly from the buffer for training. (Default = 10)
        * validation_split: proportion of sampled observations set for validation. (Default = 0.3)
        clear_buffer: whether to clear buffer after sampling (True: clear buffer, False: only discard old observations). (Default = False)

        Note: Although input_length has a default value, but it should be specified in object construction.
    Methods
    -------
        * act: return an action based on the given state.
        * learn: learn using either history or action, reward, and state.
    '''

    def __init__(self,
                 learner: learners.Dense,
                 buffer: buffers.VanillaExperienceReplay,
                 exploration_strategy: ExplorationStrategy,
                 method: str = 'backward',
                 **kwargs: Any):
        '''
        Initialize a Q-Learning agent with deep neural network Q-function approximator.
        '''

        super().__init__(learner=learner,
                         buffer=buffer,
                         exploration_strategy=exploration_strategy,
                         **kwargs)

        self._method = method.lower()
        if self._method not in ('backward', 'forward'):
            self._logger.warning(
                f'method {method} is not acceptable. Should be either "forward" or "backward". Will use "backward".')
            self._method = 'backward'

        self._buffer = buffer
        self._epoch = 0
