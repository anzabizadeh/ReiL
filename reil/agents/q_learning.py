# -*- coding: utf-8 -*-
'''
DQNAgent class
=================

A Q-learning agent with Neural Network Q-function approximator

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from typing import Any, List, Optional, Tuple, Union

import numpy as np  # type: ignore
from reil import agents, rlbase, rldata
from reil.utils import ExplorationStrategy
from reil.learners.learner import Learner
from reil.utils.buffers import Buffer


class QLearning(agents.Agent):
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
                 learner: Learner,
                 buffer: Buffer,
                 exploration_strategy: ExplorationStrategy,
                 method: str = 'backward',
                 **kwargs: Any):
        '''
        Initialize a Q-Learning agent with deep neural network Q-function approximator.
        '''

        super().__init__(learner=learner,
                         exploration_strategy=exploration_strategy,
                         **kwargs)

        self._method = method.lower()
        if self._method not in ('backward', 'forward'):
            self._logger.warning(
                f'method {method} is not acceptable. Should be either "forward" or "backward". Will use "backward".')
            self._method = 'backward'

        self._buffer = buffer
        self._buffer.setup(buffer_names=['X', 'Y'])
        self._epoch = 0

    def _q(self, state: Union[List[rldata.RLData], rldata.RLData], action: Optional[Union[List[rldata.RLData], rldata.RLData]] = None) -> List[float]:
        '''
        Return the Q-value of a state action pair.

        Arguments
        ---------
            state: the state for which Q-value is returned.
            action: the action for which Q-value is returned. 'None' uses default_actions.
        '''
        state_list = [state] if isinstance(state, rldata.RLData) else state
        len_state = len(state_list)

        if action is None:
            action_list = self._default_actions
        else:
            action_list = [action] if isinstance(action, rldata.RLData) else action

        len_action = len(action_list)

        if len_state == len_action:
            X = [state_list[i] + action_list[i]
                 for i in range(len_state)]
        elif len_action == 1:
            X = [state_list[i] + action_list[0]
                 for i in range(len_state)]
        elif len_state == 1:
            X = [state_list[0] + action_list[i]
                 for i in range(len_action)]
        else:
            raise ValueError(
                'State and action should be of the same size or at least one should be of size one.')

        return self._learner.predict(X)

    def _max_q(self, state: Union[List[rldata.RLData], rldata.RLData]) -> float:
        '''
        Return MAX(Q) of a state.

        Arguments
        ---------
            state: the state for which MAX(Q) is returned.
        '''
        try:
            q_values = self._q(state)
            max_q = np.max(q_values)
        except ValueError:
            max_q = 0

        return max_q

    def _prepare_training_from_history(self, history: rlbase.History) -> Tuple[List[rldata.RLData], List[float]]:
        '''
        Prepare training set based on history.

        Arguments:
            history: a list consisting of state, action, reward of an episode.

        Note: Learning actually occurs every batch_size iterations.
        '''
        if self._method == 'forward':
            for i in range(len(history)-1):
                state = history[i]['state']
                action = history[i]['action']
                reward: float = history[i]['reward'][0].value
                try:
                    max_q = self._max_q(history[i+1]['state'])
                    new_q = reward + self._discount_factor*max_q
                except IndexError:
                    new_q = reward

                self._buffer.add(
                    {'X': state + action, 'Y': new_q})

        else:  # backward
            q_list = [0.0] * len(history)
            for i in range(len(history)-2, -1, -1):
                state = history[i]['state']
                action = history[i]['action']
                reward: float = history[i]['reward'][0].value
                try:
                    new_q = reward + self._discount_factor*q_list[i+1]
                except IndexError:
                    new_q = reward
                q_list[i] = new_q

                self._buffer.add(
                    {'X': state + action, 'Y': new_q})
 
        temp = self._buffer.pick()

        return temp['X'], temp['Y']

    def best_actions(self,
                     state: rldata.RLData,
                     actions: Optional[List[rldata.RLData]] = None) -> Tuple[rldata.RLData, ...]:
        '''
        return a list of the best actions for a given state.

        Arguments
        ---------
            state: the state for which an action is chosen.
            actions: a set of possible actions. If not provided, default actions are used.
        '''
        q_values = self._q(state, None if actions ==
                           self._default_actions else actions)  # None is used to avoid redundant normalization of default_actions
        max_q = np.max(q_values)
        result = tuple(actions[i] for i in np.nonzero(q_values == max_q)[0])

        return result

    def reset(self) -> None:
        super().reset()
        self._buffer.reset()
