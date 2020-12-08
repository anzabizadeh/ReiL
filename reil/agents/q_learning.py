# -*- coding: utf-8 -*-
'''
QLearning class
===============

A Q-learning `agent`.


'''

from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np  # type: ignore
from reil import agents, stateful
from reil.datatypes.reildata import ReilData
from reil.learners.learner import Learner
from reil.utils.buffers import Buffer
from reil.utils.exploration_strategies import ExplorationStrategy


class QLearning(agents.Agent):
    '''
    A Q-learning `agent`.
    '''
    def __init__(self,
                 learner: Learner[float],
                 buffer: Buffer,
                 exploration_strategy: ExplorationStrategy,
                 method: str = 'backward',
                 **kwargs: Any):
        '''
        Arguments
        ---------
        learner:
            the `Learner` object that does the learning.

        exploration_strategy:
            an `ExplorationStrategy` object that determines
            whether the `action` should be exploratory or not for a given `state`
            at a given `epoch`.

        discount_factor:
            by what factor should future rewards be discounted?

        default_actions:
            a tuple of default actions.

        training_mode:
            whether the agent is in training mode or not.

        tie_breaker:
            how to choose the `action` if more than one is candidate
            to be chosen.
        '''
        super().__init__(learner=cast(Learner[float], learner),
                         exploration_strategy=exploration_strategy,
                         **kwargs)

        self._method = method.lower()
        if self._method not in ('backward', 'forward'):
            self._logger.warning(
                f'method {method} is not acceptable. '
                'Should be either "forward" or "backward". Will use "backward".')
            self._method = 'backward'

        self._buffer = buffer
        self._buffer.setup(buffer_names=['X', 'Y'])

    def _q(self,
           state: Union[Tuple[ReilData, ...], ReilData],
           action: Optional[Union[Tuple[ReilData, ...], ReilData]] = None) -> Tuple[float, ...]:
        '''
        Returns the Q-value of `state` `action` pairs.

        Arguments
-----------
        state: one state or a list of states for which Q-value is returned.
        action: one action or a list of actions for which Q-value is returned. 
        If not supplied, `default_actions` will be used.

        Note: if one of state or action is one item, it will be broadcasted to
        match the size of the other one. If both are lists, the should match in
        size.
        '''
        state_list = [state] if isinstance(state, ReilData) else state
        len_state = len(state_list)

        if action is None:
            action_list = self._default_actions
        else:
            action_list = [action] if isinstance(action, ReilData) else action

        len_action = len(action_list)

        if len_state == len_action:
            X = tuple(state_list[i] + action_list[i]
                      for i in range(len_state))
        elif len_action == 1:
            X = tuple(state_list[i] + action_list[0]
                      for i in range(len_state))
        elif len_state == 1:
            X = tuple(state_list[0] + action_list[i]
                      for i in range(len_action))
        else:
            raise ValueError(
                'State and action should be of the same size'
                ' or at least one should be of size one.')

        return cast(Tuple[float, ...], self._learner.predict(X))

    def _max_q(self, state: Union[Tuple[ReilData, ...], ReilData]) -> float:
        '''
        Returns `max(Q)` of one state or a list of states.

        Arguments
-----------
        state: one state or a list of states for which MAX(Q) is returned.
        '''
        try:
            q_values = self._q(state)
            max_q = np.max(q_values)
        except ValueError:
            max_q = 0

        return max_q

    def _prepare_training(self,
        history: stateful.History) -> agents.TrainingData:
        '''
        Prepares training set based on history.

        Arguments
-----------
        history: a `History` consisting of state, action, reward of one sample path.

        Note: Learning actually occurs every batch_size iterations.
        '''
        if self._method == 'forward':
            for i in range(len(history)-1):
                state = cast(ReilData, history[i].state)
                action = cast(ReilData, history[i].action)
                reward = cast(float, history[i].reward[0].value)
                try:
                    max_q = self._max_q(cast(ReilData, history[i+1].state))
                    new_q = reward + self._discount_factor*max_q
                except IndexError:
                    new_q = reward

                self._buffer.add(
                    {'X': state + action, 'Y': new_q})

        else:  # backward
            q_list = [0.0] * len(history)
            for i in range(len(history)-2, -1, -1):
                state = cast(ReilData, history[i].state)
                action = cast(ReilData, history[i].action)
                reward = cast(float, history[i].reward[0].value)
                q_list[i] = reward + self._discount_factor*q_list[i+1]

                self._buffer.add(
                    {'X': state + action, 'Y': q_list[i]})
 
        temp = self._buffer.pick()

        return temp['X'], temp['Y']

    def best_actions(self,
                     state: ReilData,
                     actions: Optional[Tuple[ReilData, ...]] = None) -> Tuple[ReilData, ...]:
        '''
        Returns a tuple of best `action`s based on the given `state`.

        Arguments
-----------
        state: the state for which the action should be returned.

        actions: the set of possible actions to choose from.
        '''
        # None is used to avoid redundant normalization of default_actions
        q_values = self._q(state, None if actions ==
                           self._default_actions else actions)
        max_q = np.max(q_values)
        result = tuple(actions[i] for i in np.nonzero(q_values == max_q)[0])

        return result

    def reset(self) -> None:
        '''Resets the agent at the end of a learning epoch.'''
        super().reset()
        self._buffer.reset()
