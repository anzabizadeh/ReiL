# -*- coding: utf-8 -*-
'''
QLearning class
===============

A Q-learning `agent`.


'''

from typing import Any, Optional, Tuple, Union

import numpy as np
from reil import agents, stateful
from reil.datatypes import FeatureArray
from reil.datatypes.buffers import Buffer
from reil.learners import Learner
from reil.utils.exploration_strategies import ExplorationStrategy


Feature_or_Tuple_of_Feature = Union[Tuple[FeatureArray, ...], FeatureArray]


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
            whether the `action` should be exploratory or not for a given
            `state` at a given `epoch`.

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
        super().__init__(learner=learner,
                         exploration_strategy=exploration_strategy,
                         **kwargs)

        self._method = method.lower()
        if self._method not in ('backward', 'forward'):
            self._logger.warning(
                f'method {method} is not acceptable. Should be '
                'either "forward" or "backward". Will use "backward".')
            self._method = 'backward'

        self._buffer = buffer
        self._buffer.setup(buffer_names=['X', 'Y'])

    @classmethod
    def _empty_instance(cls):
        class MockBuffer:
            def setup(self, **kwargs):
                pass

        return cls(None, MockBuffer(), None)  # type: ignore

    def _q(self,
           state: Feature_or_Tuple_of_Feature,
           action: Optional[Feature_or_Tuple_of_Feature] = None
           ) -> Tuple[float, ...]:
        '''
        Return the Q-value of `state` `action` pairs.

        Arguments
        ---------
        state:
            One state or a list of states for which Q-value is returned.

        action:
            One action or a list of actions for which Q-value is returned.
            If not supplied, `default_actions` will be used.

        Notes
        -----
        If one of state or action is one item, it will be broadcasted to
        match the size of the other one. If both are lists, the should match in
        size.


        :meta public:
        '''
        state_list = [state] if isinstance(state, FeatureArray) else state
        len_state = len(state_list)

        if action is None:
            action_list = self._default_actions
        else:
            action_list = ([action] if isinstance(action, FeatureArray)
                           else action)

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

        return self._learner.predict(X)

    def _max_q(self, state: Feature_or_Tuple_of_Feature) -> float:
        '''
        Return `max(Q)` of one state or a list of states.

        Arguments
        ---------
        state:
            One state or a list of states for which MAX(Q) is returned.


        :meta public:
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
        Use `history` to create the training set in the form of `X` and `y`
        vectors.

        Arguments
        ---------
        history:
            a `History` object from which the `agent` learns.

        Returns
        -------
        :
            a `TrainingData` object that contains `X` and 'y` vectors


        :meta public:
        '''
        if self._method == 'forward':
            for i in range(len(history)-1):
                state = history[i].state
                action = history[i].action
                reward = history[i].reward['reward'].value
                try:
                    max_q = self._max_q(history[i+1].state)  # type: ignore
                    new_q = reward + self._discount_factor*max_q
                except IndexError:
                    new_q = reward

                self._buffer.add(
                    {'X': state + action, 'Y': new_q})

        else:  # backward
            q_list = [0.0] * len(history)
            for i in range(len(history)-2, -1, -1):
                state = history[i].state
                action = history[i].action
                reward = history[i].reward['reward'].value
                q_list[i] = reward + self._discount_factor*q_list[i+1]

                self._buffer.add(
                    {'X': state + action, 'Y': q_list[i]})

        temp = self._buffer.pick()

        return temp['X'], temp['Y']

    def best_actions(self,
                     state: FeatureArray,
                     actions: Optional[Tuple[FeatureArray, ...]] = None
                     ) -> Tuple[FeatureArray, ...]:
        '''
        Find the best `action`s for the given `state`.

        Arguments
        ---------
        state:
            The state for which the action should be returned.

        actions:
            The set of possible actions to choose from.

        Returns
        -------
        :
            A list of best actions.
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
