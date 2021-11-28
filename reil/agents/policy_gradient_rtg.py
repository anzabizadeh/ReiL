# -*- coding: utf-8 -*-
'''
PolicyGradientRTG class
=======================

A reward-to-go Policy Gradient `agent`.
'''

from typing import Any, List, Tuple, Union

import numpy as np
from reil.agents.agent import Agent, TrainingData
from reil.datatypes import History
from reil.datatypes.buffers import Buffer
from reil.datatypes.feature import FeatureArray
from reil.learners import Learner
from reil.utils.exploration_strategies import (ConstantEpsilonGreedy,
                                               ExplorationStrategy)

Feature_or_Tuple_of_Feature = Union[Tuple[FeatureArray, ...], FeatureArray]


class PolicyGradientRTG(Agent[float]):
    '''
    A reward to go Policy Gradient `agent`.
    '''

    def __init__(
            self,
            learner: Learner[float],
            buffer: Buffer[FeatureArray, float],
            exploration_strategy: ExplorationStrategy,
            **kwargs: Any):
        '''
        Arguments
        ---------
        learner:
            the `Learner` object that does the learning.

        exploration_strategy:
            an `ExplorationStrategy` object that determines
            whether the `action` should be exploratory or not for a given
            `state` at a given `iteration`.

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
        super().__init__(
            learner=learner, exploration_strategy=exploration_strategy,
            **kwargs)

        # if method == 'forward' and not kwargs.get('default_actions'):
        #     raise ValueError(
        #         'forward method requires `default_actions` to be non-empty.')

        self._buffer = buffer
        self._buffer.setup(buffer_names=['X', 'Y'])

    @classmethod
    def _empty_instance(cls):  # type: ignore
        return cls(
            Learner._empty_instance(), Buffer(), ConstantEpsilonGreedy())

    @staticmethod
    def _reward_to_go(rews: List[float]):
        n = len(rews)
        rtgs = np.zeros_like(rews)  # type: ignore
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    def _prepare_training(
            self, history: History) -> TrainingData[float]:
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
        reward: float

        discount_factor = self._discount_factor

        for h in history[:-1]:
            if h.state is None or h.action is None:
                raise ValueError(f'state and action cannot be None.\n{h}')

        if history[-1].action is None:
            active_history = history[:-1]
        else:
            active_history = history

        rtg_list = [0.0] * len(active_history)
        for i in range(len(active_history)-2, -1, -1):
            reward = active_history[i].reward or 0.0
            rtg_list[i] = reward + discount_factor * rtg_list[i+1]

        rtg_list = np.array(rtg_list)  # type: ignore
        rtg_list -= np.mean(rtg_list)  # type: ignore
        rtg_list /= np.std(rtg_list)  # type: ignore

        for i in range(len(active_history)-2, -1, -1):
            self._buffer.add({
                'X': active_history[i].state,  # type: ignore
                'Y': rtg_list[i]})

        temp = self._buffer.pick()

        return temp['X'], temp['Y']  # type: ignore

    def best_actions(
            self,
            state: FeatureArray,
            actions: Tuple[FeatureArray, ...]) -> Tuple[FeatureArray, ...]:
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
        action_index = int(self._learner.predict((state,))[0])
        return (actions[action_index],)

    def reset(self) -> None:
        '''Resets the agent at the end of a learning iteration.'''
        super().reset()
        self._buffer.reset()
