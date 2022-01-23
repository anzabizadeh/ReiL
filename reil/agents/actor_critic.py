# -*- coding: utf-8 -*-
'''
ActorCritic class
=================

An Actor-Critic Policy Gradient `agent`.
'''

from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from reil.agents.agent import Agent, TrainingData
from reil.datatypes import History
from reil.datatypes.feature import FeatureGeneratorType, FeatureSet
from reil.learners import Learner
from reil.utils.exploration_strategies import NoExploration


ACLabelType = Tuple[Tuple[Tuple[int, ...], ...], float]


class ActorCritic(Agent[FeatureSet, ACLabelType]):
    '''
    A reward to go Policy Gradient `agent`.
    '''

    def __init__(
            self,
            learner: Learner[FeatureSet, ACLabelType],
            **kwargs: Any):
        '''
        Arguments
        ---------
        learner:
            the `Learner` object that does the learning.

        discount_factor:
            by what factor should future rewards be discounted?

        training_mode:
            whether the agent is in training mode or not.

        tie_breaker:
            how to choose the `action` if more than one is candidate
            to be chosen.
        '''
        super().__init__(
            learner=learner, exploration_strategy=NoExploration(),
            variable_action_count=False,
            **kwargs)

    @classmethod
    def _empty_instance(cls):  # type: ignore
        return cls(Learner._empty_instance())

    def _prepare_training(
            self, history: History) -> TrainingData[FeatureSet, int]:
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
            if h.state is None or (
                    h.action is None and h.action_taken is None):
                raise ValueError(f'state and action cannot be None.\n{h}')

        if history[-1].action_taken is None and history[-1].action is None:
            active_history = history[:-1]
        else:
            active_history = history

        rtg_list = [0.0] * (len(active_history) + 1)
        for i in range(len(active_history)-1, -1, -1):
            reward = active_history[i].reward or 0.0
            rtg_list[i] = reward + discount_factor * rtg_list[i+1]

        rtg_list = np.array(rtg_list[:-1])  # type: ignore

        return (
            [a.state for a in active_history],  # type: ignore
            [(tuple(
                (a.action_taken or a.action).index.values()),  # type: ignore
              g)
             for a, g in zip(active_history, rtg_list)],
            {})

    def act(self,
            state: FeatureSet,
            subject_id: int,
            actions: FeatureGeneratorType,
            iteration: int = 0) -> FeatureSet:
        '''
        Return an action based on the given state.

        Arguments
        ---------
        state:
            the state for which the action should be returned.

        subject_id:
            the ID of the `subject` on which action should occur.

        actions:
            the set of possible actions to choose from.

        iteration:
            the iteration in which the agent is acting.

        Raises
        ------
        ValueError
            Subject with `subject_id` not found.

        Returns
        -------
        :
            the action
        '''
        if subject_id not in self._entity_list:
            raise ValueError(f'Subject with ID={subject_id} not found.')

        logits = self._learner.predict((state,))[0]
        if self._training_trigger == 'none':
            action_index = [int(np.argmax(lo)) for lo in logits]
        else:
            action_index = [
                int(tf.random.categorical(logits=lo, num_samples=1))
                for lo in logits]

        action: FeatureSet = actions.send(f'lookup {action_index}')

        return action

    def reset(self) -> None:
        '''Resets the agent at the end of a learning iteration.'''
        super().reset()
