# -*- coding: utf-8 -*-
'''
ActorCritic class
=================

An Actor-Critic Policy Gradient `agent`.
'''

import random
from typing import Any, Optional, Tuple

import numpy as np
import tensorflow as tf
from reil.agents.agent import Agent, TrainingData
from reil.datatypes import History
from reil.datatypes.dataclasses import Index_FeatureArray
from reil.datatypes.feature import FeatureArray
from reil.learners import Learner
from reil.utils.exploration_strategies import NoExploration


ACLabelType = Tuple[Tuple[int, ...], float]


class ActorCritic(Agent[ACLabelType]):
    '''
    A reward to go Policy Gradient `agent`.
    '''

    def __init__(
            self,
            learner: Learner[ACLabelType],
            **kwargs: Any):
        '''
        Arguments
        ---------
        learner:
            the `Learner` object that does the learning.

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
            learner=learner, exploration_strategy=NoExploration(),
            **kwargs)

    @classmethod
    def _empty_instance(cls):  # type: ignore
        return cls(Learner._empty_instance())

    def _prepare_training(
            self, history: History) -> TrainingData[int]:
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
            [((a.action_taken or a.action).index, g)  # type: ignore
             for a, g in zip(active_history, rtg_list)],
            {})

    def act(self,
            state: FeatureArray,
            subject_id: int,
            actions: Optional[Tuple[FeatureArray, ...]] = None,
            iteration: int = 0) -> Index_FeatureArray:
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

        possible_actions = actions or self._default_actions

        if self._training_trigger == 'none':
            logits = self._learner.predict((state,))[0]
            action_index = int(np.argmax(logits))  # type: ignore

            i_action = Index_FeatureArray(
                action_index, possible_actions[action_index])
        else:
            i_action = self.best_actions(state, possible_actions)[0]

        return i_action

    def best_actions(
            self,
            state: FeatureArray,
            actions: Tuple[FeatureArray, ...]
    ) -> Tuple[Index_FeatureArray, ...]:
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
        logits = self._learner.predict((state,))[0]
        action_index = int(tf.random.categorical(  # type: ignore
            logits=logits, num_samples=1))

        try:
            action = Index_FeatureArray(action_index, actions[action_index])
        except IndexError:
            action_index: int = random.randrange(len(actions))
            action = Index_FeatureArray(action_index, actions[action_index])

        return (action,)

    def reset(self) -> None:
        '''Resets the agent at the end of a learning iteration.'''
        super().reset()
