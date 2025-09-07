# -*- coding: utf-8 -*-
'''
A2CAgent class
=========

An Actor-Critic Policy Gradient `agent`.
'''

from typing import Any, Self

import numpy as np
import tensorflow as tf

from reil.agents.agent import Agent, TrainingData
from reil.datatypes import History
from reil.datatypes.buffers.buffer import Buffer
from reil.datatypes.feature import FeatureGeneratorType, FeatureSet
from reil.learners.learner import Learner, LearnerProtocol
from reil.utils.exploration_strategies import NoExploration

ACLabelType = tuple[tuple[int | None, ...], float]


class A2CAgent(Agent[FeatureSet, ACLabelType]):
    '''
    An actor critic (Policy Gradient) `agent`.
    '''

    def __init__(
            self,
            learner: LearnerProtocol[FeatureSet, ACLabelType],
            buffer: Buffer[FeatureSet, ACLabelType],
            reward_clip: tuple[float | None, float | None] = (None, None),
            **kwargs: Any):
        '''
        Arguments
        ---------
        learner:
            the `Learner` object that does the learning.

        buffer:
            a buffer to store observations before each training pass.

        reward_clip:
            a tuple to clip the reward. `None` means no clipping for that side.

        **kwargs:
            additional keyword arguments to pass to the `Agent` constructor.
        '''
        super().__init__(
            learner=learner, exploration_strategy=NoExploration(),
            variable_action_count=False,
            **kwargs)

        self._buffer = buffer
        self._buffer.setup(buffer_names=['state', 'y_and_g'])
        self._reward_clip = reward_clip

    @classmethod
    def _empty_instance(cls) -> Self:
        '''
        Returns
        -------
        :
            an empty `A2CAgent` instance.
        '''
        # Use the public interface for creating an empty learner and specify types
        empty_learner: Learner[FeatureSet, ACLabelType] = Learner._empty_instance()  # type: ignore
        empty_buffer: Buffer[FeatureSet, ACLabelType] = Buffer()

        return cls(empty_learner, empty_buffer)

    def _prepare_training(
            self, history: History) -> TrainingData[FeatureSet, ACLabelType]:
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
        discount_factor = self._discount_factor
        active_history = self.get_active_history(history)

        rewards = self.extract_reward(active_history, *self._reward_clip)
        disc_reward = self.discounted_cum_sum(rewards, discount_factor)

        self._buffer.add_iter(
            {
                'state': h.state,
                'y_and_g': (
                    tuple((h.action_taken or h.action).index.values()),
                    r)
            } for h, r in zip(active_history, disc_reward))

        temp = self._buffer.pick()

        return temp['state'], temp['y_and_g'], {}

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

        training_mode = self._training_trigger != 'none'
        logits = self._learner.predict((state,), training=training_mode)[0]
        if training_mode:
            action_index = [
                int(tf.random.categorical(logits=lo, num_samples=1))
                for lo in logits]
        else:
            action_index = [int(np.argmax(lo)) for lo in logits]

        if len(action_index) == 1:
            action_index = action_index[0]
        action: FeatureSet = actions.send(f'lookup {action_index}')

        return action

    def reset(self) -> None:
        '''Resets the agent at the end of a learning iteration.'''
        super().reset()
