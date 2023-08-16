# -*- coding: utf-8 -*-
'''
PPOAgent class
=========

A Proximal Policy Optimization Policy Gradient `agent`.
'''

from typing import Any

import numpy as np
import tensorflow as tf

from reil.agents.actor_critic_agent import A2CAgent
from reil.agents.agent import TrainingData
from reil.datatypes import History
from reil.datatypes.buffers.buffer import Buffer
from reil.datatypes.feature import FeatureSet
from reil.learners.ppo_learner import PPOLearner
from reil.utils.exploration_strategies import NoExploration
from reil.utils.metrics import HistogramMetric, MetricProtocol
from reil.utils.tf_utils import ActionRank, MeanMetric

ACLabelType = tuple[tuple[tuple[int, ...], ...], float]


class PPOAgent(A2CAgent):
    '''
    A Proximal Policy Optimization `agent`.
    '''

    def __init__(
            self,
            learner: PPOLearner,
            buffer: Buffer[FeatureSet, tuple[tuple[int, ...], float, float]],
            reward_clip: tuple[float | None, float | None] = (None, None),
            gae_lambda: float = 1.0,
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
        self._learner: PPOLearner
        super(A2CAgent, self).__init__(
            learner=learner, exploration_strategy=NoExploration(),
            variable_action_count=False,
            **kwargs)

        self._buffer = buffer
        self._buffer.setup(buffer_names=['state', 'y_r_a'])
        self._reward_clip = reward_clip
        self._gae_lambda = gae_lambda
        self._metrics: dict[str, MetricProtocol] = {
            'action_rank': ActionRank(),
            'advantage_mean': MeanMetric('advantage_mean', dtype=tf.float32),
            'advantage_h': HistogramMetric('advantage_h'),
            'rewards': MeanMetric('rewards', dtype=tf.float32),
            'rewards_h': HistogramMetric('rewards_h'),
        }
        if self._summary_writer:
            self._summary_writer.set_data_types({
                'last_layer_w': 'histogram',
                'advantage_h': 'histogram', 'rewards_h': 'histogram'})

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
        discount_factor = self._discount_factor
        active_history = self.get_active_history(history)

        # add zero to the end to have the correct length of `deltas`
        rewards = self.extract_reward(
            active_history, *self._reward_clip) + [0.0]
        dis_reward = self.discounted_cum_sum(rewards, discount_factor)

        state_list: tuple[FeatureSet, ...] = tuple(
            h.state for h in active_history)  # type: ignore
        # add zero to the end to have the correct length of `deltas`
        y, values = self._learner.predict(state_list)
        values = np.append(tf.reshape(values, -1), 0.0)
        action_indices = tuple(
            list((h.action_taken or h.action).index.values())  # type: ignore
            for h in active_history)
        deltas = (
            rewards[:-1] + discount_factor * values[1:]
            - values[:-1])
        advantage = self.discounted_cum_sum(
            deltas, discount_factor * self._gae_lambda)  # type: ignore

        self._buffer.add_iter({
            'state': h.state,
            'y_r_a': (
                tuple((h.action_taken or h.action).index.values()),  # type: ignore
                dis_r, a)
        } for h, dis_r, a in zip(active_history, dis_reward, advantage))

        temp = self._buffer.pick()

        self._update_metrics(
            rewards=rewards, dis_reward=dis_reward, state_list=state_list,
            y=y, values=values, action_indices=action_indices,
            deltas=deltas, advantage=advantage)

        return temp['state'], temp['y_r_a'], {}  # type: ignore

    def _update_metrics(self, **kwargs: Any) -> None:
        '''
        updates the metrics.

        Arguments
        ---------
        kwargs:
            keyword arguments.
        '''
        super()._update_metrics(**kwargs)
        action_indices = kwargs.get('action_indices')
        y = kwargs.get('y')
        advantage = kwargs.get('advantage')
        rewards = kwargs.get('rewards')

        if action_indices is not None and y is not None:
            action_lists = list(zip(*action_indices))
            for i, yi in enumerate(y):
                self._metrics['action_rank'].update_state(
                    tf.squeeze(action_lists[i]), yi)
        if advantage is not None:
            self._metrics['advantage_mean'].update_state(advantage)
            self._metrics['advantage_h'].update_state(advantage)
        if rewards is not None:
            self._metrics['rewards'].update_state(rewards[:-1])
            self._metrics['rewards_h'].update_state(rewards[:-1])
