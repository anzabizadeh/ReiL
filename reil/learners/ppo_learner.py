# -*- coding: utf-8 -*-
'''
PPOLearner class
================

'''

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers.schedules as k_sch
from reil.datatypes.feature import FeatureSet
from reil.learners.learner import Learner
from reil.utils.tf_utils import TF2UtilsMixin
from tensorflow import keras

ACLabelType = Tuple[Tuple[Tuple[int, ...], ...], float]

eps = np.finfo(np.float32).eps.item()


@keras.utils.register_keras_serializable(
    package='reil.learners.ppo_learner')
class PPOModel(TF2UtilsMixin):
    def __init__(
            self,
            input_shape: Tuple[int, ...],
            output_lengths: Tuple[int, ...],
            actor_learning_rate: Union[
                float, k_sch.LearningRateSchedule],
            critic_learning_rate: Union[
                float, k_sch.LearningRateSchedule],
            actor_layer_sizes: Tuple[int, ...],
            critic_layer_sizes: Tuple[int, ...],
            actor_train_iterations: int,
            critic_train_iterations: int,
            GAE_lambda: float,
            target_kl: float,
            clip_ratio: Optional[float] = None,
            critic_clip_range: Optional[float] = None,
            max_grad_norm: Optional[float] = None,
            critic_loss_coef: float = 1.0,
            entropy_loss_coef: float = 0.0) -> None:

        super().__init__(models={})

        self._input_shape = input_shape
        self._output_lengths = output_lengths
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_layer_sizes = actor_layer_sizes
        self._critic_layer_sizes = critic_layer_sizes
        self._actor_train_iterations = actor_train_iterations
        self._critic_train_iterations = critic_train_iterations
        self._clip_ratio = clip_ratio
        self._critic_clip_range = critic_clip_range
        self._max_grad_norm = max_grad_norm
        self._GAE_lambda = GAE_lambda
        self._target_kl = target_kl
        self._critic_loss_coef = critic_loss_coef
        self._entropy_loss_coef = entropy_loss_coef

        input_ = keras.Input(self._input_shape)
        actor_layers = TF2UtilsMixin.mlp_functional(
            input_, self._actor_layer_sizes, 'relu', 'actor_{i:0>2}')
        logit_heads = TF2UtilsMixin.mpl_layers(
            self._output_lengths, None, 'actor_output_{i:0>2}')
        logits = [output(actor_layers) for output in logit_heads]

        self.actor = keras.Model(inputs=input_, outputs=[logits])

        critic_layers = TF2UtilsMixin.mlp_functional(
            input_, self._critic_layer_sizes, 'relu', 'critic_{i:0>2}')
        critic_output = keras.layers.Dense(
            1, name='critic_output')(critic_layers)
        self.critic = keras.Model(inputs=input_, outputs=critic_output)

        self._actor_optimizer = keras.optimizers.Adam(
            learning_rate=self._actor_learning_rate)
        self._critic_optimizer = keras.optimizers.Adam(
            learning_rate=self._critic_learning_rate)

        self._actor_loss = tf.keras.metrics.Mean(
            'actor_loss', dtype=tf.float32)
        self._critic_loss = tf.keras.metrics.Mean(
            'critic_loss', dtype=tf.float32)
        self._entropy_loss = tf.keras.metrics.Mean(
            'entropy_loss', dtype=tf.float32)
        self._kl = tf.keras.metrics.Mean(
            'kl', dtype=tf.float32)
        self._actor_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            'actor_accuracy', dtype=tf.float32)

        self._models = {
            'actor': type(self.actor),
            'critic': type(self.critic)}

    def __call__(self, inputs, training: Optional[bool] = None) -> Any:
        logits = self.actor(inputs, training)
        values = self.critic(inputs, training)
        return logits, values

    @staticmethod
    @tf.function
    def _entropy(logits: tf.Tensor):
        # Adopted the code from OpenAI baseline
        # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/distributions.py
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0

        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='x'),
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32, name='action_indices'),
            tf.TensorSpec(shape=[None], dtype=tf.float32, name='advantage'),
        )
    )
    def train_actor(
        self, x: tf.Tensor, action_indices: tf.Tensor, advantage: tf.Tensor
    ):
        y = self.actor(x)
        initial_logprobs = self.logprobs(
            y, action_indices, self._output_lengths[0])
        _advantage = tf.divide(
            advantage - tf.math.reduce_mean(advantage),
            tf.math.reduce_std(advantage) + eps,
            name='normalized_advantage')

        trainable_vars = self.actor.trainable_variables

        actor_loss = entropy_loss = kl = tf.constant(0.0, dtype=tf.float32)
        for _ in tf.range(self._actor_train_iterations):
            with tf.GradientTape() as tape:
                new_logprobs = self.logprobs(
                    self.actor(x), action_indices, self._output_lengths[0])
                ratio = tf.exp(new_logprobs - initial_logprobs)
                if self._clip_ratio is not None:
                    clipped_ratio = tf.clip_by_value(
                        ratio, 1. - self._clip_ratio, 1. + self._clip_ratio)
                    actor_loss = -tf.reduce_mean(
                        tf.minimum(
                            ratio * _advantage, clipped_ratio * _advantage))

                else:
                    actor_loss = -tf.reduce_mean(ratio * _advantage)

                if self._entropy_loss_coef:
                    entropy_loss = self._entropy(new_logprobs)
                    entropy_loss.set_shape([])
                    # entropy_loss = self._entropy_loss_coef * tf.reduce_sum(
                    #     new_logprobs * tf.math.exp(new_logprobs))

                total_loss = actor_loss - self._entropy_loss_coef * entropy_loss

            policy_grads = tape.gradient(total_loss, trainable_vars)
            if self._max_grad_norm is not None:
                policy_grads, _ = tf.clip_by_global_norm(
                    policy_grads, self._max_grad_norm)
            self._actor_optimizer.apply_gradients(
                zip(policy_grads, trainable_vars))

            y = self.actor(x)
            new_logprobs = self.logprobs(
                y, action_indices, self._output_lengths[0])
            kl = .5 * tf.reduce_mean(
                tf.square(new_logprobs - initial_logprobs))
            # kl = tf.reduce_sum(
            #     tf.exp(initial_logprobs) * (initial_logprobs - new_logprobs))
            # kl = tf.reduce_mean(initial_logprobs - new_logprobs)

            # kl = tf.reduce_sum(kl)
            if kl > 1.5 * self._target_kl:  # Early Stopping
                break

        self._kl.update_state(kl)
        self._actor_accuracy.update_state(
            tf.squeeze(action_indices), y[0])
        self._actor_loss.update_state(actor_loss)
        if self._entropy_loss_coef:
            self._entropy_loss.update_state(entropy_loss)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='x'),
            tf.TensorSpec(shape=[None], dtype=tf.float32, name='returns'),
        )
    )
    def train_critic(self, x, returns):
        old_values = self.critic(x)
        for _ in tf.range(self._critic_train_iterations):
            with tf.GradientTape() as tape:
                new_values = self.critic(x)
                if self._critic_clip_range is not None:
                    values_clipped = old_values + tf.clip_by_value(
                        new_values - old_values, -self._critic_clip_range,
                        self._critic_clip_range)
                    loss_unclipped = tf.square(returns - new_values)
                    loss_clipped = tf.square(returns - values_clipped)
                    critic_loss = 0.5 * tf.reduce_mean(
                        tf.maximum(loss_unclipped, loss_clipped)
                    )
                else:
                    critic_loss = tf.reduce_mean(
                        tf.square(returns - new_values))

            self._critic_loss.update_state(critic_loss)
            trainable_vars = self.critic.trainable_variables
            value_grads = tape.gradient(critic_loss, trainable_vars)
            if self._max_grad_norm is not None:
                value_grads, _ = tf.clip_by_global_norm(
                    value_grads, self._max_grad_norm)

            self._critic_optimizer.apply_gradients(
                zip(value_grads, trainable_vars))

    @staticmethod
    def logprobs(logits, action_indices, action_count):
        return -tf.nn.softmax_cross_entropy_with_logits(
            tf.one_hot(tf.squeeze(action_indices), action_count),
            tf.squeeze(logits))

        # logprobabilities_all = tf.nn.log_softmax(tf.squeeze(logits))
        # logprobability = tf.reduce_sum(
        #     tf.one_hot(tf.squeeze(action_indices), action_count)
        #     * logprobabilities_all,
        #     axis=1)
        # # logprobability = tf.reduce_sum(
        # #     tf.one_hot(action_indices, action_count) * logprobabilities_all,
        # #     axis=1)

        # return logprobability

    def train_step(self, data):
        x, (action_indices, returns, advantage) = data
        self.train_actor(x, action_indices, advantage)
        self.train_critic(x, returns)

        metrics = {
            'actor_loss': self._actor_loss.result(),
            'critic_loss': self._critic_loss.result()
        }

        if self._entropy_loss_coef:
            metrics['entropy_loss'] = self._entropy_loss.result()

        metrics['total_loss'] = sum(x for x in metrics.values())
        metrics['kl'] = self._kl.result()

        self._actor_loss.reset_states()
        self._critic_loss.reset_states()
        self._entropy_loss.reset_states()
        self._actor_accuracy.reset_states()

        return metrics


class PPOLearner(Learner[FeatureSet, ACLabelType]):
    '''
    PPO Learner
    '''

    def __init__(
            self,
            model: PPOModel,
            **kwargs: Any) -> None:
        '''
        Arguments
        ---------
        tensorboard_path:
            A path to save tensorboard outputs. If not provided,
            tensorboard will be disabled.
        '''

        super().__init__(**kwargs)

        self._model = model

        self._iteration = 0

    def predict(
            self, X: Tuple[FeatureSet, ...], training: Optional[bool] = None
    ) -> Tuple[ACLabelType, ...]:
        '''
        predict `y` for a given input list `X`.

        Arguments
        ---------
        X:
            A list of `FeatureSet` as inputs to the prediction model.

        training:
            Whether the learner is in training mode. (Default = None)

        Returns
        -------
        :
            The predicted `y`.
        '''
        return self._model(TF2UtilsMixin.convert_to_tensor(X), training=training)

    def learn(
            self, X: Tuple[FeatureSet, ...],
            Y: Tuple[ACLabelType, ...]) -> Dict[str, float]:
        '''
        Learn using the training set `X` and `Y`.

        Arguments
        ---------
        X:
            A list of `FeatureSet` as inputs to the learning model.

        Y:
            A list of float labels for the learning model.
        '''
        _X = TF2UtilsMixin.convert_to_tensor(X)
        if len(_X.shape) == 1:
            _X = tf.expand_dims(_X, axis=0)

        action_index_temp, return_temp, advantage_temp = tuple(zip(*Y))
        action_index: tf.Tensor = tf.convert_to_tensor(action_index_temp)
        returns = tf.convert_to_tensor(return_temp, dtype=tf.float32)
        advantage = tf.convert_to_tensor(advantage_temp, dtype=tf.float32)

        metrics = self._model.train_step(
            (_X, (action_index, returns, advantage)))

        self._iteration += 1

        return metrics

    def get_parameters(self) -> Any:
        return (
            self._model.actor.get_weights(), self._model.critic.get_weights())

    def set_parameters(self, parameters: Any):
        self._model.actor.set_weights(parameters[0])
        self._model.critic.set_weights(parameters[1])
