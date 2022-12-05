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
from reil.utils.tf_utils import (MeanMetric, SparseCategoricalAccuracyMetric,
                                 TF2UtilsMixin)
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
            actor_hidden_activation: str = 'relu',
            actor_head_activation: Optional[str] = None,
            critic_hidden_activation: str = 'relu',
            clip_ratio: Optional[float] = None,
            critic_clip_range: Optional[float] = None,
            max_grad_norm: Optional[float] = None,
            critic_loss_coef: float = 1.0,
            entropy_loss_coef: float = 0.0,
            regularizer_coef: float = 0.0) -> None:

        super().__init__(models={})

        self._input_shape = input_shape
        self._output_lengths = [
            tf.constant(i, dtype=tf.int32) for i in output_lengths]
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
        self._actor_hidden_activation = actor_hidden_activation
        self._critic_hidden_activation = critic_hidden_activation
        self._actor_head_activation = actor_head_activation
        self._critic_loss_coef = critic_loss_coef
        self._entropy_loss_coef = entropy_loss_coef
        self._regularizer_coef = regularizer_coef

        input_: tf.Tensor = keras.Input(self._input_shape)  # type: ignore
        actor_layers = TF2UtilsMixin.mlp_functional(
            input_, self._actor_layer_sizes, actor_hidden_activation, 'actor_{i:0>2}')
        logit_heads = TF2UtilsMixin.mpl_layers(
            output_lengths, actor_head_activation, 'actor_output_{i:0>2}')
        logits = tuple(output(actor_layers) for output in logit_heads)

        self.actor = keras.Model(
            inputs=input_,
            outputs=logits if len(logits) > 1 else tuple(logits))

        critic_layers = TF2UtilsMixin.mlp_functional(
            input_, self._critic_layer_sizes, critic_hidden_activation, 'critic_{i:0>2}')
        critic_output = keras.layers.Dense(
            1, name='critic_output')(critic_layers)
        self.critic = keras.Model(inputs=input_, outputs=critic_output)

        self._actor_optimizer = keras.optimizers.Adam(
            learning_rate=self._actor_learning_rate)  # type: ignore
        self._critic_optimizer = keras.optimizers.Adam(
            learning_rate=self._critic_learning_rate)  # type: ignore

        self._actor_loss = MeanMetric('actor_loss', dtype=tf.float32)
        self._critic_loss = MeanMetric('critic_loss', dtype=tf.float32)
        self._entropy_loss = MeanMetric('entropy_loss', dtype=tf.float32)
        self._regularizer_loss = MeanMetric(
            'regularizer_loss', dtype=tf.float32)
        self._actor_accuracy = SparseCategoricalAccuracyMetric(
            'actor_accuracy', dtype=tf.float32)
        self._kl = MeanMetric('kl', dtype=tf.float32)
        self._actor_accuracy = SparseCategoricalAccuracyMetric(
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
        print('tracing PPOModel._entropy')
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0

        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='x'),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='action_indices'),
            tf.TensorSpec(shape=[None], dtype=tf.float32, name='advantage'),
        )
    )
    def train_actor(  # noqa: C901
        self, x: tf.Tensor, action_indices: tf.Tensor, advantage: tf.Tensor
    ):
        print(f'tracing {self.__class__.__qualname__}.train_actor')
        lengths = self._output_lengths
        starts = tf.pad(tf.cast(lengths[:-1], tf.int32), [[1, 0]])
        ends = tf.math.cumsum(lengths)
        m = len(lengths)

        logits_concat = tf.concat(self.actor(x), axis=1, name='all_logits')

        initial_logprobs = tf.expand_dims(
            self.logprobs(
                logits_concat[:, starts[0]:ends[0]],  # type: ignore
                tf.gather(action_indices, 0, axis=1),
                tf.gather(self._output_lengths, 0)),
            axis=1)
        for j in tf.range(1, m):
            initial_logprobs = tf.concat([
                initial_logprobs,
                tf.expand_dims(
                    self.logprobs(
                        logits_concat[:, starts[j]:ends[j]],  # type: ignore
                        tf.gather(action_indices, j, axis=1),
                        tf.gather(self._output_lengths, j)),
                    axis=1)], axis=1)

        _advantage = tf.divide(
            advantage - tf.math.reduce_mean(advantage),
            tf.math.reduce_std(advantage) + eps,
            name='normalized_advantage')

        trainable_vars = self.actor.trainable_variables

        total_loss = tf.constant(0.0, dtype=tf.float32)
        actor_loss = entropy_loss = regularizer_loss = kl = tf.constant(
            0.0, dtype=tf.float32)
        for _ in tf.range(self._actor_train_iterations):
            with tf.GradientTape() as tape:
                logits_concat = tf.concat(self.actor(x), axis=1, name='all_logits')
                for j in tf.range(m):
                    new_logprobs = self.logprobs(
                        logits_concat[:, starts[j]:ends[j]],  # type: ignore
                        tf.gather(action_indices, j, axis=1),
                        tf.gather(self._output_lengths, j))

                    ratio = tf.exp(new_logprobs - tf.gather(initial_logprobs, j, axis=1))
                    if self._clip_ratio is None:
                        actor_loss = -tf.reduce_mean(ratio * _advantage)
                    else:
                        clipped_ratio = tf.clip_by_value(
                            ratio, 1. - self._clip_ratio, 1. + self._clip_ratio)
                        actor_loss = -tf.reduce_mean(
                            tf.minimum(
                                ratio * _advantage, clipped_ratio * _advantage))

                    if self._entropy_loss_coef:
                        entropy_loss = self._entropy(new_logprobs)
                        entropy_loss.set_shape([])
                        # entropy_loss = self._entropy_loss_coef * tf.reduce_sum(
                        #     new_logprobs * tf.math.exp(new_logprobs))

                    if self._regularizer_coef:
                        weights_concat = tf.concat([
                            self.actor.layers[-1].weights[0],
                            tf.expand_dims(self.actor.layers[-1].weights[1], axis=0)
                        ], axis=0)
                        regularizer_loss = tf.reduce_sum(
                            tf.math.reduce_euclidean_norm(weights_concat, axis=0)
                            # tf.reduce_max(tf.math.abs(weights_concat), axis=0)
                        )

                    total_loss += (
                        actor_loss
                        + self._entropy_loss_coef * entropy_loss
                        + self._regularizer_coef * regularizer_loss
                    )

            policy_grads = tape.gradient(total_loss, trainable_vars)
            if self._max_grad_norm is not None:
                policy_grads, _ = tf.clip_by_global_norm(
                    policy_grads, self._max_grad_norm)
            self._actor_optimizer.apply_gradients(
                zip(policy_grads, trainable_vars))

            logits_concat = tf.concat(self.actor(x), axis=1, name='all_logits')

            new_logprobs = tf.expand_dims(
                self.logprobs(
                    logits_concat[:, starts[0]:ends[0]],  # type: ignore
                    tf.gather(action_indices, 0, axis=1),
                    tf.gather(self._output_lengths, 0)),
                axis=1)
            for j in tf.range(1, m):
                new_logprobs = tf.concat([
                    new_logprobs,
                    tf.expand_dims(
                        self.logprobs(
                            logits_concat[:, starts[j]:ends[j]],  # type: ignore
                            tf.gather(action_indices, j, axis=1),
                            tf.gather(self._output_lengths, j)),
                        axis=1)], axis=1)
            # new_logprobs = tf.concat([
            #     tf.expand_dims(
            #         self.logprobs(
            #             logits_concat[:, starts[j]:ends[j]],
            #             tf.gather(action_indices, 0, axis=1),
            #             tf.gather(self._output_lengths, j)),
            #         axis=1)
                # for j in tf.range(m)], axis=1)
            kl = .5 * tf.reduce_mean(
                tf.square(new_logprobs - initial_logprobs))  # type: ignore

            if kl > 1.5 * self._target_kl:  # Early Stopping
                break

        self._kl.update_state(kl)
        # self._actor_accuracy.update_state(
        #     tf.squeeze(action_indices), y[0])
        self._actor_loss.update_state(actor_loss)
        if self._entropy_loss_coef:
            self._entropy_loss.update_state(entropy_loss)
        if self._regularizer_coef:
            self._regularizer_loss.update_state(regularizer_loss)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='x'),
            tf.TensorSpec(shape=[None], dtype=tf.float32, name='returns'),
        )
    )
    def train_critic(self, x, returns):
        print(f'tracing {self.__class__.__qualname__}.train_critic')
        old_values = self.critic(x)
        for _ in tf.range(self._critic_train_iterations):
            with tf.GradientTape() as tape:
                new_values = self.critic(x)
                if self._critic_clip_range is not None:
                    values_clipped = old_values + tf.clip_by_value(  # type: ignore
                        new_values - old_values, -self._critic_clip_range,  # type: ignore
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
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='logits'),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name='action_indices'),
            tf.TensorSpec(shape=[], dtype=tf.int32, name='action_count'),
        )
    )
    def logprobs(logits, action_indices, action_count):
        print('tracing PPOModel.logprobs')
        return -tf.nn.softmax_cross_entropy_with_logits(  # type: ignore
            tf.one_hot(tf.squeeze(action_indices), action_count),
            tf.squeeze(logits))

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

        if self._regularizer_coef:
            metrics['regularizer_loss'] = self._regularizer_loss.result()

        metrics['total_loss'] = sum(
            x for x in metrics.values())  # type: ignore
        metrics['kl'] = self._kl.result()

        self._actor_loss.reset_states()
        self._critic_loss.reset_states()
        self._entropy_loss.reset_states()
        self._actor_accuracy.reset_states()

        return metrics


@keras.utils.register_keras_serializable(
    package='reil.learners.ppo_learner')
class PPONeighborEffect(PPOModel):
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
        entropy_loss_coef: float = 0.0,
        effect_widths: Union[int, Tuple[int, ...]] = 0,
        effect_decay_factors: Union[float, Tuple[float, ...]] = 0.,
        regularizer_coef: float = 0.0
    ) -> None:
        super().__init__(
            input_shape=input_shape,
            output_lengths=output_lengths,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_layer_sizes=actor_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            actor_train_iterations=actor_train_iterations,
            critic_train_iterations=critic_train_iterations,
            GAE_lambda=GAE_lambda,
            target_kl=target_kl,
            clip_ratio=clip_ratio,
            critic_clip_range=critic_clip_range,
            max_grad_norm=max_grad_norm,
            critic_loss_coef=critic_loss_coef,
            entropy_loss_coef=entropy_loss_coef,
            regularizer_coef=regularizer_coef
        )
        output_heads = len(output_lengths)
        if isinstance(effect_widths, int):
            _effect_widths = [effect_widths] * output_heads
        elif not effect_widths:
            _effect_widths = [0] * output_heads
        elif len(effect_widths) != output_heads:
            raise ValueError(
                'effect_widths should be an int or a tuple of size '
                f'{output_heads}.')
        else:
            _effect_widths = effect_widths

        if isinstance(effect_decay_factors, float) or not effect_decay_factors:
            _effect_decay_factors = [effect_decay_factors] * output_heads
        elif not effect_decay_factors:
            _effect_decay_factors = [0.] * output_heads
        elif len(effect_decay_factors) != output_heads:
            raise ValueError(
                'effect_decay_factors should be a float or a tuple of size '
                f'{output_heads}.')
        else:
            _effect_decay_factors = effect_decay_factors

        self._effect_widths = tf.constant(
            _effect_widths, name='effect_width', dtype=tf.int32)
        self._effect_decay_factors = tf.constant(
            _effect_decay_factors, name='effect_decay_factors',
            dtype=tf.float32)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(dict(
            effect_widths=tuple(self._effect_widths.numpy()),
            effect_decay_factors=tuple(self._effect_decay_factors.numpy()),
        ))

        return config

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='x'),
            tf.TensorSpec(shape=[None, None],
                          dtype=tf.int32, name='action_indices'),
            tf.TensorSpec(shape=[None], dtype=tf.float32, name='advantage'),
        )
    )
    def train_actor(  # noqa: C901
        self, x: tf.Tensor, action_indices: tf.Tensor, advantage: tf.Tensor
    ):
        print(f'tracing {self.__class__.__qualname__}.train_actor')
        lengths = self._output_lengths
        starts = tf.pad(tf.cast(lengths[:-1], tf.int32), [[1, 0]])
        ends = tf.math.cumsum(lengths)
        m = len(lengths)

        logits_concat = tf.concat(self.actor(x), axis=1, name='all_logits')

        initial_logprobs = tf.expand_dims(
            self.logprobs(
                logits_concat[:, starts[0]:ends[0]],  # type: ignore
                tf.gather(action_indices, 0, axis=1),
                tf.gather(self._output_lengths, 0)),
            axis=1)
        for j in tf.range(1, m):
            initial_logprobs = tf.concat([
                initial_logprobs,
                tf.expand_dims(
                    self.logprobs(
                        logits_concat[:, starts[j]:ends[j]],  # type: ignore
                        tf.gather(action_indices, j, axis=1),
                        tf.gather(self._output_lengths, j)),
                    axis=1)], axis=1)

        _advantage = tf.divide(
            advantage - tf.math.reduce_mean(advantage),
            tf.math.reduce_std(advantage) + eps,
            name='normalized_advantage')

        trainable_vars = self.actor.trainable_variables

        total_loss = tf.constant(0.0, dtype=tf.float32)
        actor_loss = entropy_loss = regularizer_loss = kl = tf.constant(
            0.0, dtype=tf.float32)
        for _ in tf.range(self._actor_train_iterations):
            with tf.GradientTape() as tape:
                logits_concat = tf.concat(
                    self.actor(x), axis=1, name='all_logits')
                for j in tf.range(m):
                    logits_slice = logits_concat[:, starts[j]:ends[j]]  # type: ignore
                    y_slice = tf.gather(action_indices, j, axis=1)
                    output_length = tf.gather(self._output_lengths, j)

                    j_one_hot = tf.one_hot(j, depth=m, dtype=tf.int32)
                    effect_width = tf.dynamic_partition(  # type: ignore
                        self._effect_widths, j_one_hot, 2)[1][0]

                    new_logprobs = self.logprobs(
                        logits_slice, y_slice, output_length)
                    if tf.equal(effect_width, 0):
                        ratio = tf.exp(
                            new_logprobs - tf.gather(initial_logprobs, j, axis=1))
                        if self._clip_ratio is not None:
                            clipped_ratio = tf.clip_by_value(
                                ratio, 1. - self._clip_ratio, 1. + self._clip_ratio)
                            actor_loss = -tf.reduce_mean(
                                tf.minimum(
                                    ratio * _advantage, clipped_ratio * _advantage))

                        else:
                            actor_loss = -tf.reduce_mean(ratio * _advantage)
                    else:
                        for diff in tf.range(-effect_width, effect_width + 1):
                            temp = y_slice + diff
                            _length = tf.dynamic_partition(  # type: ignore
                                lengths, j_one_hot, 2)[1][0]
                            in_range_indicator = tf.logical_and(
                                tf.greater_equal(temp, 0),
                                tf.less(temp, _length))

                            # if not tf.reduce_all(in_range_indicator):
                            #     continue

                            in_range_indices = tf.cast(
                                in_range_indicator, tf.int32)

                            advantage_in_range = tf.dynamic_partition(  # type: ignore
                                _advantage, in_range_indices, 2)[1]
                            y_in_range = tf.dynamic_partition(  # type: ignore
                                temp, in_range_indices, 2)[1]
                            initial_logprobs_in_range = tf.dynamic_partition(  # type: ignore
                                initial_logprobs, in_range_indices, 2)[1]

                            logits_in_range = tf.dynamic_partition(  # type: ignore
                                logits_slice, in_range_indices, 2)[1]
                            new_logprobs_in_range = self.logprobs(
                                logits_in_range, y_in_range, output_length)

                            abs_diff = tf.cast(tf.abs(diff), dtype=tf.float32)
                            effect_decay = tf.dynamic_partition(  # type: ignore
                                self._effect_decay_factors, j_one_hot, 2)[1][0]
                            effect = tf.pow(effect_decay, abs_diff)
                            ratio = tf.exp(
                                new_logprobs_in_range - tf.gather(
                                    initial_logprobs_in_range, j, axis=1))
                            if self._clip_ratio is not None:
                                clipped_ratio = tf.clip_by_value(
                                    ratio, 1. - self._clip_ratio, 1. + self._clip_ratio)
                                _loss = -tf.reduce_mean(
                                    tf.minimum(
                                        ratio * advantage_in_range,
                                        clipped_ratio * advantage_in_range))

                            else:
                                _loss = - \
                                    tf.reduce_mean(ratio * advantage_in_range)

                            actor_loss += effect * _loss

                    if self._entropy_loss_coef:
                        entropy_loss = self._entropy(new_logprobs)
                        entropy_loss.set_shape([])

                    if self._regularizer_coef:
                        weights_concat = tf.concat([
                            self.actor.layers[-1].weights[0],
                            tf.expand_dims(
                                self.actor.layers[-1].weights[1], axis=0)
                        ], axis=0)
                        regularizer_loss = tf.reduce_sum(
                            tf.math.reduce_euclidean_norm(weights_concat, axis=0)
                            # tf.reduce_max(tf.math.abs(weights_concat), axis=0)
                        )

                    total_loss += (
                        actor_loss
                        + self._entropy_loss_coef * entropy_loss
                        + self._regularizer_coef * regularizer_loss
                    )

            policy_grads = tape.gradient(total_loss, trainable_vars)
            if self._max_grad_norm is not None:
                policy_grads, _ = tf.clip_by_global_norm(
                    policy_grads, self._max_grad_norm)
            self._actor_optimizer.apply_gradients(
                zip(policy_grads, trainable_vars))

            logits_concat = tf.concat(self.actor(x), axis=1, name='all_logits')

            new_logprobs = tf.expand_dims(
                self.logprobs(
                    logits_concat[:, starts[0]:ends[0]],  # type: ignore
                    tf.gather(action_indices, 0, axis=1),
                    tf.gather(self._output_lengths, 0)),
                axis=1)
            for j in tf.range(1, m):
                new_logprobs = tf.concat([
                    new_logprobs,
                    tf.expand_dims(
                        self.logprobs(
                            logits_concat[:, starts[j]:ends[j]],  # type: ignore
                            tf.gather(action_indices, j, axis=1),
                            tf.gather(self._output_lengths, j)),
                        axis=1)], axis=1)
            kl = .5 * tf.reduce_mean(
                tf.square(new_logprobs - initial_logprobs))  # type: ignore

            if kl > 1.5 * self._target_kl:  # Early Stopping
                break

        self._kl.update_state(kl)
        self._actor_loss.update_state(actor_loss)
        if self._entropy_loss_coef:
            self._entropy_loss.update_state(entropy_loss)
        if self._regularizer_coef:
            self._regularizer_loss.update_state(regularizer_loss)


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

        return metrics  # type: ignore

    def get_parameters(self) -> Any:
        return (
            self._model.actor.get_weights(), self._model.critic.get_weights())

    def set_parameters(self, parameters: Any):
        self._model.actor.set_weights(parameters[0])
        self._model.critic.set_weights(parameters[1])
