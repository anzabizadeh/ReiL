# -*- coding: utf-8 -*-
'''
PPOLearner class
================

'''
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import tensorflow as tf
from tensorflow import Tensor, TensorShape, TensorSpec

from reil.datatypes.feature import FeatureSet
from reil.learners.learner import Learner
from reil.utils.tf_utils import (JIT_COMPILE, MeanMetric, SparseCategoricalAccuracyMetric,
                                 TF2UtilsMixin, entropy, logprobs)

keras = tf.keras
from keras.optimizers.schedules.learning_rate_schedule import \
    LearningRateSchedule  # noqa: E402

ACLabelType = tuple[tuple[tuple[int, ...], ...], float]

eps: Tensor = tf.constant(np.finfo(np.float32).eps.item(), dtype=tf.float32)
zero_int32: Tensor = tf.constant(0, tf.int32)
one_int32: Tensor = tf.constant(1, tf.int32)
zero_float32: Tensor = tf.constant(0., tf.float32)
one_float32: Tensor = tf.constant(1., tf.float32)


@tf.function(jit_compile=JIT_COMPILE)
def _less_than_condition(j: Tensor, m: Tensor, *rest) -> Tensor:
    return tf.less(j, m, name='less_than')  # type: ignore


@keras.utils.register_keras_serializable(package='reil.learners.ppo_learner')
class PPOModel(TF2UtilsMixin):
    def __init__(
            self,
            input_shape: tuple[int, ...],
            action_per_head: tuple[int, ...],
            actor_learning_rate: float | LearningRateSchedule,
            critic_learning_rate: float | LearningRateSchedule,
            actor_layer_sizes: tuple[int, ...],
            critic_layer_sizes: tuple[int, ...],
            actor_train_iterations: int,
            critic_train_iterations: int,
            target_kl: float,
            actor_hidden_activation: str = 'relu',
            actor_head_activation: str | None = None,
            critic_hidden_activation: str = 'relu',
            clip_ratio: float | None = None,
            critic_clip_range: float | None = None,
            max_grad_norm: float | None = None,
            critic_loss_coef: float = 1.0,
            entropy_loss_coef: float = 0.0,
            regularizer_coef: float = 0.0) -> None:

        super().__init__(models={})

        self._input_shape = input_shape
        self._action_per_head: list[Tensor] = [
            tf.constant(i, dtype=tf.int32, name=f'action_in_head_{i}')
            for i in action_per_head
        ]
        self._head_count: Tensor = tf.constant(
            len(action_per_head), dtype=tf.int32, name='head_count')
        self._starts: Tensor = tf.pad(
            tf.cast(action_per_head[:-1], tf.int32), [[1, 0]], name='starts')
        self._ends: Tensor = tf.math.cumsum(action_per_head, name='ends')

        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_layer_sizes = actor_layer_sizes
        self._critic_layer_sizes = critic_layer_sizes
        self._actor_train_iterations = actor_train_iterations
        self._critic_train_iterations = critic_train_iterations
        self._clip_ratio: Tensor | None
        self._critic_clip_range: Tensor | None
        self._max_grad_norm: Tensor | None
        if clip_ratio is None:
            self._clip_ratio = None
        else:
            self._clip_ratio = tf.constant(
                clip_ratio, dtype=tf.float32, name='clip_ratio')
        if critic_clip_range is None:
            self._critic_clip_range = None
        else:
            self._critic_clip_range = tf.constant(
                critic_clip_range, dtype=tf.float32, name='critic_clip_range')
        if max_grad_norm is None:
            self._max_grad_norm = None
        else:
            self._max_grad_norm = tf.constant(
                max_grad_norm, dtype=tf.float32, name='max_gradient_norm')
        # self._GAE_lambda = GAE_lambda  # see proximal_policy_optimization.py
        self._target_kl = target_kl
        self._1_5_target_kl: Tensor = tf.multiply(1.5, target_kl, name='1.5_target_kl')
        self._actor_hidden_activation = actor_hidden_activation
        self._critic_hidden_activation = critic_hidden_activation
        self._actor_head_activation = actor_head_activation
        self._critic_loss_coef: Tensor = tf.constant(
            critic_loss_coef, dtype=tf.float32, name='critic_loss_coef')
        self._entropy_loss_coef: Tensor = tf.constant(
            entropy_loss_coef, dtype=tf.float32, name='entropy_loss_coef')
        self._regularizer_coef: Tensor = tf.constant(
            regularizer_coef, dtype=tf.float32, name='regularizer_coef')

        input_: Tensor = keras.Input(self._input_shape)  # type: ignore
        actor_layers = TF2UtilsMixin.mlp_functional(
            input_, self._actor_layer_sizes, actor_hidden_activation, 'actor_{i:0>2}')
        logit_heads = TF2UtilsMixin.mpl_layers(
            action_per_head, actor_head_activation, 'actor_output_{i:0>2}')
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

    def __call__(self, inputs, training: bool | None = None) -> Any:
        logits = self.actor(inputs, training)
        values = self.critic(inputs, training)
        return logits, values

    @staticmethod
    @tf.function  # (jit_compile=False) see tf_utils.logprobs
    def _logprobs_j(
            j: Tensor, logits_concat: Tensor, starts: Tensor, ends: Tensor,
            action_indices: Tensor, action_per_head: Tensor, expand_dim: bool = True) -> Tensor:
        temp = logprobs(
            logits_concat[:, starts[j]:ends[j]],  # type: ignore
            tf.gather(action_indices, j, axis=1),
            tf.gather(action_per_head, j))
        if expand_dim:
            return tf.expand_dims(temp, axis=1)

        return temp

    @staticmethod
    @tf.function(jit_compile=False)
    def _logprobs_concat(logits_concat, starts, ends, action_indices, action_per_head, head_count):
        def _body(
                j, head_count, logits_concat, starts, ends,
                action_indices, action_per_head, results):
            return [
                j + 1, head_count, logits_concat, starts, ends, action_indices, action_per_head,
                tf.concat([
                    results,
                    PPOModel._logprobs_j(
                        j, logits_concat, starts, ends,
                        action_indices, action_per_head)],
                    axis=1)
            ]
        result: Tensor = PPOModel._logprobs_j(  # type: ignore
            zero_int32, logits_concat, starts, ends, action_indices, action_per_head)
        result = tf.while_loop(  # type: ignore
            cond=_less_than_condition,
            body=_body,
            loop_vars=(
                one_int32, head_count, logits_concat, starts, ends, action_indices,
                action_per_head, result),
            shape_invariants=(
                TensorShape([]), TensorShape([]), TensorShape([None, None]),
                TensorShape([None]), TensorShape([None]), TensorShape([None, None]),
                [o.get_shape() for o in action_per_head], TensorShape([None, None])),
            parallel_iterations=1
        )

        return result[-1]

    @tf.function(
        input_signature=(
            TensorSpec(shape=[None, None], dtype=tf.float32, name='x'),
            TensorSpec(shape=[None, None], dtype=tf.int32, name='action_indices'),
            TensorSpec(shape=[None], dtype=tf.float32, name='advantage'),
        ),
        jit_compile=False
    )
    def train_actor(  # noqa: C901
        self, x: Tensor, action_indices: Tensor, advantage: Tensor
    ):
        print(f'tracing {self.__class__.__qualname__}.train_actor')
        action_per_head = self._action_per_head
        head_count = self._head_count
        starts = self._starts
        ends = self._ends

        logits_concat = tf.concat(self.actor(x), axis=1, name='all_logits')

        initial_logprobs = self._logprobs_concat(
            logits_concat, starts, ends, action_indices, action_per_head, head_count)
        # initial_logprobs = tf.expand_dims(
        #     self.logprobs(
        #         logits_concat[:, starts[0]:ends[0]],  # type: ignore
        #         tf.gather(action_indices, 0, axis=1),
        #         tf.gather(self._action_per_head, 0)),
        #     axis=1)
        # for j in tf.range(one_int32, self._head_count):
        #     tf.autograph.experimental.set_loop_options(
        #         shape_invariants=(initial_logprobs, [None, None])
        #     )
        #     initial_logprobs = tf.concat([
        #         initial_logprobs,
        #         tf.expand_dims(
        #             self.logprobs(
        #                 logits_concat[:, starts[j]:ends[j]],  # type: ignore
        #                 tf.gather(action_indices, j, axis=1),
        #                 tf.gather(self._action_per_head, j)),
        #             axis=1)], axis=1)

        _advantage = tf.divide(
            advantage - tf.math.reduce_mean(advantage),
            tf.math.reduce_std(advantage) + eps,
            name='normalized_advantage')

        trainable_vars = self.actor.trainable_variables

        actor_loss = entropy_loss = regularizer_loss = kl = zero_float32
        for _ in tf.range(self._actor_train_iterations):
            total_loss = zero_float32
            with tf.GradientTape() as tape:
                logits_concat = tf.concat(self.actor(x), axis=1, name='all_logits')
                for j in tf.range(head_count):
                    new_logprobs = self._logprobs_j(
                        j, logits_concat, starts, ends, action_indices,
                        self._action_per_head, False)
                    # new_logprobs = self.logprobs(
                    #     logits_concat[:, starts[j]:ends[j]],  # type: ignore
                    #     tf.gather(action_indices, j, axis=1),
                    #     tf.gather(self._action_per_head, j))

                    actor_loss = self._compute_actor_loss(
                        initial_logprobs, new_logprobs, _advantage, j)

                    if tf.cast(self._entropy_loss_coef, tf.bool):
                        entropy_loss = entropy(new_logprobs)
                        entropy_loss.set_shape([])
                        # entropy_loss = self._entropy_loss_coef * tf.reduce_sum(
                        #     new_logprobs * tf.math.exp(new_logprobs))

                    if tf.cast(self._regularizer_coef, tf.bool):
                        regularizer_loss = self._compute_regularizer_loss()

                    total_loss = tf.add_n(
                        [
                            total_loss,
                            actor_loss,
                            tf.multiply(self._entropy_loss_coef, entropy_loss),
                            tf.multiply(self._regularizer_coef, regularizer_loss)
                        ],
                        name='total_loss'
                    )

            policy_grads = tape.gradient(total_loss, trainable_vars)
            if self._max_grad_norm is not None:
                policy_grads, _ = tf.clip_by_global_norm(
                    policy_grads, self._max_grad_norm, name='clipped_policy_grads')
            self._actor_optimizer.apply_gradients(zip(policy_grads, trainable_vars))

            logits_concat = tf.concat(self.actor(x), axis=1, name='all_logits')

            new_logprobs = self._logprobs_concat(
                logits_concat, starts, ends, action_indices, action_per_head, head_count)

            # new_logprobs = tf.expand_dims(
            #     self.logprobs(
            #         logits_concat[:, starts[0]:ends[0]],  # type: ignore
            #         tf.gather(action_indices, 0, axis=1),
            #         tf.gather(self._action_per_head, 0)),
            #     axis=1)
            # for j in tf.range(one_int32, self._head_count):
            #     tf.autograph.experimental.set_loop_options(
            #         shape_invariants=(new_logprobs, [None, None])
            #     )
            #     new_logprobs = tf.concat([
            #         new_logprobs,
            #         tf.expand_dims(
            #             self.logprobs(
            #                 logits_concat[:, starts[j]:ends[j]],  # type: ignore
            #                 tf.gather(action_indices, j, axis=1),
            #                 tf.gather(self._action_per_head, j)),
            #             axis=1)], axis=1)

            kl = .5 * tf.reduce_mean(
                tf.square(tf.subtract(new_logprobs, initial_logprobs, name='delta_logprobs')),
                name='kl')

            if tf.greater(kl, self._1_5_target_kl):  # Early Stopping
                break

        self._kl.update_state(kl)
        # self._actor_accuracy.update_state(
        #     tf.squeeze(action_indices), y[0])
        self._actor_loss.update_state(actor_loss)
        if tf.cast(self._entropy_loss_coef, tf.bool):
            self._entropy_loss.update_state(entropy_loss)
        if tf.cast(self._regularizer_coef, tf.bool):
            self._regularizer_loss.update_state(regularizer_loss)

    @tf.function  # (jit_compile=False)
    def _compute_regularizer_loss(self):
        weights_concat = tf.concat([
            self.actor.layers[-1].weights[0],
            tf.expand_dims(self.actor.layers[-1].weights[1], axis=0)
        ], axis=0, name='actor_weights')
        regularizer_loss = tf.reduce_sum(
            tf.math.reduce_euclidean_norm(weights_concat, axis=0),
            name='regularizer_loss'
            # tf.reduce_max(tf.math.abs(weights_concat), axis=0)
        )

        return regularizer_loss

    @tf.function(jit_compile=JIT_COMPILE)
    def _compute_actor_loss(self, initial_logprobs, new_logprobs, _advantage, j):
        ratio: Tensor = tf.exp(
            tf.subtract(
                new_logprobs, tf.gather(initial_logprobs, j, axis=1),
                name='delta_logprobs'),
            name='ratio'
        )
        if self._clip_ratio is None:
            actor_loss = -tf.reduce_mean(
                tf.multiply(ratio, _advantage), name='actor_loss')
        else:
            clipped_ratio = tf.clip_by_value(
                ratio,
                tf.subtract(one_float32, self._clip_ratio),
                tf.add(one_float32, self._clip_ratio),
                name='clipped_ratio')
            actor_loss = -tf.reduce_mean(
                tf.minimum(
                    tf.multiply(ratio, _advantage, name='ratio_times_adv'),
                    tf.multiply(clipped_ratio, _advantage,
                                name='clipped_ratio_times_adv')),
                name='actor_loss_clipped')

        return actor_loss

    @tf.function(
        input_signature=(
            TensorSpec(shape=[None, None], dtype=tf.float32, name='x'),
            TensorSpec(shape=[None], dtype=tf.float32, name='returns'),
        ),
        jit_compile=JIT_COMPILE
    )
    def train_critic(self, x, returns):
        print(f'tracing {self.__class__.__qualname__}.train_critic')
        old_values = self.critic(x)
        for _ in tf.range(self._critic_train_iterations):
            with tf.GradientTape() as tape:
                new_values = self.critic(x)
                if self._critic_clip_range is not None:
                    values_clipped = tf.add(
                        old_values,
                        tf.clip_by_value(
                            tf.subtract(new_values, old_values, name='delta_values'),
                            tf.negative(self._critic_clip_range, name='neg_critic_clip_range'),
                            self._critic_clip_range),
                        name='clipped_values'
                    )
                    loss_unclipped = tf.square(
                        tf.subtract(returns, new_values, name='delta_return'),
                        name='square_delta_return')
                    loss_clipped = tf.square(
                        tf.subtract(returns, values_clipped, name='delta_clipped_return'),
                        name='square_delta_clipped_return')
                    critic_loss = tf.multiply(
                        0.5,
                        tf.reduce_mean(tf.maximum(loss_unclipped, loss_clipped)),
                        name='clipped_critic_loss'
                    )
                else:
                    critic_loss = tf.reduce_mean(
                        tf.square(tf.subtract(returns, new_values, name='delta_return')),
                        name='critic_loss')

            self._critic_loss.update_state(critic_loss)
            trainable_vars = self.critic.trainable_variables
            value_grads = tape.gradient(critic_loss, trainable_vars)
            if self._max_grad_norm is not None:
                value_grads, _ = tf.clip_by_global_norm(
                    value_grads, self._max_grad_norm, name='clipped_value_grads')

            self._critic_optimizer.apply_gradients(
                zip(value_grads, trainable_vars))

    def train_step(self, data):
        x, (action_indices, returns, advantage) = data
        self.train_actor(x, action_indices, advantage)
        self.train_critic(x, returns)

        metrics = {
            'actor_loss': self._actor_loss.result(),
            'critic_loss': self._critic_loss.result()
        }

        if tf.cast(self._entropy_loss_coef, tf.bool):
            metrics['entropy_loss'] = self._entropy_loss.result()

        if tf.cast(self._regularizer_coef, tf.bool):
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
        input_shape: tuple[int, ...],
        action_per_head: tuple[int, ...],
        actor_learning_rate: float | LearningRateSchedule,
        critic_learning_rate: float | LearningRateSchedule,
        actor_layer_sizes: tuple[int, ...],
        critic_layer_sizes: tuple[int, ...],
        actor_train_iterations: int,
        critic_train_iterations: int,
        target_kl: float,
        actor_hidden_activation: str = 'relu',
        actor_head_activation: str | None = None,
        critic_hidden_activation: str = 'relu',
        clip_ratio: float | None = None,
        critic_clip_range: float | None = None,
        max_grad_norm: float | None = None,
        critic_loss_coef: float = 1.0,
        entropy_loss_coef: float = 0.0,
        effect_widths: int | tuple[int, ...] = 0,
        effect_decay_factors: float | tuple[float, ...] = 0.,
        effect_prob: float | Callable[[], Tensor] = 1.0,
        regularizer_coef: float = 0.0
    ) -> None:
        super().__init__(
            input_shape=input_shape,
            action_per_head=action_per_head,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_layer_sizes=actor_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            actor_train_iterations=actor_train_iterations,
            critic_train_iterations=critic_train_iterations,
            target_kl=target_kl,
            actor_hidden_activation=actor_hidden_activation,
            actor_head_activation=actor_head_activation,
            critic_hidden_activation=critic_hidden_activation,
            clip_ratio=clip_ratio,
            critic_clip_range=critic_clip_range,
            max_grad_norm=max_grad_norm,
            critic_loss_coef=critic_loss_coef,
            entropy_loss_coef=entropy_loss_coef,
            regularizer_coef=regularizer_coef
        )
        output_heads = len(action_per_head)
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
        if isinstance(effect_prob, float):
            probability = tf.constant(effect_prob, dtype=tf.float32, name='constant_effect_prob')

            def effect_probability() -> Tensor:
                return probability
        else:
            effect_probability = effect_prob

            self._effect_prob = effect_probability

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(dict(
            effect_widths=tuple(self._effect_widths.numpy()),
            effect_decay_factors=tuple(self._effect_decay_factors.numpy()),
        ))

        return config

    @tf.function(
        input_signature=(
            TensorSpec(shape=[None, None], dtype=tf.float32, name='x'),
            TensorSpec(shape=[None, None],
                          dtype=tf.int32, name='action_indices'),
            TensorSpec(shape=[None], dtype=tf.float32, name='advantage'),
        ),
        jit_compile=False
    )
    def train_actor(  # noqa: C901
        self, x: Tensor, action_indices: Tensor, advantage: Tensor
    ):
        print(f'tracing {self.__class__.__qualname__}.train_actor')
        action_per_head = self._action_per_head
        head_count = self._head_count
        starts = self._starts
        ends = self._ends
        # effect_prob = self._effect_prob()

        logits_concat = tf.concat(self.actor(x), axis=1, name='all_logits')

        initial_logprobs = self._logprobs_concat(
            logits_concat, starts, ends, action_indices, action_per_head, head_count)
        # initial_logprobs = tf.expand_dims(
        #     self.logprobs(
        #         logits_concat[:, starts[0]:ends[0]],  # type: ignore
        #         tf.gather(action_indices, 0, axis=1),
        #         tf.gather(self._action_per_head, 0)),
        #     axis=1)
        # for j in tf.range(one_int32, m):
        #     tf.autograph.experimental.set_loop_options(
        #         shape_invariants=(initial_logprobs, [None, None])
        #     )
        #     initial_logprobs = tf.concat([
        #         initial_logprobs,
        #         tf.expand_dims(
        #             self.logprobs(
        #                 logits_concat[:, starts[j]:ends[j]],  # type: ignore
        #                 tf.gather(action_indices, j, axis=1),
        #                 tf.gather(self._action_per_head, j)),
        #             axis=1)], axis=1)

        _advantage = tf.divide(
            advantage - tf.math.reduce_mean(advantage),
            tf.math.reduce_std(advantage) + eps,
            name='normalized_advantage')

        trainable_vars = self.actor.trainable_variables

        actor_loss = entropy_loss = regularizer_loss = kl = zero_float32
        for _ in tf.range(self._actor_train_iterations):
            total_loss = zero_float32
            with tf.GradientTape() as tape:
                logits_concat = tf.concat(
                    self.actor(x), axis=1, name='all_logits')
                for j in tf.range(head_count):
                    logits_slice = logits_concat[:, starts[j]:ends[j]]  # type: ignore
                    y_slice = tf.gather(action_indices, j, axis=1)
                    action_in_head_j = tf.gather(self._action_per_head, j)

                    j_one_hot = tf.one_hot(j, depth=head_count, dtype=tf.int32)
                    effect_width = tf.cond(
                        tf.less(tf.random.uniform([1]), self._effect_prob()),
                        lambda: tf.dynamic_partition(  # type: ignore
                            self._effect_widths, j_one_hot, 2)[1][0],
                        lambda: zero_int32
                    )
                    new_logprobs = self._logprobs_j(
                        j, logits_concat, starts, ends, action_indices,
                        self._action_per_head, False)
                    if tf.equal(effect_width, zero_int32):
                        actor_loss = self._compute_actor_loss(
                            initial_logprobs, new_logprobs, _advantage, j)
                        # ratio = tf.exp(
                        #     new_logprobs - tf.gather(initial_logprobs, j, axis=1))
                        # if self._clip_ratio is not None:
                        #     clipped_ratio = tf.clip_by_value(
                        #         ratio, 1. - self._clip_ratio, 1. + self._clip_ratio)
                        #     actor_loss = -tf.reduce_mean(
                        #         tf.minimum(
                        #             ratio * _advantage, clipped_ratio * _advantage))

                        # else:
                        #     actor_loss = -tf.reduce_mean(ratio * _advantage)
                    else:
                        for diff in tf.range(
                                tf.negative(effect_width), effect_width):
                                # tf.reduce_sum(effect_width, 1)):
                            temp = tf.add(y_slice, diff, name='y_plus_diff')
                            action_in_head_j = tf.dynamic_partition(  # type: ignore
                                action_per_head, j_one_hot, 2, name='action_in_head_j'
                            )[1][0]
                            in_range_indicator = tf.logical_and(
                                tf.greater_equal(temp, zero_int32),
                                tf.less(temp, action_in_head_j),
                                name='in_range_indicator')

                            # if not tf.reduce_all(in_range_indicator):
                            #     continue

                            in_range_indices = tf.cast(
                                in_range_indicator, tf.int32)

                            advantage_in_range = tf.dynamic_partition(  # type: ignore
                                _advantage, in_range_indices, 2,
                                name='advantage_in_range')[1]
                            y_in_range = tf.dynamic_partition(  # type: ignore
                                temp, in_range_indices, 2,
                                name='y_in_range')[1]
                            initial_logprobs_in_range = tf.dynamic_partition(  # type: ignore
                                initial_logprobs, in_range_indices, 2,
                                name='initial_logprobs_in_range')[1]

                            logits_in_range = tf.dynamic_partition(  # type: ignore
                                logits_slice, in_range_indices, 2,
                                name='logits_in_range')[1]
                            new_logprobs_in_range = logprobs(
                                logits_in_range, y_in_range, action_in_head_j)

                            abs_diff = tf.cast(tf.abs(diff), dtype=tf.float32, name='abs_diff')
                            effect_decay = tf.dynamic_partition(  # type: ignore
                                self._effect_decay_factors, j_one_hot, 2,
                                name='effect_decay')[1][0]
                            effect = tf.pow(effect_decay, abs_diff)
                            ratio = tf.exp(
                                tf.subtract(new_logprobs_in_range, tf.gather(
                                    initial_logprobs_in_range, j, axis=1)))
                            if self._clip_ratio is None:
                                _loss = -tf.reduce_mean(
                                    tf.multiply(ratio, advantage_in_range),
                                    name=f'actor_loss_head_j')
                            else:
                                clipped_ratio = tf.clip_by_value(
                                    ratio,
                                    tf.subtract(one_float32, self._clip_ratio),
                                    tf.add(one_float32, self._clip_ratio),
                                    name='clipped_ratio')
                                _loss = -tf.reduce_mean(
                                    tf.minimum(
                                        tf.multiply(
                                            ratio, advantage_in_range,
                                            name='ratio_times_adv_in_range'),
                                        tf.multiply(clipped_ratio, advantage_in_range,
                                                    name='clipped_ratio_times_adv_in_range')),
                                    name=f'actor_loss_clipped_head_j')

                            actor_loss = tf.add_n(
                                [actor_loss, tf.multiply(effect, _loss)],
                                name='actor_loss')

                    if tf.cast(self._entropy_loss_coef, tf.bool):
                        entropy_loss = entropy(new_logprobs)
                        entropy_loss.set_shape([])

                    if tf.cast(self._regularizer_coef, tf.bool):
                        regularizer_loss = self._compute_regularizer_loss()

                    total_loss = tf.add_n(
                        [
                            total_loss,
                            actor_loss,
                            tf.multiply(self._entropy_loss_coef, entropy_loss),
                            tf.multiply(self._regularizer_coef, regularizer_loss)
                        ],
                        name='total_loss'
                    )

            policy_grads = tape.gradient(total_loss, trainable_vars)
            if self._max_grad_norm is not None:
                policy_grads, _ = tf.clip_by_global_norm(
                    policy_grads, self._max_grad_norm)
            self._actor_optimizer.apply_gradients(
                zip(policy_grads, trainable_vars))

            logits_concat = tf.concat(self.actor(x), axis=1, name='all_logits')
            new_logprobs = self._logprobs_concat(
                logits_concat, starts, ends, action_indices, action_per_head, head_count)

            kl = .5 * tf.reduce_mean(
                tf.square(tf.subtract(new_logprobs, initial_logprobs, name='delta_logprobs')),
                name='kl')

            if tf.greater(kl, self._1_5_target_kl):  # Early Stopping
                break

        self._kl.update_state(kl)
        self._actor_loss.update_state(actor_loss)
        if tf.cast(self._entropy_loss_coef, tf.bool):
            self._entropy_loss.update_state(entropy_loss)
        if tf.cast(self._regularizer_coef, tf.bool):
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
            self, X: tuple[FeatureSet, ...], training: bool | None = None
    ) -> tuple[ACLabelType, ...]:
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
            self, X: tuple[FeatureSet, ...],
            Y: tuple[ACLabelType, ...]) -> dict[str, float]:
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
        action_index: Tensor = tf.convert_to_tensor(action_index_temp)
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
