# -*- coding: utf-8 -*-
'''
PPOTandemModel class
====================

'''
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import tensorflow as tf
from tensorflow import Tensor, TensorShape, TensorSpec

from reil.utils.tf_utils import (JIT_COMPILE, MeanMetric,
                                 SparseCategoricalAccuracyMetric,
                                 TF2UtilsMixin, entropy, logprobs, reset_metric)

keras = tf.keras

from keras.optimizers.schedules import \
    LearningRateSchedule  # noqa: E402
from keras.optimizers import Adam  # noqa: E402

ACLabelType = tuple[tuple[tuple[int, ...], ...], float]

eps: Tensor = tf.constant(np.finfo(np.float32).eps.item(), dtype=tf.float32)
zero_int32: Tensor = tf.constant(0, tf.int32)
one_int32: Tensor = tf.constant(1, tf.int32)
zero_float32: Tensor = tf.constant(0., tf.float32)
one_float32: Tensor = tf.constant(1., tf.float32)


@tf.function(jit_compile=JIT_COMPILE)
def _less_than_condition(j: Tensor, m: Tensor, *rest) -> Tensor:
    return tf.less(j, m, name='less_than')  # type: ignore


@keras.utils.register_keras_serializable(package='reil.learners.ppo_learner_tandem')
class PPOTandemModel(TF2UtilsMixin):
    def __init__(
            self,
            input_shape: tuple[int, ...],
            action_per_head: tuple[int, ...],
            actor_learning_rate: float | LearningRateSchedule,
            critic_learning_rate: float | LearningRateSchedule,
            actor_layer_sizes: dict[str, tuple[int, ...]],
            critic_layer_sizes: tuple[int, ...],
            actor_train_iterations: int,
            critic_train_iterations: int,
            target_kl: float,
            training_switch: dict[str, int] | None = None,
            backprop_mode: Literal['separate', 'shared', 'all'] = 'all',
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
        self._action_per_head_units = tuple(action_per_head)
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

        self._training_switch = training_switch
        self._training_counter: int = 0
        if training_switch is not None:
            self._training_sequence = list(training_switch)
            self._current_switch = len(self._training_sequence)

        self._backprop_mode: Literal['separate', 'shared', 'all'] = backprop_mode
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

        self._build_networks()

        self._training_switch = training_switch
        if training_switch is not None:
            self._freeze_layers()

        optimizer = keras.optimizers.Adam if self._training_switch is None else Adam
        self._actor_optimizer = optimizer(learning_rate=self._actor_learning_rate)  # type: ignore
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

    def _build_networks(self):
        input_: Tensor = keras.Input(self._input_shape)  # type: ignore
        logits = TF2UtilsMixin.mlp_functional_w_concat(
            input_=input_, layer_sizes=self._actor_layer_sizes,
            activation=self._actor_hidden_activation,
            layer_name_format='actor_{i:0>2}',
            action_per_head=self._action_per_head_units,
            head_activation=self._actor_head_activation,
            output_name_format='actor_output_{i:0>2}',
            backprop_mode=self._backprop_mode,
            normalize_before_concat='batch')

        self.actor = keras.Model(
            inputs=input_,
            outputs=logits if len(logits) > 1 else tuple(logits))

        critic_layers = TF2UtilsMixin.mlp_functional(
            input_, self._critic_layer_sizes,
            self._critic_hidden_activation, 'critic_{i:0>2}')
        critic_output = keras.layers.Dense(
            1, name='critic_output')(critic_layers)
        self.critic = keras.Model(inputs=input_, outputs=critic_output)

    def __call__(self, inputs, training: bool | None = None) -> Any:
        logits = self.actor(inputs, training=training)
        values = self.critic(inputs, training=training)

        return logits, values

    def _actor_logits_concat(self, x: Tensor, action_indices: Tensor) -> Tensor:
        '''Concatenated per-head actor logits for `x`.

        Default (soft-coupling) tandem: a single forward of `self.actor(x)`
        whose duration head reads the dose LOGITS (see `_build_networks` /
        `mlp_functional_w_concat`). `action_indices` is unused here but is part
        of the contract so `PPOTandemConditionalModel` can override this to
        condition the duration head on the *taken* dose instead. Called at all
        forward sites in `train_actor` so the override applies to initial, per-
        iteration, and post-update log-probs alike (PPO ratio stays consistent).
        '''
        return tf.concat(self.actor(x), axis=1, name='all_logits')

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
                    PPOTandemModel._logprobs_j(
                        j, logits_concat, starts, ends,
                        action_indices, action_per_head)],
                    axis=1)
            ]
        result: Tensor = PPOTandemModel._logprobs_j(  # type: ignore
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

        logits_concat = self._actor_logits_concat(x, action_indices)

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

        advantage_ = tf.divide(
            advantage - tf.math.reduce_mean(advantage),
            tf.math.reduce_std(advantage) + eps,
            name='normalized_advantage')

        trainable_vars = self.actor.trainable_variables

        actor_loss = entropy_loss = regularizer_loss = kl = zero_float32
        for _ in tf.range(self._actor_train_iterations):
            total_loss = zero_float32
            with tf.GradientTape() as tape:
                logits_concat = self._actor_logits_concat(x, action_indices)
                for j in tf.range(head_count):
                    new_logprobs_j = self._logprobs_j(
                        j, logits_concat, starts, ends, action_indices,
                        self._action_per_head, False)
                    # new_logprobs = self.logprobs(
                    #     logits_concat[:, starts[j]:ends[j]],  # type: ignore
                    #     tf.gather(action_indices, j, axis=1),
                    #     tf.gather(self._action_per_head, j))

                    actor_loss = self._compute_actor_loss(
                        initial_logprobs, new_logprobs_j, advantage_, j)

                    if tf.cast(self._entropy_loss_coef, tf.bool):
                        entropy_loss = entropy(new_logprobs_j)
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

            logits_concat = self._actor_logits_concat(x, action_indices)

            new_logprobs = self._logprobs_concat(
                logits_concat, starts, ends, action_indices, action_per_head, head_count)

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

    def per_neuron_l2_vector(self) -> Tensor:
        # Per-output-neuron L2 norm of (weights + bias) for every actor head,
        # concatenated across heads. Length == sum(action_per_head) (e.g. 7+6
        # = 13 for dose+duration). This is the tandem counterpart of
        # PPOModel.per_neuron_l2_vector: that single-head version reduces the
        # one final layer, whereas the tandem actor exposes one Dense output
        # head per action component. The heads are named 'actor_output_<nn>'
        # (see _build_networks / mlp_functional_w_concat); sorting by name
        # yields action_per_head order (head 00/01 -> dose, then duration),
        # matching the logits-concat order used in train_actor and the
        # PPOLearner.learn diagnostics. Eager-safe; called once per train_step
        # like the single-head version.
        heads = sorted(
            (layer for layer in self.actor.layers
             if layer.name.startswith('actor_output_')),
            key=lambda layer: layer.name)
        vectors = []
        for head in heads:
            weights_concat = tf.concat([
                head.weights[0],
                tf.expand_dims(head.weights[1], axis=0)
            ], axis=0)
            vectors.append(
                tf.math.reduce_euclidean_norm(weights_concat, axis=0))

        return tf.concat(vectors, axis=0)

    @tf.function(jit_compile=JIT_COMPILE)
    def _compute_actor_loss(self, initial_logprobs, new_logprobs, advantage_, j):
        ratio: Tensor = tf.exp(
            tf.subtract(
                new_logprobs, tf.gather(initial_logprobs, j, axis=1),
                name='delta_logprobs'),
            name='ratio'
        )
        if self._clip_ratio is None:
            actor_loss = -tf.reduce_mean(
                tf.multiply(ratio, advantage_), name='actor_loss')
        else:
            clipped_ratio = tf.clip_by_value(
                ratio,
                tf.subtract(one_float32, self._clip_ratio),
                tf.add(one_float32, self._clip_ratio),
                name='clipped_ratio')
            actor_loss = -tf.reduce_mean(
                tf.minimum(
                    tf.multiply(ratio, advantage_, name='ratio_times_adv'),
                    tf.multiply(clipped_ratio, advantage_,
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

    def _freeze_layers(self):
        self._training_counter = 0
        self._current_switch += 1
        if self._current_switch >= len(self._training_sequence):
            self._current_switch = 0
        part = self._training_sequence[self._current_switch]
        for layer in self.actor.layers:
            layer.trainable = (part in layer.name) or (part == 'all')

    def train_step(self, data):
        x, (action_indices, returns, advantage) = data
        self._training_counter += 1
        if self._training_switch is not None and (
                self._training_counter >= self._training_switch[
                    self._training_sequence[self._current_switch]]):
            self._freeze_layers()
            print({layer.name: layer.trainable for layer in self.actor.layers})
            # self._actor_optimizer.build(self.actor.trainable_variables)

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

        reset_metric(self._actor_loss)
        reset_metric(self._critic_loss)
        reset_metric(self._entropy_loss)
        reset_metric(self._actor_accuracy)

        return metrics


@keras.utils.register_keras_serializable(
    package='reil.learners.ppo_learner_tandem')
class PPOTandemConditionalModel(PPOTandemModel):
    '''Tandem where the duration head conditions on the PRESCRIBED (taken) dose
    value instead of the soft dose logits (Paper-3 Axis-A, user 2026-07-05).

    ``dose_conditioning``:
      ``'pct_change'`` — duration signal = the sampled % dose change ``p``.
      ``'new_dose'``   — signal = ``last_dose * (1 + p)`` where ``last_dose`` is
                         de-normalised from the state dose feature over
                         ``dose_range``. Mirrors the subject's real new-dose
                         formula (``warfarin.py`` ``last_dose * (1 + p)``).

    The actor is one 2-input Keras model ``[state, dose_signal] ->
    (dose_logits, dur_logits)``; the dose branch ignores the signal. During
    training the signal is recomputed IN-GRAPH from the taken dose index
    (``action_indices[:, 0]``) via the ``_actor_logits_concat`` override, so no
    rollout data is carried through the buffer and the PPO ratio stays
    consistent (old + new duration log-probs condition on the same taken dose).
    During rollout the agent samples the dose from ``act_dose_logits`` then the
    duration from ``act_duration_logits`` (two-stage).
    '''

    def __init__(
            self, *args,
            dose_conditioning: Literal['pct_change', 'new_dose'] = 'pct_change',
            dose_values: tuple[float, ...] = (),
            dose_feature_index: int = 0,
            dose_range: tuple[float, float] = (0.0, 15.0),
            **kwargs: Any) -> None:
        self._dose_conditioning = dose_conditioning
        self._dose_values_list = tuple(float(v) for v in dose_values)
        self._dose_feature_index = int(dose_feature_index)
        self._dose_range = (float(dose_range[0]), float(dose_range[1]))
        super().__init__(*args, **kwargs)  # calls _build_networks (overridden)
        self._dose_values_t: Tensor = tf.constant(
            self._dose_values_list, dtype=tf.float32, name='dose_values')
        self._dose_lo: Tensor = tf.constant(
            self._dose_range[0], dtype=tf.float32, name='dose_lo')
        self._dose_hi: Tensor = tf.constant(
            self._dose_range[1], dtype=tf.float32, name='dose_hi')

    def _build_networks(self):
        input_: Tensor = keras.Input(self._input_shape, name='state')  # type: ignore
        signal_in: Tensor = keras.Input((1,), name='dose_signal')  # type: ignore

        dose_trunk = TF2UtilsMixin.mlp_functional(
            input_, self._actor_layer_sizes['dose'],
            self._actor_hidden_activation, 'actor__dose_{i:0>2}')
        dose_logits = keras.layers.Dense(
            self._action_per_head_units[0],
            activation=self._actor_head_activation,
            name='actor_output_00')(dose_trunk)

        dur_input = keras.layers.Concatenate(axis=-1)([input_, signal_in])
        dur_trunk = TF2UtilsMixin.mlp_functional(
            dur_input, self._actor_layer_sizes['duration'],
            self._actor_hidden_activation, 'actor__duration_{i:0>2}')
        dur_logits = keras.layers.Dense(
            self._action_per_head_units[1],
            activation=self._actor_head_activation,
            name='actor_output_01')(dur_trunk)

        self.actor = keras.Model(
            inputs=[input_, signal_in], outputs=[dose_logits, dur_logits])

        critic_layers = TF2UtilsMixin.mlp_functional(
            input_, self._critic_layer_sizes,
            self._critic_hidden_activation, 'critic_{i:0>2}')
        critic_output = keras.layers.Dense(1, name='critic_output')(critic_layers)
        self.critic = keras.Model(inputs=input_, outputs=critic_output)

    def _dose_signal(self, x: Tensor, dose_idx: Tensor) -> Tensor:
        '''dose_idx: int tensor [batch] -> signal [batch, 1] float.'''
        p = tf.gather(self._dose_values_t, dose_idx)
        if self._dose_conditioning == 'new_dose':
            prev = (x[:, self._dose_feature_index]
                    * (self._dose_hi - self._dose_lo) + self._dose_lo)
            signal = prev * (one_float32 + p)
        else:  # pct_change
            signal = p
        return tf.expand_dims(signal, axis=1)

    def _actor_logits_concat(self, x: Tensor, action_indices: Tensor) -> Tensor:
        signal = self._dose_signal(x, action_indices[:, 0])
        return tf.concat(self.actor([x, signal]), axis=1, name='all_logits')

    def __call__(self, inputs, training: bool | None = None) -> Any:
        # Diagnostics / critic-value path (no taken dose here): forward with a
        # zero dose signal. The duration logits from this path are used only for
        # metrics; the agent's two-stage act() supplies the real conditioning.
        zeros = tf.zeros((tf.shape(inputs)[0], 1), dtype=tf.float32)
        logits = self.actor([inputs, zeros], training=training)
        values = self.critic(inputs, training=training)
        return logits, values

    def act_dose_logits(self, x: Tensor) -> Tensor:
        '''Dose-head logits for rollout (dose branch ignores the signal).'''
        zeros = tf.zeros((tf.shape(x)[0], 1), dtype=tf.float32)
        return self.actor([x, zeros])[0]

    def act_duration_logits(self, x: Tensor, dose_idx: Tensor) -> Tensor:
        '''Duration-head logits conditioned on the just-sampled dose index.'''
        signal = self._dose_signal(x, dose_idx)
        return self.actor([x, signal])[1]
