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

from reil.utils.tf_utils import (JIT_COMPILE, GradientGate, MeanMetric,
                                 SparseCategoricalAccuracyMetric,
                                 TF2UtilsMixin, entropy, logprobs, reset_metric)

keras = tf.keras

from keras.optimizers.schedules import \
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
            training_schedule: dict[str, int] | None = None,
            coupling_gradient: Literal['full', 'blocked', 'gated'] = 'full',
            head_loss_weights: tuple[float, ...] | None = None,
            actor_hidden_activation: str = 'relu',
            actor_head_activation: str | None = None,
            critic_hidden_activation: str = 'relu',
            clip_ratio: float | None = None,
            critic_clip_range: float | None = None,
            max_grad_norm: float | None = None,
            critic_loss_coef: float = 1.0,
            entropy_loss_coef: float = 0.0,
            regularizer_coef: float = 0.0,
            per_head_advantage: bool = False,
            separate_critics: bool = False) -> None:

        super().__init__(models={})

        # per_head_advantage (Paper-3 JA-2): when True, each actor head is
        # trained on its OWN reward/advantage stream and the critic outputs one
        # value per head (dose value, duration value). The agent
        # (PPO4WarfarinTandemPerHeadAgent) supplies returns/advantage as
        # [batch, head_count] tensors and train_step routes them through the
        # *_per_head train methods. Default False -> single shared advantage +
        # scalar critic == the original tandem behaviour, byte-for-byte.
        self._per_head_advantage = bool(per_head_advantage)
        self._critic_output_dim = (
            len(action_per_head) if self._per_head_advantage else 1)
        # separate_critics (2026-07-11): build ONE independent critic body per
        # head (fully separate value networks) instead of a single shared body
        # with a per-head output layer. Only the input state vector is shared;
        # each head's value function has its own hidden stack. Requires
        # per_head_advantage (per-head returns/values). self.critic keeps the
        # same input->[batch, head_count] signature (the per-head 1-D outputs are
        # concatenated), so training/agent/serialization are unchanged.
        self._separate_critics = bool(separate_critics)
        if self._separate_critics and not self._per_head_advantage:
            raise ValueError(
                'separate_critics=True requires per_head_advantage=True '
                '(separate value networks need per-head returns).')
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

        # Training-design axes (2026-07-06 restart, user spec; Ch. 3 is a
        # guiding document only — no back-compat with the retired
        # backprop_mode/training_switch vocabulary):
        #
        # coupling_gradient — gradient routing across the section-to-section
        #   coupling in mlp_functional_w_concat. Sections always train on
        #   their own head's loss; a trunk (optional 'trunk' entry in
        #   actor_layer_sizes) always trains from every head:
        #   'full' (was 'all'), 'blocked' (was 'shared'),
        #   'gated' (was 'separate'; schedule-driven, == 'blocked' without a
        #   training_schedule).
        #
        # head_loss_weights — per-head multiplier on the PPO + entropy loss
        #   (None = all 1.0). E.g. (0.0, 1.0) with coupling_gradient='full'
        #   = "downstream-only": everything trains solely through the last
        #   head's loss (dissertation §3.2.2's 'shared', now expressible).
        #
        # training_schedule — alternating-freeze phases {name: train steps},
        #   cycled in key order (was training_switch, which never worked:
        #   layer.trainable toggles cannot reach the once-traced train_actor).
        #   Phase names: 'all', 'trunk', 'heads', a section name, or any
        #   layer-name substring. See _apply_training_phase.
        if coupling_gradient not in ('full', 'blocked', 'gated'):
            raise ValueError(
                f'Unknown coupling_gradient {coupling_gradient!r}. Use '
                "'full' (was backprop_mode 'all'), 'blocked' (was 'shared') "
                "or 'gated' (was 'separate').")
        self._coupling_gradient: Literal['full', 'blocked', 'gated'] = \
            coupling_gradient

        if head_loss_weights is None:
            head_loss_weights = tuple(1.0 for _ in action_per_head)
        if len(head_loss_weights) != len(action_per_head):
            raise ValueError(
                f'head_loss_weights needs one weight per head: got '
                f'{head_loss_weights} for {len(action_per_head)} heads.')
        self._head_loss_weights = tuple(float(w) for w in head_loss_weights)
        self._head_loss_weights_t: Tensor = tf.constant(
            self._head_loss_weights, dtype=tf.float32,
            name='head_loss_weights')

        self._training_schedule = training_schedule
        self._training_counter: int = 0
        if training_schedule is not None:
            self._training_sequence = list(training_schedule)
            self._current_phase = len(self._training_sequence)
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

        self._init_training_schedule_state()
        if training_schedule is not None:
            self._advance_training_phase()
        elif coupling_gradient == 'gated':
            self._logger.warning(
                "coupling_gradient='gated' without a training_schedule "
                "behaves exactly like 'blocked' (the gate stays closed).")

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

    def _make_critic(self, input_: Tensor) -> keras.Model:
        '''Build the critic network.

        Default: a single shared body with a `critic_output_dim`-wide output
        layer (one shared value net; per-head values are just output columns).

        `separate_critics`: one INDEPENDENT body per head — only `input_` is
        shared; each head's value function has its own hidden stack. The 1-D
        outputs are concatenated so `self.critic(x) -> [batch, head_count]`
        keeps the same signature (train_critic_per_head / the agent / saving
        are unchanged).
        '''
        if self._separate_critics and self._critic_output_dim > 1:
            columns = []
            for h in range(self._critic_output_dim):
                body = TF2UtilsMixin.mlp_functional(
                    input_, self._critic_layer_sizes,
                    self._critic_hidden_activation, f'critic_h{h}_{{i:0>2}}')
                columns.append(
                    keras.layers.Dense(1, name=f'critic_output_h{h}')(body))
            critic_output = keras.layers.Concatenate(
                axis=-1, name='critic_output')(columns)
        else:
            critic_layers = TF2UtilsMixin.mlp_functional(
                input_, self._critic_layer_sizes,
                self._critic_hidden_activation, 'critic_{i:0>2}')
            critic_output = keras.layers.Dense(
                self._critic_output_dim, name='critic_output')(critic_layers)
        return keras.Model(inputs=input_, outputs=critic_output)

    def _build_networks(self):
        input_: Tensor = keras.Input(self._input_shape)  # type: ignore
        logits = TF2UtilsMixin.mlp_functional_w_concat(
            input_=input_, layer_sizes=self._actor_layer_sizes,
            activation=self._actor_hidden_activation,
            layer_name_format='actor_{i:0>2}',
            action_per_head=self._action_per_head_units,
            head_activation=self._actor_head_activation,
            output_name_format='actor_{name}_output',
            coupling_gradient=self._coupling_gradient,
            normalize_before_concat='layer')
        # Head layers in action_per_head order — the single source of truth
        # for per_neuron_l2_vector, the regularizer, and 'heads' phases.
        self._head_layer_names = [
            f'actor_{name}_output'
            for name in self._actor_layer_sizes if name != 'trunk']

        self.actor = keras.Model(
            inputs=input_,
            outputs=logits if len(logits) > 1 else tuple(logits))

        self.critic = self._make_critic(input_)

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
                        # Entropy of head j's FULL action distribution (not the
                        # taken-action log-prob). Subtracted from the loss below
                        # so a higher coef ENCOURAGES exploration — the standard
                        # PPO entropy bonus. (Pre-2026-07-06 this called
                        # `entropy(new_logprobs_j)` on the scalar taken-action
                        # log-prob and ADDED it, so entropy reg was a no-op /
                        # mild collapse pressure — verified: entropy still
                        # crashed to 0 at coef 0.05.)
                        entropy_loss = tf.reduce_mean(
                            entropy(logits_concat[:, starts[j]:ends[j]]))
                        entropy_loss.set_shape([])

                    # head_loss_weights scales head j's PPO + entropy terms
                    # (weight 0 removes the head's own losses entirely; with
                    # coupling_gradient='full' the head still trains through
                    # downstream losses — the "downstream-only" regime).
                    total_loss = tf.add(
                        total_loss,
                        tf.multiply(
                            tf.gather(self._head_loss_weights_t, j),
                            tf.add(
                                actor_loss,
                                tf.multiply(
                                    tf.negative(self._entropy_loss_coef),
                                    entropy_loss))),
                        name='total_loss')

                if tf.cast(self._regularizer_coef, tf.bool):
                    # Dose-head group-lasso, added once per iteration outside
                    # the head loop (pre-restart it was added once per head,
                    # scaling the coefficient by head_count). Duration is not
                    # regularised — see _compute_regularizer_loss.
                    regularizer_loss = self._compute_regularizer_loss()
                    total_loss = tf.add(
                        total_loss,
                        tf.multiply(
                            self._regularizer_coef, regularizer_loss),
                        name='total_loss_w_regularizer')

            policy_grads = tape.gradient(total_loss, trainable_vars)
            if self._grad_masks is not None:
                # training_schedule freeze: masks are tf.Variables captured by
                # this (single) trace; `_apply_training_phase` re-assigns them
                # eagerly, so phase changes take effect without a retrace.
                policy_grads = [
                    None if g is None else tf.multiply(g, m)
                    for g, m in zip(policy_grads, self._grad_masks)]
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
        # Group-lasso (per-output-neuron L2 of weights+bias) on the DOSE head
        # ONLY. The regularizer is the action-forging / dose-table
        # explainability mechanism carried over from Paper 2: it drives whole
        # DOSE output neurons to zero to eliminate dose actions. The duration
        # head is DELIBERATELY NOT regularized — Paper 3 does not forge or
        # eliminate durations — so `regularizer_coef` never touches its
        # weights. Consequence for diagnostics: the duration slice of
        # per_neuron_l2_vector stays ~constant and inflates n_actions_alive by
        # a fixed offset (see per_neuron_l2_vector). dose head is head 0
        # (`_head_layer_names[0]`; the tandem is dose-first).
        # (Pre-restart this read `actor.layers[-1]` — whichever head was
        # topologically last, i.e. usually duration — an unintended target;
        # a brief all-heads variant on 2026-07-06 was likewise wrong.)
        head = self.actor.get_layer(self._head_layer_names[0])
        weights_concat = tf.concat([
            head.weights[0],
            tf.expand_dims(head.weights[1], axis=0)
        ], axis=0, name='dose_head_weights')
        return tf.reduce_sum(
            tf.math.reduce_euclidean_norm(weights_concat, axis=0),
            name='regularizer_loss')

    def per_neuron_l2_vector(self) -> Tensor:
        # Per-output-neuron L2 norm of (weights + bias) for every actor head,
        # concatenated across heads in `_head_layer_names` order (==
        # action_per_head order == the logits-concat order in train_actor).
        # Length == sum(action_per_head) (e.g. 7+6 = 13 for dose+duration).
        # Tandem counterpart of PPOModel.per_neuron_l2_vector. Eager-safe;
        # called once per train_step by PPOLearner.learn diagnostics.
        #
        # ANALYSIS CAVEAT (action forging is DOSE-only): only the first
        # `action_per_head[0]` entries — the DOSE slice — carry the
        # action-elimination signal, because `_compute_regularizer_loss`
        # penalises the dose head alone. The trailing duration entries are
        # NEVER regularised: their norms stay ~constant, effectively never
        # cross the 1e-4 "dead" threshold, and so add a CONSTANT offset of
        # `action_per_head[1]` (e.g. +6) to `n_actions_alive`. For any
        # dose-action-count analysis, slice `[:action_per_head[0]]` and treat
        # the duration tail as a fixed baseline, not a learned quantity.
        vectors = []
        for name in self._head_layer_names:
            head = self.actor.get_layer(name)
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

    @tf.function(
        input_signature=(
            TensorSpec(shape=[None, None], dtype=tf.float32, name='x'),
            TensorSpec(shape=[None, None], dtype=tf.int32, name='action_indices'),
            TensorSpec(shape=[None, None], dtype=tf.float32, name='advantage'),
        ),
        jit_compile=False
    )
    def train_actor_per_head(self, x, action_indices, advantage):  # noqa: C901
        # Per-head twin of train_actor (JA-2). Identical body EXCEPT: advantage
        # arrives as [batch, head_count], is normalised per-column (each head's
        # advantage standardised on its own, so the small-variance duration
        # signal is not swamped by the dose control variance), and head j is
        # trained on advantage column j. Kept as a separate traced fn so the
        # single-advantage train_actor above is untouched (fixed input sig).
        print(f'tracing {self.__class__.__qualname__}.train_actor_per_head')
        action_per_head = self._action_per_head
        head_count = self._head_count
        starts = self._starts
        ends = self._ends

        logits_concat = self._actor_logits_concat(x, action_indices)
        initial_logprobs = self._logprobs_concat(
            logits_concat, starts, ends, action_indices, action_per_head, head_count)

        advantage_ = tf.divide(
            advantage - tf.math.reduce_mean(advantage, axis=0),
            tf.math.reduce_std(advantage, axis=0) + eps,
            name='normalized_advantage_per_head')

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

                    actor_loss = self._compute_actor_loss(
                        initial_logprobs, new_logprobs_j,
                        tf.gather(advantage_, j, axis=1), j)

                    if tf.cast(self._entropy_loss_coef, tf.bool):
                        entropy_loss = tf.reduce_mean(
                            entropy(logits_concat[:, starts[j]:ends[j]]))
                        entropy_loss.set_shape([])

                    total_loss = tf.add(
                        total_loss,
                        tf.multiply(
                            tf.gather(self._head_loss_weights_t, j),
                            tf.add(
                                actor_loss,
                                tf.multiply(
                                    tf.negative(self._entropy_loss_coef),
                                    entropy_loss))),
                        name='total_loss')

                if tf.cast(self._regularizer_coef, tf.bool):
                    regularizer_loss = self._compute_regularizer_loss()
                    total_loss = tf.add(
                        total_loss,
                        tf.multiply(
                            self._regularizer_coef, regularizer_loss),
                        name='total_loss_w_regularizer')

            policy_grads = tape.gradient(total_loss, trainable_vars)
            if self._grad_masks is not None:
                policy_grads = [
                    None if g is None else tf.multiply(g, m)
                    for g, m in zip(policy_grads, self._grad_masks)]
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
        self._actor_loss.update_state(actor_loss)
        if tf.cast(self._entropy_loss_coef, tf.bool):
            self._entropy_loss.update_state(entropy_loss)
        if tf.cast(self._regularizer_coef, tf.bool):
            self._regularizer_loss.update_state(regularizer_loss)

    @tf.function(
        input_signature=(
            TensorSpec(shape=[None, None], dtype=tf.float32, name='x'),
            TensorSpec(shape=[None, None], dtype=tf.float32, name='returns'),
        ),
        jit_compile=JIT_COMPILE
    )
    def train_critic_per_head(self, x, returns):
        # Per-head twin of train_critic. The critic outputs head_count values
        # (self._critic_output_dim); returns is [batch, head_count]; the MSE is
        # element-wise over both dims (column j fits head j's return stream).
        print(f'tracing {self.__class__.__qualname__}.train_critic_per_head')
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

    def _init_training_schedule_state(self) -> None:
        '''(Re)create the runtime freeze state for `training_schedule`.

        One non-trainable 0/1 mask variable per actor trainable variable.
        `train_actor` is a `tf.function` traced ONCE (fixed input signature),
        so toggling `layer.trainable` can never reach the traced graph — the
        retired training_switch silently did nothing after the first trace.
        Mask variables are captured by the trace and re-assigned eagerly on a
        phase change, so what trains changes at runtime without a retrace and
        without disturbing the optimizer's variable list.

        Not pickled — `__setstate__` rebuilds this against the reloaded actor.
        '''
        self._has_trunk = any(
            '_trunk_' in layer.name for layer in self.actor.layers)
        if self._training_schedule is None:
            self._grad_masks: list[tf.Variable] | None = None
            self._mask_index: dict[Any, int] = {}
            self._gate_layers: list[GradientGate] = []
            return

        trainable_vars = self.actor.trainable_variables
        self._grad_masks = [
            tf.Variable(1.0, trainable=False, dtype=tf.float32,
                        name=f'grad_mask_{i:03d}')
            for i in range(len(trainable_vars))]
        # Keyed by id(): Keras-3 Variables have no .ref(); the map only ever
        # addresses these exact live objects and is rebuilt on load.
        self._mask_index = {
            id(v): i for i, v in enumerate(trainable_vars)}
        self._gate_layers = [
            layer for layer in self.actor.layers
            if isinstance(layer, GradientGate)]

    def _advance_training_phase(self):
        self._training_counter = 0
        self._current_phase += 1
        if self._current_phase >= len(self._training_sequence):
            self._current_phase = 0
        self._apply_training_phase()

    def _phase_trains_layer(self, phase: str, layer_name: str) -> bool:
        '''Whether `phase` (a `training_schedule` key) trains `layer_name`.

        - `'all'`     — every layer.
        - `'heads'`   — the Dense output heads (`'_output'` in the name).
        - `'trunk'`   — the shared trunk (`'_trunk_'` layers) when one
                        exists; without a trunk, every non-head layer (the
                        hidden stacks — the de-facto trunk).
        - otherwise   — substring match; sections are addressable by name
                        because heads carry their section ('dose' matches
                        'actor__dose_01' AND 'actor_dose_output').
        '''
        if phase == 'all':
            return True
        if phase == 'heads':
            return '_output' in layer_name
        if phase == 'trunk':
            if self._has_trunk:
                return '_trunk_' in layer_name
            return '_output' not in layer_name
        return phase in layer_name

    def _apply_training_phase(self) -> None:
        '''Set gradient masks (and coupling gates) for the current phase.

        Gates ('gated' coupling only) OPEN during `'all'` and `'trunk'`
        phases — the trunk absorbs every head's advantage, including through
        the coupling — and stay closed in every other phase (own-loss
        training). Frozen == zero gradient; with Adam, a group that trained
        in the previous phase keeps a decaying momentum tail for a few steps
        (~0.9^t); a group frozen from the start never moves.
        '''
        part = self._training_sequence[self._current_phase]
        assert self._grad_masks is not None
        matched = 0
        for layer in self.actor.layers:
            flag = 1.0 if self._phase_trains_layer(part, layer.name) else 0.0
            for weight in layer.trainable_weights:
                self._grad_masks[self._mask_index[id(weight)]].assign(flag)
                matched += flag != 0.0
        if not matched:
            self._logger.warning(
                f'training_schedule phase {part!r} matches no trainable '
                'actor weights — nothing will train until the next phase.')

        gate_value = 1.0 if part in ('all', 'trunk') else 0.0
        for gate_layer in self._gate_layers:
            gate_layer.gate.assign(gate_value)

    def __getstate__(self):
        state = super().__getstate__()
        # Runtime handles into the live actor graph — rebuilt on load.
        for key in ('_grad_masks', '_mask_index', '_gate_layers'):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._init_training_schedule_state()
        if self._training_schedule is not None:
            self._apply_training_phase()

    def train_step(self, data):
        x, (action_indices, returns, advantage) = data
        self._training_counter += 1
        if self._training_schedule is not None and (
                self._training_counter >= self._training_schedule[
                    self._training_sequence[self._current_phase]]):
            self._advance_training_phase()
            print(
                'training schedule -> phase '
                f'{self._training_sequence[self._current_phase]!r}')

        if self._per_head_advantage:
            # advantage / returns arrive as [batch, head_count]; each head
            # trains on its own column (dose -> control reward, duration ->
            # burden+safety reward).
            self.train_actor_per_head(x, action_indices, advantage)
            self.train_critic_per_head(x, returns)
        else:
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

        # Optional shared trunk: z replaces the raw state as both sections'
        # input. The dose signal stays derived from the RAW state (the
        # normalised dose feature lives there, not in z).
        z: Tensor = input_
        trunk_sizes = self._actor_layer_sizes.get('trunk')
        if trunk_sizes:
            z = TF2UtilsMixin.mlp_functional(
                input_, tuple(trunk_sizes),
                self._actor_hidden_activation, 'actor__trunk_{i:0>2}')

        dose_stack = TF2UtilsMixin.mlp_functional(
            z, self._actor_layer_sizes['dose'],
            self._actor_hidden_activation, 'actor__dose_{i:0>2}')
        dose_logits = keras.layers.Dense(
            self._action_per_head_units[0],
            activation=self._actor_head_activation,
            name='actor_dose_output')(dose_stack)

        dur_input = keras.layers.Concatenate(axis=-1)([z, signal_in])
        dur_stack = TF2UtilsMixin.mlp_functional(
            dur_input, self._actor_layer_sizes['duration'],
            self._actor_hidden_activation, 'actor__duration_{i:0>2}')
        dur_logits = keras.layers.Dense(
            self._action_per_head_units[1],
            activation=self._actor_head_activation,
            name='actor_duration_output')(dur_stack)

        self._head_layer_names = ['actor_dose_output', 'actor_duration_output']
        self.actor = keras.Model(
            inputs=[input_, signal_in], outputs=[dose_logits, dur_logits])

        self.critic = self._make_critic(input_)

    def _dose_signal(self, x: Tensor, dose_idx: Tensor) -> Tensor:
        '''dose_idx: int tensor [batch] -> signal [batch, 1] float.

        The signal is kept on the SAME normalised scale as the state features
        (min-max over dose_range -> ~[0, 1]). Feeding the raw new dose in mg
        (0-15, ~15x the [0,1] state features) made new_dose training
        collapse-prone (verified 2026-07-05: entropy crash on 1/3 seeds); the
        network sees the same information either way, but normalisation keeps
        the gradients well-scaled.
        '''
        p = tf.gather(self._dose_values_t, dose_idx)
        if self._dose_conditioning == 'new_dose':
            prev_mg = (x[:, self._dose_feature_index]
                       * (self._dose_hi - self._dose_lo) + self._dose_lo)
            new_dose_mg = prev_mg * (one_float32 + p)
            signal = (new_dose_mg - self._dose_lo) / (self._dose_hi - self._dose_lo)
        else:  # pct_change: p is already ~[-1, 1]
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


@keras.utils.register_keras_serializable(
    package='reil.learners.ppo_learner_tandem')
class ExpectedDoseSignal(keras.layers.Layer):
    '''Differentiable expected new-dose signal for the duration head.

    signal = normalise( last_dose_mg * (1 + E[p]) ),
      E[p] = softmax(dose_logits) . dose_values   (expected % dose change)
      last_dose_mg = de-normalised state dose feature (min-max over dose_range)
    Output shape [batch, 1], on the same [0, 1] scale as the state features.
    Because it flows through softmax(dose_logits), gradients propagate from the
    duration loss back into the dose head — restoring the stabilising
    dose->duration coupling that the sampled-dose conditioning severs.
    '''

    def __init__(self, dose_values, dose_feature_index=0,
                 dose_range=(0.0, 15.0), **kwargs):
        super().__init__(**kwargs)
        self.dose_values = [float(v) for v in dose_values]
        self.dose_feature_index = int(dose_feature_index)
        self.dose_range = [float(dose_range[0]), float(dose_range[1])]

    def call(self, inputs):
        state, dose_logits = inputs
        dv = tf.constant(self.dose_values, dtype=tf.float32)
        lo, hi = self.dose_range
        probs = tf.nn.softmax(dose_logits, axis=-1)
        p_exp = tf.reduce_sum(probs * dv, axis=-1, keepdims=True)  # [b, 1]
        last_norm = state[:, self.dose_feature_index:self.dose_feature_index + 1]
        last_mg = last_norm * (hi - lo) + lo
        new_mg = last_mg * (1.0 + p_exp)
        return (new_mg - lo) / (hi - lo)

    def get_config(self):
        c = super().get_config()
        c.update(dict(dose_values=self.dose_values,
                      dose_feature_index=self.dose_feature_index,
                      dose_range=self.dose_range))
        return c


@keras.utils.register_keras_serializable(
    package='reil.learners.ppo_learner_tandem')
class PPOTandemExpectedDoseModel(PPOTandemModel):
    '''Tandem where the duration head conditions on the DIFFERENTIABLE expected
    new dose (Paper-3 Axis-A, 2026-07-06). Unlike PPOTandemConditionalModel
    (which conditions on the sampled dose and severs the gradient), the
    coupling here is `ExpectedDoseSignal(softmax(dose_logits))`, so the duration
    loss trains the dose trunk too — the stabiliser the logits tandem had, but
    with a 1-D dose-VALUE signal instead of the 21-D logits.

    Single-input actor (state -> [dose_logits, dur_logits]) with the coupling
    internal to the forward, so the standard tandem act()/train_actor/__call__
    paths apply unchanged (no two-stage rollout).
    '''

    def __init__(self, *args, dose_values: tuple[float, ...] = (),
                 dose_feature_index: int = 0,
                 dose_range: tuple[float, float] = (0.0, 15.0),
                 **kwargs: Any) -> None:
        self._dose_values_list = tuple(float(v) for v in dose_values)
        self._dose_feature_index = int(dose_feature_index)
        self._dose_range = (float(dose_range[0]), float(dose_range[1]))
        super().__init__(*args, **kwargs)

    def _build_networks(self):
        input_: Tensor = keras.Input(self._input_shape, name='state')  # type: ignore

        # Optional shared trunk (z feeds both sections). ExpectedDoseSignal
        # keeps reading the RAW state — the dose feature index addresses the
        # state layout, not the latent.
        z: Tensor = input_
        trunk_sizes = self._actor_layer_sizes.get('trunk')
        if trunk_sizes:
            z = TF2UtilsMixin.mlp_functional(
                input_, tuple(trunk_sizes),
                self._actor_hidden_activation, 'actor__trunk_{i:0>2}')

        dose_stack = TF2UtilsMixin.mlp_functional(
            z, self._actor_layer_sizes['dose'],
            self._actor_hidden_activation, 'actor__dose_{i:0>2}')
        dose_logits = keras.layers.Dense(
            self._action_per_head_units[0],
            activation=self._actor_head_activation,
            name='actor_dose_output')(dose_stack)

        signal = ExpectedDoseSignal(
            self._dose_values_list, self._dose_feature_index, self._dose_range,
            name='expected_dose_signal')([input_, dose_logits])
        dur_input = keras.layers.Concatenate(axis=-1)([z, signal])
        dur_stack = TF2UtilsMixin.mlp_functional(
            dur_input, self._actor_layer_sizes['duration'],
            self._actor_hidden_activation, 'actor__duration_{i:0>2}')
        dur_logits = keras.layers.Dense(
            self._action_per_head_units[1],
            activation=self._actor_head_activation,
            name='actor_duration_output')(dur_stack)

        self._head_layer_names = ['actor_dose_output', 'actor_duration_output']
        self.actor = keras.Model(inputs=input_, outputs=[dose_logits, dur_logits])

        self.critic = self._make_critic(input_)
