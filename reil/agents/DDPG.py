# typing: ignore

import os
import pickle
import time
from collections import deque
from functools import reduce
from typing import Any, Protocol, Tuple

import numpy as np
from pyparsing import Optional
import tensorflow as tf
from reil.datatypes.buffers.circular_buffer import CircularBuffer
from reil.logger import Logger
from reil import set_reil_random_seed
from reil.utils.tf_utils import TF2UtilsMixin

logger = Logger()


class AdaptiveParamNoiseSpec(object):
    def __init__(
            self, initial_stddev: float = 0.1,
            desired_action_stddev: float = 0.1,
            adoption_coefficient: float = 1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance: float):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = (
            'AdaptiveParamNoiseSpec(initial_stddev={}, '
            'desired_action_stddev={}, adoption_coefficient={})'
        )
        return fmt.format(
            self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class ActionNoise(Protocol):
    def reset(self):
        pass


class NormalActionNoise(ActionNoise):
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(
            self, mu: np.array, sigma: float,
            theta: float = .15, dt: float = 1e-2,
            x0: np.array | None = None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __repr__(self):
        return f'OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={self.sigma})'


class RingBuffer:
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.randint(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return

        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return len(self.observations0)


class RunningMeanStd(tf.Module):
    def __init__(self, epsilon=1e-2, shape=(), default_clip_range=np.inf):

        self._sum = tf.Variable(
            initial_value=np.zeros(shape=shape, dtype=np.float64),
            dtype=tf.float64,
            name="runningsum", trainable=False)
        self._sumsq = tf.Variable(
            initial_value=np.full(
                shape=shape, fill_value=epsilon, dtype=np.float64),
            dtype=tf.float64,
            name="runningsumsq", trainable=False)
        self._count = tf.Variable(
            initial_value=epsilon,
            dtype=tf.float64,
            name="count", trainable=False)
        self.shape = shape
        self.epsilon = epsilon
        self.default_clip_range = default_clip_range

    def update(self, x):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n * 2 + 1, 'float64')
        self._sum.assign_add(totalvec[0:n].reshape(self.shape))
        self._sumsq.assign_add(totalvec[n:2 * n].reshape(self.shape))
        self._count.assign_add(totalvec[2 * n])

    @property
    def mean(self):
        return tf.cast(self._sum / self._count, tf.float32)

    @property
    def std(self):
        return tf.sqrt(tf.maximum(
            tf.cast(self._sumsq / self._count, tf.float32) - tf.square(self.mean),
            self.epsilon)
        )

    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return tf.clip_by_value((v - self.mean) / self.std, -clip_range, clip_range)

    def denormalize(self, v):
        return self.mean + v * self.std


class Model(tf.keras.Model):
    def __init__(self, name, **network_kwargs):
        super(Model, self).__init__(name=name)
        self.network_kwargs = network_kwargs

    @property
    def perturbable_vars(self):
        return [
            var for var in self.trainable_variables
            if 'layer_normalization' not in var.name
        ]


class Actor(Model):
    def __init__(
            self, input_shape: tuple[int, ...], output_lengths: tuple[int, ...],
            layer_sizes: tuple[int, ...],
            name='actor', **network_kwargs):
        super().__init__(name=name, **network_kwargs)
        self._input_shape = input_shape
        self._output_lengths = output_lengths
        self._layer_sizes = layer_sizes
        input_ = tf.keras.Input(shape=input_shape)
        self.network_builder = TF2UtilsMixin.mlp_functional(
            input=input_, layer_sizes=layer_sizes,
            layer_name_format='actor_{i:0>2}',
            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
            activation=network_kwargs['activation']
        )

        self.output_layer = tf.keras.layers.Dense(
            units=output_lengths,
            activation=tf.keras.activations.tanh,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-3e-3, maxval=3e-3))
        self.output = self.output_layer(self.network_builder.outputs[0])

    def __call__(self, inputs, training: bool | None = None) -> Any:
        return self.output_layer(self.network_builder(inputs))


class Critic(Model):
    def __init__(
            self, input_shape: tuple[int, ...], output_lengths: tuple[int, ...],
            layer_sizes: tuple[int, ...],
            name='actor', **network_kwargs):
        super().__init__(name=name, **network_kwargs)
        self.layer_norm = True
        input_ = tf.keras.Input(shape=input_shape)
        self.network_builder = TF2UtilsMixin.mlp_functional(
            input=input_, layer_sizes=layer_sizes,
            layer_name_format='critic_{i:0>2}',
            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
            activation=network_kwargs['activation']
        )

        self.output_layer = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-3e-3, maxval=3e-3),
            name='output')
        self.output = self.output_layer(self.network_builder.outputs[0])

    @tf.function
    def call(self, obs, actions):
        # this assumes observation and action can be concatenated
        x = tf.concat([obs, actions], axis=-1)
        x = self.network_builder(x)
        return self.output_layer(x)

    @property
    def output_vars(self):
        return self.output_layer.trainable_variables


def normalize(
        x: tf.Tensor, stats: RunningMeanStd | None) -> tf.Tensor:
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)


def denormalize(
        x: tf.Tensor, stats: RunningMeanStd | None) -> tf.Tensor:
    if stats is None:
        return x
    return x * stats.std + stats.mean


@tf.function
def reduce_std(
        x: tf.Tensor, axis: int | None = None,
        keepdims: bool = False) -> tf.Tensor:
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_var(
        x: tf.Tensor, axis: int | None = None,
        keepdims: bool = False) -> tf.Tensor:
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


@tf.function
def update_perturbed_actor(
        actor: Actor, perturbed_actor: Actor,
        param_noise_stddev: float) -> None:
    perturbable_vars = actor.perturbable_vars
    for var, perturbed_var in zip(actor.variables, perturbed_actor.variables):
        if var in perturbable_vars:
            perturbed_var.assign(
                var + tf.random.normal(
                    shape=tf.shape(var), mean=0., stddev=param_noise_stddev))
        else:
            perturbed_var.assign(var)


class DDPG(tf.Module):
    def __init__(  # noqa: C901
            self, actor: Actor, critic: Critic, buffer: Memory,
            observation_shape, action_shape,
            param_noise: AdaptiveParamNoiseSpec | None = None,
            action_noise: ActionNoise | None = None,
            gamma: float = 0.99, tau: float = 0.001,
            normalize_returns: bool = False, enable_popart: bool = False,
            normalize_observations: bool = True, batch_size: int = 128,
            observation_range: tuple[float, float] = (-5., 5.),
            action_range: tuple[float, float] = (-1., 1.),
            return_range: tuple[float, float] = (-np.inf, np.inf),
            critic_l2_reg: float = 0., actor_lr: float = 1e-4,
            critic_lr: float = 1e-3,
            clip_norm: tuple[float, float] | None = None,
            reward_scale: float = 1.):

        self.gamma = gamma
        self.tau = tau
        self._buffer = buffer
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.observation_shape = observation_shape
        self.critic = critic
        self.actor = actor
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg
        self.actor_lr = tf.constant(actor_lr)
        self.critic_lr = tf.constant(critic_lr)

        # Observation normalization.
        if self.normalize_observations:
            with tf.name_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None

        # Return normalization.
        if self.normalize_returns:
            with tf.name_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # Create target networks.
        self.target_critic = Critic(
            actor._output_lengths, observation_shape, name='target_critic',
            **critic.network_kwargs)
        self.target_actor = Actor(
            actor._output_lengths, observation_shape, name='target_actor',
            **actor.network_kwargs)

        # Set up parts.
        if self.param_noise is not None:
            self.setup_param_noise()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=critic_lr)

        logger.info('setting up actor optimizer')
        actor_shapes = [var.get_shape().as_list()
                        for var in self.actor.trainable_variables]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                              for shape in actor_shapes])
        logger.info(f'  actor shapes: {actor_shapes}')
        logger.info(f'  actor params: {actor_nb_params}')
        logger.info('setting up critic optimizer')
        critic_shapes = [
            var.get_shape().as_list()
            for var in self.critic.trainable_variables
        ]
        critic_nb_params = sum([
            reduce(lambda x, y: x * y, shape)
            for shape in critic_shapes
        ])
        logger.info(f'  critic shapes: {critic_shapes}')
        logger.info(f'  critic params: {critic_nb_params}')
        if self.critic_l2_reg > 0.:
            critic_reg_vars = []
            for layer in self.critic.network_builder.layers[1:]:
                critic_reg_vars.append(layer.kernel)
            for var in critic_reg_vars:
                logger.info(f'  regularizing: {var.name}')
            logger.info(
                f'  applying l2 regularization with {self.critic_l2_reg}')

        logger.info('setting up critic target updates ...')
        for var, target_var in zip(self.critic.variables, self.target_critic.variables):
            logger.info(f'  {target_var.name} <- {var.name}')
        logger.info('setting up actor target updates ...')
        for var, target_var in zip(self.actor.variables, self.target_actor.variables):
            logger.info(f'  {target_var.name} <- {var.name}')

        if self.param_noise:
            logger.info('setting up param noise')
            for var, perturbed_var in zip(self.actor.variables, self.perturbed_actor.variables):
                if var in actor.perturbable_vars:
                    logger.info(
                        f'  {perturbed_var.name} <- {var.name} + noise')
                else:
                    logger.info(
                        f'  {perturbed_var.name} <- {var.name}')
            for var, perturbed_var in zip(
                    self.actor.variables, self.perturbed_adaptive_actor.variables):
                if var in actor.perturbable_vars:
                    logger.info(
                        f'  {perturbed_var.name} <- {var.name} + noise')
                else:
                    logger.info(
                        f'  {perturbed_var.name} <- {var.name}')

        if self.normalize_returns and self.enable_popart:
            self.setup_popart()

    def setup_param_noise(self):
        assert self.param_noise is not None

        # Configure perturbed actor.
        self.perturbed_actor = Actor(
            self.actor.nb_actions, self.observation_shape,
            name='param_noise_actor', **self.actor.network_kwargs)

        # Configure separate copy for stddev adoption.
        self.perturbed_adaptive_actor = Actor(
            self.actor.nb_actions, self.observation_shape,
            name='adaptive_param_noise_actor', **self.actor.network_kwargs)

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        for vs in [self.critic.output_vars, self.target_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='obs'),
            tf.TensorSpec(shape=[], dtype=tf.bool, name='apply_noise'),
            tf.TensorSpec(shape=[], dtype=tf.bool, name='compute_Q')
        )
    )
    def step(self, obs, apply_noise=True, compute_Q=True):
        normalized_obs = tf.clip_by_value(normalize(
            obs, self.obs_rms), self.observation_range[0], self.observation_range[1])
        actor_tf = self.actor(normalized_obs)
        if self.param_noise is not None and apply_noise:
            action = self.perturbed_actor(normalized_obs)
        else:
            action = actor_tf

        if compute_Q:
            normalized_critic_with_actor_tf = self.critic(
                normalized_obs, actor_tf)
            q = denormalize(tf.clip_by_value(normalized_critic_with_actor_tf,
                            self.return_range[0], self.return_range[1]), self.ret_rms)
        else:
            q = None

        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            action += noise
        action = tf.clip_by_value(
            action, self.action_range[0], self.action_range[1])

        return action, q, None, None

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        reward *= self.reward_scale

        B = obs0.shape[0]
        for b in range(B):
            self._buffer.append(obs0[b], action[b],
                               reward[b], obs1[b], terminal1[b])
            if self.normalize_observations:
                self.obs_rms.update(np.array([obs0[b]]))

    def train(self):
        batch = self._buffer.sample(batch_size=self.batch_size)
        obs0, obs1 = tf.constant(batch['obs0']), tf.constant(batch['obs1'])
        actions, rewards, terminals1 = tf.constant(batch['actions']), tf.constant(
            batch['rewards']), tf.constant(batch['terminals1'], dtype=tf.float32)
        normalized_obs0, target_Q = self.compute_normalized_obs0_and_target_Q(
            obs0, obs1, rewards, terminals1)

        if self.normalize_returns and self.enable_popart:
            old_mean = self.ret_rms.mean
            old_std = self.ret_rms.std
            self.ret_rms.update(target_Q.flatten())
            # renormalize Q outputs
            new_mean = self.ret_rms.mean
            new_std = self.ret_rms.std
            for vs in [self.critic.output_vars, self.target_critic.output_vars]:
                kernel, bias = vs
                kernel.assign(kernel * old_std / new_std)
                bias.assign((bias * old_std + old_mean - new_mean) / new_std)

        actor_grads, actor_loss = self.get_actor_grads(normalized_obs0)
        critic_grads, critic_loss = self.get_critic_grads(
            normalized_obs0, actions, target_Q)

        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables))

        return critic_loss, actor_loss

    @tf.function
    def compute_normalized_obs0_and_target_Q(self, obs0, obs1, rewards, terminals1):
        normalized_obs0 = tf.clip_by_value(normalize(
            obs0, self.obs_rms), self.observation_range[0], self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(normalize(
            obs1, self.obs_rms), self.observation_range[0], self.observation_range[1])
        Q_obs1 = denormalize(self.target_critic(
            normalized_obs1, self.target_actor(normalized_obs1)), self.ret_rms)
        target_Q = rewards + (1. - terminals1) * self.gamma * Q_obs1
        return normalized_obs0, target_Q

    @tf.function
    def get_actor_grads(self, normalized_obs0):
        with tf.GradientTape() as tape:
            actor_tf = self.actor(normalized_obs0)
            normalized_critic_with_actor_tf = self.critic(
                normalized_obs0, actor_tf)
            critic_with_actor_tf = denormalize(
                tf.clip_by_value(
                    normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]),
                self.ret_rms)
            actor_loss = -tf.reduce_mean(critic_with_actor_tf)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        if self.clip_norm:
            actor_grads = [tf.clip_by_norm(
                grad, clip_norm=self.clip_norm) for grad in actor_grads]
        return actor_grads, actor_loss

    @tf.function
    def get_critic_grads(self, normalized_obs0, actions, target_Q):
        with tf.GradientTape() as tape:
            normalized_critic_tf = self.critic(normalized_obs0, actions)
            normalized_critic_target_tf = tf.clip_by_value(
                normalize(target_Q, self.ret_rms),
                self.return_range[0], self.return_range[1])
            critic_loss = tf.reduce_mean(
                tf.square(normalized_critic_tf - normalized_critic_target_tf))
            # The first is input layer, which is ignored here.
            if self.critic_l2_reg > 0.:
                # Ignore the first input layer.
                for layer in self.critic.network_builder.layers[1:]:
                    # The original l2_regularizer takes half of sum square.
                    critic_loss += (self.critic_l2_reg / 2.) * \
                        tf.reduce_sum(tf.square(layer.kernel))
        critic_grads = tape.gradient(
            critic_loss, self.critic.trainable_variables)
        if self.clip_norm:
            critic_grads = [tf.clip_by_norm(
                grad, clip_norm=self.clip_norm) for grad in critic_grads]
        return critic_grads, critic_loss

    def initialize(self):
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    @tf.function
    def update_target_net(self):
        for var, target_var in zip(self.actor.variables, self.target_actor.variables):
            target_var.assign((1. - self.tau) * target_var + self.tau * var)
        for var, target_var in zip(self.critic.variables, self.target_critic.variables):
            target_var.assign((1. - self.tau) * target_var + self.tau * var)

    def get_stats(self):

        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self._buffer.sample(batch_size=self.batch_size)
        obs0 = self.stats_sample['obs0']
        actions = self.stats_sample['actions']
        normalized_obs0 = tf.clip_by_value(normalize(
            obs0, self.obs_rms), self.observation_range[0], self.observation_range[1])
        normalized_critic_tf = self.critic(normalized_obs0, actions)
        critic_tf = denormalize(tf.clip_by_value(
            normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        actor_tf = self.actor(normalized_obs0)
        normalized_critic_with_actor_tf = self.critic(
            normalized_obs0, actor_tf)
        critic_with_actor_tf = denormalize(
            tf.clip_by_value(
                normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]),
            self.ret_rms)

        stats = {}
        if self.normalize_returns:
            stats['ret_rms_mean'] = self.ret_rms.mean
            stats['ret_rms_std'] = self.ret_rms.std
        if self.normalize_observations:
            stats['obs_rms_mean'] = tf.reduce_mean(self.obs_rms.mean)
            stats['obs_rms_std'] = tf.reduce_mean(self.obs_rms.std)
        stats['reference_Q_mean'] = tf.reduce_mean(critic_tf)
        stats['reference_Q_std'] = reduce_std(critic_tf)
        stats['reference_actor_Q_mean'] = tf.reduce_mean(critic_with_actor_tf)
        stats['reference_actor_Q_std'] = reduce_std(critic_with_actor_tf)
        stats['reference_action_mean'] = tf.reduce_mean(actor_tf)
        stats['reference_action_std'] = reduce_std(actor_tf)

        if self.param_noise:
            perturbed_actor_tf = self.perturbed_actor(normalized_obs0)
            stats['reference_perturbed_action_mean'] = tf.reduce_mean(
                perturbed_actor_tf)
            stats['reference_perturbed_action_std'] = reduce_std(
                perturbed_actor_tf)
            stats.update(self.param_noise.get_stats())
        return stats

    def adapt_param_noise(self, obs0):
        if self.param_noise is None:
            return 0.

        mean_distance = self.get_mean_distance(obs0).numpy()

        self.param_noise.adapt(mean_distance)
        return mean_distance

    @tf.function
    def get_mean_distance(self, obs0):
        # Perturb a separate copy of the policy to adjust the scale for
        # the next "real" perturbation.
        update_perturbed_actor(
            self.actor, self.perturbed_adaptive_actor, self.param_noise.current_stddev)

        normalized_obs0 = tf.clip_by_value(normalize(
            obs0, self.obs_rms), self.observation_range[0], self.observation_range[1])
        actor_tf = self.actor(normalized_obs0)
        adaptive_actor_tf = self.perturbed_adaptive_actor(normalized_obs0)
        mean_distance = tf.sqrt(tf.reduce_mean(
            tf.square(actor_tf - adaptive_actor_tf)))
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            update_perturbed_actor(
                self.actor, self.perturbed_actor, self.param_noise.current_stddev)


def learn(  # noqa: C901
        network, env,
        seed=None,
        total_timesteps=None,
        nb_epochs=None,  # with default settings, perform 1M steps total
        nb_epoch_cycles=20,
        nb_rollout_steps=100,
        reward_scale=1.0,
        render=False,
        render_eval=False,
        noise_type='adaptive-param_0.2',
        normalize_returns=False,
        normalize_observations=True,
        critic_l2_reg=1e-2,
        actor_lr=1e-4,
        critic_lr=1e-3,
        popart=False,
        gamma=0.99,
        clip_norm=None,
        nb_train_steps=50,  # per epoch cycle and MPI worker,
        nb_eval_steps=100,
        batch_size=64,  # per MPI worker
        tau=0.01,
        eval_env=None,
        param_noise_adaption_interval=50,
        load_path=None,
        **network_kwargs):

    set_reil_random_seed(seed)

    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(
            total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
    else:
        nb_epochs = 500

    nb_actions = env.action_space.shape[-1]
    # we assume symmetric actions.
    assert (np.abs(env.action_space.low) == env.action_space.high).all()

    buffer = Memory(limit=int(1e6), action_shape=env.action_space.shape,
                    observation_shape=env.observation_space.shape)
    critic = Critic(nb_actions, ob_shape=env.observation_space.shape,
                    network=network, **network_kwargs)
    actor = Actor(nb_actions, ob_shape=env.observation_space.shape,
                  network=network, **network_kwargs)

    action_noise = None
    param_noise = None
    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(
                    stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(
                    nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(
                    nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError(
                    'unknown noise type "{}"'.format(current_noise_type))

    max_action = env.action_space.high
    logger.info(
        f'scaling actions by {max_action} before executing in env')

    agent = DDPG(
        actor, critic, buffer,
        env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns,
        normalize_observations=normalize_observations, batch_size=batch_size,
        action_noise=action_noise, param_noise=param_noise,
        critic_l2_reg=critic_l2_reg, actor_lr=actor_lr, critic_lr=critic_lr,
        enable_popart=popart, clip_norm=clip_norm, reward_scale=reward_scale)

    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    if load_path is not None:
        load_path = os.expanduser(load_path)
        ckpt = tf.train.Checkpoint(model=agent)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        ckpt.restore(manager.latest_checkpoint)
        print("Restoring from {}".format(manager.latest_checkpoint))

    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)

    # Prepare everything.
    agent.initialize()

    agent.reset()

    obs = env.reset()
    if eval_env is not None:
        eval_obs = eval_env.reset()
    nenvs = obs.shape[0]

    episode_reward = np.zeros(nenvs, dtype=np.float32)  # vector
    episode_step = np.zeros(nenvs, dtype=int)  # vector
    episodes = 0  # scalar
    t = 0  # scalar

    epoch = 0

    start_time = time.time()

    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0
    for epoch in range(nb_epochs):
        for _ in range(nb_epoch_cycles):
            # Perform rollouts.
            if nenvs > 1:
                # if simulating multiple envs in parallel, impossible to reset
                # agent at the end of the episode in each
                # of the environments, so resetting here instead
                agent.reset()
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                action, q, _, _ = agent.step(tf.constant(
                    obs), apply_noise=True, compute_Q=True)
                action, q = action.numpy(), q.numpy()

                # Execute next action.
                if render:
                    env.render()

                # max_action is of dimension A, whereas action is dimension (nenvs, A)
                # - the multiplication gets broadcasted to the batch
                # scale for execution in env (as far as DDPG is concerned,
                # every action is in [-1, 1])
                new_obs, r, done, info = env.step(max_action * action)
                # note these outputs are batched from vecenv

                t += 1
                if render:
                    env.render()
                episode_reward += r
                episode_step += 1

                # Book-keeping.
                epoch_actions.append(action)
                epoch_qs.append(q)
                # the batched data will be unrolled in memory.py's append.
                agent.store_transition(obs, action, r, new_obs, done)

                obs = new_obs

                for d in range(len(done)):
                    if done[d]:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward[d])
                        episode_rewards_history.append(episode_reward[d])
                        epoch_episode_steps.append(episode_step[d])
                        episode_reward[d] = 0.
                        episode_step[d] = 0
                        epoch_episodes += 1
                        episodes += 1
                        if nenvs == 1:
                            agent.reset()

            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if buffer.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    batch = agent.buffer.sample(batch_size=batch_size)
                    obs0 = tf.constant(batch['obs0'])
                    distance = agent.adapt_param_noise(obs0)
                    epoch_adaptive_distances.append(distance)

                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

            # Evaluate.
            eval_episode_rewards = []
            eval_qs = []
            if eval_env is not None:
                nenvs_eval = eval_obs.shape[0]
                eval_episode_reward = np.zeros(nenvs_eval, dtype=np.float32)
                for t_rollout in range(nb_eval_steps):
                    eval_action, eval_q, _, _ = agent.step(
                        eval_obs, apply_noise=False, compute_Q=True)
                    # scale for execution in env
                    # (as far as DDPG is concerned, every action is in [-1, 1])
                    eval_obs, eval_r, eval_done, eval_info = eval_env.step(
                        max_action * eval_action)
                    if render_eval:
                        eval_env.render()
                    eval_episode_reward += eval_r

                    eval_qs.append(eval_q)
                    for d in range(len(eval_done)):
                        if eval_done[d]:
                            eval_episode_rewards.append(eval_episode_reward[d])
                            eval_episode_rewards_history.append(
                                eval_episode_reward[d])
                            eval_episode_reward[d] = 0.0

        mpi_size = 1

        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        stats = agent.get_stats()
        combined_stats = stats.copy()
        combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
        combined_stats['rollout/return_std'] = np.std(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(
            episode_rewards_history)
        combined_stats['rollout/return_history_std'] = np.std(
            episode_rewards_history)
        combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
        combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
        combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats['train/param_noise_distance'] = np.mean(
            epoch_adaptive_distances)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(t) / float(duration)
        combined_stats['total/episodes'] = episodes
        combined_stats['rollout/episodes'] = epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(epoch_actions)
        # Evaluation statistics.
        if eval_env is not None:
            combined_stats['eval/return'] = eval_episode_rewards
            combined_stats['eval/return_history'] = np.mean(
                eval_episode_rewards_history)
            combined_stats['eval/Q'] = eval_qs
            combined_stats['eval/episodes'] = len(eval_episode_rewards)

        combined_stats_sums = np.array(
            [np.array(x).flatten()[0] for x in combined_stats.values()])

        combined_stats = {
            k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps'] = t

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])

        logger.dump_tabular()
        logger.info('')
        logdir = logger.get_dir()
        if logdir:
            if hasattr(env, 'get_state'):
                with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                    pickle.dump(env.get_state(), f)
            if eval_env and hasattr(eval_env, 'get_state'):
                with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                    pickle.dump(eval_env.get_state(), f)

    return agent
