# -*- coding: utf-8 -*-
'''
ActorCriticLearner class
======================

The DenseSoftMax learner, comprised of a fully-connected with a softmax in the
output layer.
'''
import datetime
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from reil.datatypes.feature import FeatureSet
from reil.learners.learner import Learner
from reil.utils.tf_utils import TF2IOMixin
from tensorflow import keras

ACLabelType = Tuple[Tuple[Tuple[int, ...], ...], float]

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

eps = np.finfo(np.float32).eps.item()


class DeepActorCriticModel(keras.Model):
    def __init__(
            self,
            output_lengths: List[int],
            learning_rate: Union[
                float, keras.optimizers.schedules.LearningRateSchedule],
            shared_layer_sizes: Tuple[int, ...],
            actor_layer_sizes: Tuple[int, ...] = (),
            critic_layer_sizes: Tuple[int, ...] = (),
            critic_loss_coef: float = 1.0,
            entropy_loss_coef: float = 0.0,
        ):
        super().__init__()

        self._output_lengths = output_lengths
        self._shared_layer_sizes = shared_layer_sizes
        self._actor_layer_sizes = actor_layer_sizes
        self._critic_layer_sizes = critic_layer_sizes
        self._critic_loss_coef = critic_loss_coef
        self._entropy_loss_coef = entropy_loss_coef
        self._learning_rate = learning_rate

        self._metrics = {
            'actor_loss': tf.keras.metrics.Mean(
                'actor_loss', dtype=tf.float32),
            'critic_loss': tf.keras.metrics.Mean(
                'critic_loss', dtype=tf.float32),
            'entropy_loss': tf.keras.metrics.Mean(
                'entropy_loss', dtype=tf.float32),
            'total_loss': tf.keras.metrics.Mean(
                'total_loss', dtype=tf.float32),
            'actor_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(
                    'actor_accuracy', dtype=tf.float32),
            'return': tf.keras.metrics.Mean(
                'return', dtype=tf.float32)
            }

    @property
    def metrics(self):
        return list(self._metrics.values())

    def build(self, input_shape: Tuple[int, ...]):
        self._shared = [
            keras.layers.Dense(v, activation='relu', name=f'shared_{i+2:0>2}')
            for i, v in enumerate(self._shared_layer_sizes, 1)]

        # self._shared[0] = self._shared[0](keras.Input(shape=input_shape))

        self._actor_layers = [
            keras.layers.Dense(v, activation='relu', name=f'actor_{i:0>2}')
            for i, v in enumerate(self._actor_layer_sizes, 1)]

        self._critic_layers = [
            keras.layers.Dense(v, activation='relu', name=f'critic_{i:0>2}')
            for i, v in enumerate(self._critic_layer_sizes, 1)]

        self._actor_outputs = [
            keras.layers.Dense(
                output_length, activation='softmax',
                name=f'actor_output_{i:02}')
            for i, output_length in enumerate(self._output_lengths, 1)]

        self._critic_output = keras.layers.Dense(1, name='critic_output')

        self.optimizer = keras.optimizers.Adam(
            learning_rate=self._learning_rate)

    def call(self, inputs, training=None):
        x = inputs
        for layer in self._shared:
            x = layer(x, training=training)

        x_actor = x_critic = x

        for layer in self._actor_layers:
            x_actor = layer(x_actor, training=training)

        for layer in self._critic_layers:
            x_critic = layer(x_critic, training=training)

        logits = [
            layer(x_actor) for layer in self._actor_outputs]
        values = self._critic_output(x_critic)

        return logits, values

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='X'),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='Y'),
            tf.TensorSpec(shape=[None], dtype=tf.float32, name='returns')))
    def _gradients(
            self, x: tf.Tensor, y: tf.Tensor, returns: tf.Tensor):
        print('tracing')
        lengths = tf.constant(self._output_lengths)
        starts = tf.pad(lengths[:-1], [[1, 0]])
        ends = tf.math.cumsum(lengths)

        m = len(lengths)

        returns = tf.divide(
            returns - tf.math.reduce_mean(returns),
            tf.math.reduce_std(returns) + eps,
            name='normalized_returns')

        with tf.GradientTape() as tape:
            logits, values = self.call(
                tf.squeeze(x, name='x'), training=True)
            logits_concat = tf.concat(logits, axis=1, name='all_logits')
            values: tf.Tensor = tf.squeeze(values, name='values')

            if self._entropy_loss_coef:
                entropy_loss = self._entropy_loss_coef * tf.reduce_sum(logits_concat * tf.math.exp(logits_concat))
            else:
                entropy_loss = 0.0

            if self._critic_loss_coef:
                critic_loss = self._critic_loss_coef * huber_loss(
                    y_true=values, y_pred=returns)
            else:
                critic_loss = 0.0

            advantage = tf.stop_gradient(
                tf.subtract(returns, values, name='advantage'))

            actor_loss = tf.constant(0.0)
            for j in tf.range(m):
                logits_slice = logits_concat[:, starts[j]:ends[j]]
                y_slice = y[:, j]  # type: ignore
                action_probs = tfp.distributions.Categorical(
                    logits=logits_slice)
                log_prob = action_probs.log_prob(y_slice)
                _loss = -tf.reduce_sum(advantage * tf.squeeze(log_prob))
                actor_loss += _loss

                self._metrics['actor_accuracy'].update_state(
                    y_slice, logits_slice)

            total_loss = actor_loss + critic_loss + entropy_loss

            self._metrics['actor_loss'].update_state(actor_loss)
            self._metrics['critic_loss'].update_state(critic_loss)
            self._metrics['entropy_loss'].update_state(entropy_loss)
            self._metrics['total_loss'].update_state(total_loss)
 
        return tape.gradient(total_loss, self.trainable_variables)

    def train_step(self, data):
        x, y, returns = data

        self._metrics['return'].update_state(returns[0])

        gradient = self._gradients(x, y, returns)

        # If wanted to graph _gradients function in Tensorboard.
        # ------------------------------------------------------
        # if self._iteration:
        #     gradient = self._gradients(_X, _Y, G)
        # else:
        #     tf.summary.trace_on(graph=True, profiler=True)
        #     gradient = self._gradients(_X, _Y, G)
        #     with self._train_summary_writer.as_default():
        #         tf.summary.trace_export(
        #             name='gradient',
        #             step=0,
        #             profiler_outdir=str(self._tensorboard_path)
        #         )

        self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))

        metrics = {
            name: metric.result() for name, metric in self._metrics.items()}

        return metrics


class ActorCriticLearner(TF2IOMixin, Learner[FeatureSet, ACLabelType]):
    '''
    The DenseSoftMax learner, comprised of a fully-connected with a softmax
    in the output layer.

    This class uses `tf.keras` to build a sequential dense network with one
    output.
    '''

    def __init__(
            self,
            model: DeepActorCriticModel,
            tensorboard_path: Optional[Union[str, pathlib.PurePath]] = None,
            tensorboard_filename: Optional[str] = None,
            **kwargs: Any) -> None:
        '''
        Arguments
        ---------
        tensorboard_path:
            A path to save tensorboard outputs. If not provided,
            tensorboard will be disabled.

        Raises
        ------
        ValueError
            Validation split not in the range of (0.0, 1.0).
        '''

        super().__init__(models=['_model'], **kwargs)

        self._model = model

        self._iteration = 0

        self._tensorboard_path: Optional[pathlib.PurePath] = None
        self._tensorboard_filename = tensorboard_filename
        if (tensorboard_path or tensorboard_filename) is not None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self._tensorboard_path = pathlib.PurePath(
                tensorboard_path or './logs')
            filename = current_time + (f'-{tensorboard_filename}' or '')
            self._train_summary_writer = \
                tf.summary.create_file_writer(  # type: ignore
                str(self._tensorboard_path / filename))

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
        return self._model(self.convert_to_tensor(X), training=training)

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
        _X = self.convert_to_tensor(X)
        if len(_X.shape) == 1:
            _X = tf.expand_dims(_X, axis=0)

        Y_temp, G_temp = tuple(zip(*Y))
        G = tf.convert_to_tensor(G_temp, dtype=tf.float32)
        _Y: tf.Tensor = tf.convert_to_tensor(Y_temp)

        metrics = self._model.train_step((_X, _Y, G))

        with self._train_summary_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=self._iteration)

        self._iteration += 1

        return metrics
