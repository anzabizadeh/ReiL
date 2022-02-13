# -*- coding: utf-8 -*-
'''
Dense class
===========

The Dense learner.
'''
import datetime
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf
from reil.datatypes.feature import FeatureSet
from reil.learners.learner import Learner
from reil.learners.learning_rate_schedulers import (ConstantLearningRate,
                                                    LearningRateScheduler)
from reil.utils.tf_utils import (ArgMaxLayer, BroadcastAndConcatLayer,
                                 MaxLayer, TF2IOMixin)
from tensorflow import keras


class DeepQModel(keras.Model):
    def __init__(
            self,
            learning_rate: Union[float, keras.optimizers.schedules.LearningRateSchedule],
            validation_split: float = 0.3,
            hidden_layer_sizes: Tuple[int, ...] = (1,)):
        super().__init__()

        if not 0.0 < validation_split < 1.0:
            raise ValueError('validation split should be in (0.0, 1.0).')

        self._validation_split = validation_split
        self._hidden_layer_sizes = hidden_layer_sizes
        self._learning_rate = learning_rate

        self._metrics = {}

    @property
    def metrics(self):
        return list(self._metrics.values())

    def build(self, input_shape: Tuple[int, ...]):
        self._inputs = [
            keras.Input(shape=(shape,), name=name)
            for name, shape in zip(
                ('state', 'action'), input_shape)]  # type: ignore

        self._concat = BroadcastAndConcatLayer(name='concat')

        self._layers = [
            keras.layers.Dense(
                size, activation='relu', name=f'layer_{i:0>2}')
            for i, size in enumerate(self._hidden_layer_sizes, 1)]

        self._output = keras.layers.Dense(1, name='Q')

        max_layer = MaxLayer(name='max')
        argmax_layer = ArgMaxLayer(name='argmax')

        self._model = keras.Model(self._inputs, output, name='Q_network')
        self._max = keras.Model(self._inputs, max_layer, name='max_Q')
        self._argmax = keras.Model(self._inputs, argmax, name='argmax_Q')

        self._model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self._learning_rate), loss='mae')

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





class QLearner(TF2IOMixin, Learner[Tuple[FeatureSet, ...], float]):
    '''
    The Dense learner for Q learning.

    This class uses `tf.keras` to build a sequential dense network with one
    output.
    '''

    def __init__(
            self,
            model: DeepQModel,
            tensorboard_path: Optional[Union[str, pathlib.PurePath]] = None,
            tensorboard_filename: Optional[str] = None,
            **kwargs: Any) -> None:
        '''
        Arguments
        ---------
        learning_rate:
            A `LearningRateScheduler` object that determines the learning rate
            based on iteration. If any scheduler other than constant is
            provided, the model uses the `new_rate` method of the scheduler
            to determine the learning rate at each iteration.

        validation_split:
            How much of the training set should be used for validation?

        hidden_layer_sizes:
            A list of number of neurons for each layer.

        input_lengths:
            Size of the input data. If not supplied, the network will be
            generated based on the size of the first data point in `predict` or
            `learn` methods. The inputs correspond to states and actions in
            `Qlearning`.

        tensorboard_path:
            A path to save tensorboard outputs. If not provided,
            tensorboard will be disabled.

        Raises
        ------
        ValueError
            Validation split not in the range of (0.0, 1.0).
        '''

        super().__init__(models=['_model'], **kwargs)

        self._iteration: int = 0

        self._model = model

        self._callbacks: List[Any] = []
        self._tensorboard_path: Optional[pathlib.PurePath] = None
        self._tensorboard_filename = tensorboard_filename
        if (tensorboard_path or tensorboard_filename) is not None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self._tensorboard_path = pathlib.PurePath(
                tensorboard_path or './logs')
            filename = current_time + (f'-{tensorboard_filename}' or '')

            self._tensorboard = keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_path / filename)
            # , histogram_freq=1)  #, write_images=True)
            self._callbacks.append(self._tensorboard)

    def _generate_network(self) -> None:
        '''
        Generate a multilayer neural net using `keras.Dense`.
        '''
        inputs = [
            keras.Input(shape=(shape,), name=name)
            for name, shape in zip(
                ('state', 'action'), self._input_lengths)]  # type: ignore

        concat = BroadcastAndConcatLayer(name='concat')(inputs)

        layer = keras.layers.Dense(
            self._hidden_layer_sizes[0],
            activation='relu', name='layer_01')(concat)

        for i, size in enumerate(self._hidden_layer_sizes[1:]):
            layer = keras.layers.Dense(
                size, activation='relu', name=f'layer_{i+2:0>2}')(layer)

        output = keras.layers.Dense(1, name='Q')(layer)

        max_layer = MaxLayer(name='max')(output)
        argmax = ArgMaxLayer(name='argmax')(output)

        self._model = keras.Model(inputs, output, name='Q_network')
        self._max = keras.Model(inputs, max_layer, name='max_Q')
        self._argmax = keras.Model(inputs, argmax, name='argmax_Q')

        self._model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self._learning_rate), loss='mae')

        self._no_model = False

    def argmax(
            self, states: Tuple[FeatureSet, ...],
            actions: Tuple[FeatureSet, ...]) -> Tuple[FeatureSet, FeatureSet]:
        _X = [self.convert_to_tensor(states), self.convert_to_tensor(actions)]
        if self._no_model:
            self._input_lengths = tuple(x.shape[1] for x in _X)
            self._generate_network()

        index = self._argmax(_X).numpy()[0]
        try:
            state = states[index]
        except IndexError:
            state = states[0]

        try:
            action = actions[index]
        except IndexError:
            action = actions[0]

        return state, action

    def max(
            self, states: Tuple[FeatureSet, ...],
            actions: Tuple[FeatureSet, ...]) -> float:
        result = self._max([
            self.convert_to_tensor(states), self.convert_to_tensor(actions)])

        return result

    def predict(
            self, X: Tuple[Tuple[FeatureSet, ...], ...],
            training: Optional[bool] = None) -> Tuple[float, ...]:
        result = self._model(
            [self.convert_to_tensor(x) for x in X], training=training)

        return result

    def learn(
            self, X: Tuple[Tuple[FeatureSet, ...], ...],
            Y: Tuple[float, ...]) -> Dict[str, float]:
        '''
        Learn using the training set `X` and `Y`.

        Arguments
        ---------
        X:
            A list of `FeatureSet` as inputs to the learning model.

        Y:
            A list of float labels for the learning model.
        '''
        _X = [self.convert_to_tensor(x) for x in X]
        if self._no_model:
            self._input_lengths = tuple(x.shape[1] for x in _X)
            self._generate_network()

        return self._model.fit(  # type: ignore
            _X, tf.convert_to_tensor(Y),  # type: ignore
            initial_epoch=self._iteration, epochs=self._iteration+1,
            callbacks=self._callbacks,
            validation_split=self._validation_split,
            verbose=0)

    def reset(self) -> None:
        '''
        reset the learner.
        '''
        self._iteration += 1

    def __getstate__(self):
        state = super().__getstate__()
        state['_max'] = None
        state['_argmax'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)

        if not self._no_model:
            inputs = self._model.inputs
            output = self._model.output

            max_layer = MaxLayer(name='max')(output)
            argmax = ArgMaxLayer(name='argmax')(output)

            self._max = keras.Model(inputs, max_layer, name='max_Q')
            self._argmax = keras.Model(inputs, argmax, name='argmax_Q')


if tf.__version__[0] == '1':  # type: ignore
    raise RuntimeError('Dense requires TF version 2.')
