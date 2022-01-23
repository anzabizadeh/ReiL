# -*- coding: utf-8 -*-
'''
DenseActorCritic class
======================

The DenseSoftMax learner, comprised of a fully-connected with a softmax in the
output layer.
'''
import pathlib
from typing import Any, List, Optional, Tuple, Union

import tensorflow as tf
import tensorflow_probability as tfp
from reil.datatypes.feature import FeatureSet
from reil.learners.learner import Learner
from reil.learners.learning_rate_schedulers import (ConstantLearningRate,
                                                    LearningRateScheduler)
from reil.utils.tf_utils import TF2IOMixin
from tensorflow import keras

ACLabelType = Tuple[Tuple[Tuple[int, ...], ...], float]


class DenseActorCritic(TF2IOMixin, Learner[FeatureSet, ACLabelType]):
    '''
    The DenseSoftMax learner, comprised of a fully-connected with a softmax
    in the output layer.

    This class uses `tf.keras` to build a sequential dense network with one
    output.
    '''

    def __init__(
            self,
            learning_rate: Union[float, LearningRateScheduler],
            output_lengths: List[int],
            validation_split: float = 0.3,
            hidden_layer_sizes: Tuple[int, ...] = (1,),
            input_length: Optional[int] = None,
            tensorboard_path: Optional[Union[str, pathlib.PurePath]] = None,
            **kwargs: Any) -> None:
        '''
        Arguments
        ---------
        output_lengths:
            A list of the sizes of all categorical outputs.

        learning_rate:
            A `LearningRateScheduler` object that determines the learning rate
            based on iteration. If any scheduler other than constant is
            provided, the model uses the `new_rate` method of the scheduler
            to determine the learning rate at each iteration.

        validation_split:
            How much of the training set should be used for validation?

        hidden_layer_sizes:
            A list of number of neurons for each layer.

        input_length:
            Size of the input data. If not supplied, the network will be
            generated based on the size of the first data point in `predict` or
            `learn` methods.

        tensorboard_path:
            A path to save tensorboard outputs. If not provided,
            tensorboard will be disabled.

        Raises
        ------
        ValueError
            Validation split not in the range of (0.0, 1.0).
        '''

        super().__init__(
            models=['_model'], learning_rate=learning_rate, **kwargs)

        if not 0.0 < validation_split < 1.0:
            raise ValueError('validation split should be in (0.0, 1.0).')

        self._validation_split = validation_split
        self._hidden_layer_sizes = hidden_layer_sizes
        self._input_length = input_length
        self._output_lengths = output_lengths

        self._tensorboard_path: Optional[pathlib.PurePath] = None
        self._callbacks: List[Any] = []
        if tensorboard_path is not None:
            self._tensorboard_path = pathlib.PurePath(
                'logs', tensorboard_path)
            self._tensorboard = keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_path, write_graph=True)
            # , histogram_freq=1)  #, write_images=True)
            self._callbacks.append(self._tensorboard)

        if not isinstance(self._learning_rate, ConstantLearningRate):
            learning_rate_scheduler = \
                keras.callbacks.LearningRateScheduler(
                    self._learning_rate.new_rate, verbose=0)
            self._callbacks.append(learning_rate_scheduler)

        self._no_model: bool = True
        if self._input_length is not None:
            self._generate_network()

    def _generate_network(self) -> None:
        '''
        Generate a multilayer neural net using `keras.Dense` and a softmax
        layer in the output.
        '''

        # separate networks
        # input_ = keras.Input(shape=(self._input_length,))  # type: ignore

        # layer = input_
        # for i, v in enumerate(self._hidden_layer_sizes, 1):
        #     layer = keras.layers.Dense(  # type: ignore
        #         v, activation='relu', name=f'actor_{i:0>2}')(layer)

        # actor_output = keras.layers.Dense(
        #     self._output_length, activation='softmax', name='actor')(layer)

        # layer = input_
        # for i, v in enumerate(self._hidden_layer_sizes, 1):
        #     layer = keras.layers.Dense(  # type: ignore
        #         v, activation='relu', name=f'critic_{i:0>2}')(layer)

        # critic_output = keras.layers.Dense(
        #     1, activation='sigmoid', name='critic')(layer)

        # self._actor = keras.Model(
        #     inputs=input_, outputs=actor_output)
        # self._actor.compile(  # type: ignore
        #     optimizer=keras.optimizers.Adam(
        #         learning_rate=self._learning_rate.initial_lr))

        # self._critic = keras.Model(
        #     inputs=input_, outputs=critic_output)
        # self._critic.compile(  # type: ignore
        #     optimizer=keras.optimizers.Adam(
        #         learning_rate=self._learning_rate.initial_lr))

        # single network
        input_ = keras.Input(shape=(self._input_length,))  # type: ignore

        layer = input_
        for i, v in enumerate(self._hidden_layer_sizes, 1):
            layer = keras.layers.Dense(  # type: ignore
                v, activation='relu', name=f'layer_{i:0>2}')(layer)

        actor_outputs = [
            keras.layers.Dense(
                output_length, activation='softmax',
                name=f'actor_{i:02}')(layer)
            for i, output_length in enumerate(self._output_lengths)]
        critic_output = keras.layers.Dense(
            1, activation='sigmoid', name='critic')(layer)

        self._model = keras.Model(
            inputs=input_, outputs=[actor_outputs, critic_output])
        self._model.compile(  # type: ignore
            optimizer=keras.optimizers.Adam(
                learning_rate=self._learning_rate.initial_lr))

        self._no_model = False

    def predict(self, X: Tuple[FeatureSet, ...]) -> Tuple[ACLabelType, ...]:
        '''
        predict `y` for a given input list `X`.

        Arguments
        ---------
        X:
            A list of `FeatureSet` as inputs to the prediction model.

        Returns
        -------
        :
            The predicted `y`.
        '''
        _X: List[List[float]] = [x.normalized.flattened for x in X]
        if self._no_model:
            self._input_length = len(_X[0])
            self._generate_network()

        logits, val = self._model(tf.convert_to_tensor(_X))

        # logits = self._actor.predict(np.array(_X))  # type: ignore
        # val = self._critic.predict(np.array(_X))  # type: ignore

        return logits, val

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='X'),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='Y'),
            tf.TensorSpec(shape=[None], dtype=tf.float32, name='G')))
    def _gradients(
            self, X: tf.Tensor, Y: tf.Tensor, G: tf.Tensor):
        lengths = self._output_lengths
        m = len(lengths)

        with tf.GradientTape() as tape:
            logits, values = self._model(tf.squeeze(X, name='X'))
            values: tf.Tensor = tf.squeeze(values, name='values')

            critic_loss = tf.keras.losses.mean_squared_error(
                y_true=G, y_pred=values)

            advantage = tf.subtract(G, values, name='advantage')
            advantage -= tf.math.reduce_mean(advantage)
            std = tf.math.reduce_std(advantage)
            advantage = tf.cond(
                tf.equal(std, 0.0),
                lambda: advantage,
                lambda: advantage/std,
                name='normalized_advantage')

            logits_concat = tf.concat(logits, axis=1, name='all_logits')
            n = tf.shape(advantage)[0]
            actor_loss = tf.constant(0.0)
            for i in tf.range(n):
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[(actor_loss, tf.TensorShape([]))])
                adv = advantage[i]
                y = Y[i]  # type: ignore
                ps = tf.RaggedTensor.from_row_lengths(
                    values=logits_concat[i], row_lengths=lengths)
                for j in tf.range(m):
                    action_probs = tfp.distributions.Categorical(
                        logits=ps[j])
                    log_prob = action_probs.log_prob(y[j])
                    temp = adv * tf.squeeze(log_prob)
                    temp.set_shape([])
                    actor_loss -= temp
 
        return tape.gradient(
            {'actor_loss': actor_loss, 'critic_loss': critic_loss},
            self._model.trainable_variables)

    def learn(
            self, X: Tuple[FeatureSet, ...], Y: Tuple[ACLabelType, ...],
            **kwargs: Any) -> None:
        '''
        Learn using the training set `X` and `Y`.

        Arguments
        ---------
        X:
            A list of `FeatureSet` as inputs to the learning model.

        Y:
            A list of float labels for the learning model.
        '''
        _X = tf.convert_to_tensor([x.normalized.flattened for x in X])
        if len(_X.shape) == 1:
            _X = tf.expand_dims(_X, axis=0)

        Y_temp, G_temp = tuple(zip(*Y))
        G = tf.convert_to_tensor(G_temp, dtype=tf.float32)
        _Y: tf.Tensor = tf.convert_to_tensor(Y_temp)

        if self._no_model:
            self._input_length = _X.shape[1]
            self._generate_network()

        gradient = self._gradients(_X, _Y, G)

        self._model.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(
                gradient, self._model.trainable_variables)
            if grad is not None)
