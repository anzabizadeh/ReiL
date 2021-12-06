# -*- coding: utf-8 -*-
'''
DenseSoftMax class
==================

The DenseSoftMax learner, comprised of a fully-connected with a softmax in the
output layer.
'''
import pathlib
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from reil.datatypes.feature import FeatureArray
from reil.learners.dense import Dense_tf_2
from reil.learners.learning_rate_schedulers import LearningRateScheduler
from tensorflow import keras


class DenseSoftMax(Dense_tf_2):
    '''
    The DenseSoftMax learner, comprised of a fully-connected with a softmax
    in the output layer.

    This class uses `tf.keras` to build a sequential dense network with one
    output.
    '''

    def __init__(
            self,
            learning_rate: LearningRateScheduler,
            output_length: int,
            validation_split: float = 0.3,
            hidden_layer_sizes: Tuple[int, ...] = (1,),
            input_length: Optional[int] = None,
            tensorboard_path: Optional[Union[str, pathlib.PurePath]] = None,
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
            learning_rate=learning_rate,
            validation_split=validation_split,
            hidden_layer_sizes=hidden_layer_sizes,
            input_length=None,
            tensorboard_path=tensorboard_path,
            **kwargs)

        self._input_length = input_length
        self._output_length = output_length

        if self._input_length is not None:
            self._generate_network()

    def _generate_network(self) -> None:
        '''
        Generate a multilayer neural net using `keras.Dense` and a softmax
        layer in the output.
        '''

        input_ = keras.Input(shape=(self._input_length,))  # type: ignore

        layer = input_
        for i, v in enumerate(self._hidden_layer_sizes[:-1], 1):
            layer = keras.layers.Dense(  # type: ignore
                v, activation='relu', name=f'layer_{i:0>2}')(layer)

        output = keras.layers.Dense(
            self._output_length, activation='softmax', name='output')(layer)

        self._model = keras.Model(inputs=input_, outputs=output)
        self._model.compile(  # type: ignore
            optimizer=keras.optimizers.Adam(
                learning_rate=self._learning_rate.initial_lr))

        self._ann_ready = True

    def predict(self, X: Tuple[FeatureArray, ...]) -> Tuple[float, ...]:
        '''
        predict `y` for a given input list `X`.

        Arguments
        ---------
        X:
            A list of `FeatureArray` as inputs to the prediction model.

        Returns
        -------
        :
            The predicted `y`.
        '''
        _X: List[List[float]] = [x.normalized.flattened for x in X]
        if not self._ann_ready:
            self._input_length = len(_X[0])
            self._generate_network()

        probs: np.array = self._model.predict(np.array(_X))  # type: ignore

        return probs  # type: ignore

    def learn(
            self, X: Tuple[FeatureArray, ...], Y: Tuple[int, ...],
            **kwargs: Any
            ) -> None:
        '''
        Learn using the training set `X` and `Y`.

        Arguments
        ---------
        X:
            A list of `FeatureArray` as inputs to the learning model.

        Y:
            A list of float labels for the learning model.
        '''
        _X: List[List[float]] = [x.normalized.flattened for x in X]
        G = kwargs['G']
        if not self._ann_ready:
            self._input_length = len(_X[0])
            self._generate_network()

        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, _X)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self._model(state)
                action_probs = tf.compat.v1.distributions.Categorical(
                    probs=probs)
                log_prob = action_probs.log_prob(Y[idx])

                loss += -g * tf.squeeze(log_prob)

        gradient = tape.gradient(
            loss, self._model.trainable_variables)
        # print(gradient)
        self._model.optimizer.apply_gradients(
            zip(gradient, self._model.trainable_variables))
