# -*- coding: utf-8 -*-
'''
Dense class
===========

The Dense learner.
'''
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


class QDense(TF2IOMixin, Learner[Tuple[FeatureSet, ...], float]):
    '''
    The Dense learner for Q learning.

    This class uses `tf.keras` to build a sequential dense network with one
    output.
    '''

    def __init__(
            self,
            learning_rate: Union[float, LearningRateScheduler],
            validation_split: float = 0.3,
            hidden_layer_sizes: Tuple[int, ...] = (1,),
            input_lengths: Optional[Tuple[int, int]] = None,
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

        super().__init__(
            models=['_model'],
            learning_rate=learning_rate, **kwargs)

        self._iteration: int = 0

        self._hidden_layer_sizes = hidden_layer_sizes
        self._input_lengths = input_lengths

        if not 0.0 < validation_split < 1.0:
            raise ValueError('validation split should be in (0.0, 1.0).')

        self._validation_split = validation_split

        self._callbacks: List[Any] = []
        self._tensorboard_path: Optional[pathlib.PurePath] = None
        self._model = keras.models.Sequential()

        if tensorboard_path is not None:
            self._tensorboard_path = pathlib.PurePath(
                'logs', tensorboard_path)
            self._tensorboard = keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_path)
            # , histogram_freq=1)  #, write_images=True)
            self._callbacks.append(self._tensorboard)

        if not isinstance(self._learning_rate, ConstantLearningRate):
            learning_rate_scheduler = \
                keras.callbacks.LearningRateScheduler(
                    self._learning_rate.new_rate, verbose=0)
            self._callbacks.append(learning_rate_scheduler)

        self._no_model: bool = True
        if self._input_lengths is not None:
            self._generate_network()

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
                learning_rate=self._learning_rate.initial_lr), loss='mae')

        self._no_model = False

    @staticmethod
    def _to_tensor(data: Tuple[FeatureSet, ...]) -> tf.Tensor:
        return tf.convert_to_tensor([x.normalized.flattened for x in data])

    def argmax(
            self, states: Tuple[FeatureSet, ...],
            actions: Tuple[FeatureSet, ...]) -> Tuple[FeatureSet, FeatureSet]:
        _X = [self._to_tensor(states), self._to_tensor(actions)]
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
        result = self._max([self._to_tensor(states), self._to_tensor(actions)])

        return result

    def predict(
            self, X: Tuple[Tuple[FeatureSet, ...], ...]) -> Tuple[float, ...]:
        result = self._model([self._to_tensor(x) for x in X])

        return result

    def learn(
            self, X: Tuple[Tuple[FeatureSet, ...], ...],
            Y: Tuple[float, ...], **kwargs: Any) -> None:
        '''
        Learn using the training set `X` and `Y`.

        Arguments
        ---------
        X:
            A list of `FeatureSet` as inputs to the learning model.

        Y:
            A list of float labels for the learning model.
        '''
        _X = [self._to_tensor(x) for x in X]
        if self._no_model:
            self._input_lengths = tuple(x.shape[1] for x in _X)
            self._generate_network()

        self._model.fit(  # type: ignore
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
