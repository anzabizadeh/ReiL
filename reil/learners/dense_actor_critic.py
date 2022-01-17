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
from reil.datatypes.feature import FeatureSet
from reil.learners.learner import Learner
from reil.learners.learning_rate_schedulers import (ConstantLearningRate,
                                                    LearningRateScheduler)
from reil.utils.tf_utils import TF2IOMixin
from tensorflow import keras

ACLabelType = Tuple[Tuple[Tuple[int, ...], ...], float]


class DenseActorCritic(Learner[ACLabelType], TF2IOMixin):
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

        super().__init__(learning_rate=learning_rate, **kwargs)

        if not 0.0 < validation_split < 1.0:
            raise ValueError('validation split should be in (0.0, 1.0).')

        self._validation_split = validation_split
        self._hidden_layer_sizes = hidden_layer_sizes
        self._input_length = input_length
        self._output_lengths = output_lengths

        self._tensorboard_path: Optional[pathlib.PurePath] = None
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

        self._ann_ready: bool = False
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

        self._ann_ready = True

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
        if not self._ann_ready:
            self._input_length = len(_X[0])
            self._generate_network()

        logits, val = self._model(tf.convert_to_tensor(_X))

        # logits = self._actor.predict(np.array(_X))  # type: ignore
        # val = self._critic.predict(np.array(_X))  # type: ignore

        return logits, val

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None], dtype=tf.float32)))
    def _learn(
            self, X: tf.Tensor, Y: tf.Tensor, G: tf.Tensor):
        print('tracing _learn')
        with tf.GradientTape(persistent=True) as tape:
            logits, values = self._model(tf.squeeze(X))
            values: tf.Tensor = tf.squeeze(values)

            critic_loss = tf.keras.losses.mean_squared_error(
                y_true=G, y_pred=values)

            advantage = tf.subtract(G, values)
            advantage -= tf.math.reduce_mean(advantage)
            std = tf.math.reduce_std(advantage)
            advantage = tf.cond(
                tf.equal(std, 0.0), lambda: advantage, lambda: advantage/std)

            advantage = tf.data.Dataset.from_tensor_slices(advantage)

            logits_concat = tf.concat(logits, axis=1)
            logits_ds = tf.data.Dataset.from_tensor_slices(logits_concat)
            Y_ds = tf.data.Dataset.from_tensor_slices(Y)

            actor_loss = tf.Variable(0.0)
            for temp in tf.data.Dataset.zip((advantage, Y_ds, logits_ds)):
                adv = temp[0]
                y = tf.data.Dataset.from_tensor_slices(temp[1])
                ps = tf.data.Dataset.from_tensor_slices(
                    tf.RaggedTensor.from_row_lengths(
                        values=temp[2], row_lengths=self._output_lengths))
                for _y, p in tf.data.Dataset.zip((y, ps)):
                    action_probs = tf.compat.v1.distributions.Categorical(
                        logits=p)
                    log_prob = action_probs.log_prob(_y)
                    actor_loss.assign_sub(adv * tf.squeeze(log_prob))

        gradient = tape.gradient(
            critic_loss, self._model.trainable_variables)
        self._model.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradient, self._model.trainable_variables)
            if grad is not None)

        gradient = tape.gradient(
            actor_loss, self._model.trainable_variables)
        self._model.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradient, self._model.trainable_variables)
            if grad is not None)

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

        if not self._ann_ready:
            self._input_length = _X.shape[1]
            self._generate_network()

        self._learn(_X, _Y, G)

        # with tf.GradientTape(persistent=True) as tape:
        #     logits, values = self._model(tf.squeeze(_X))
        #     values: tf.Tensor = tf.squeeze(values)

        #     critic_loss = tf.Variable(DenseActorCritic._critic_loss(values, G))

        #     advantage = DenseActorCritic._advantage(values, G)

        #     actor_loss = tf.Variable(0.0)
        #     for temp in tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(advantage), tf.data.Dataset.from_tensor_slices(_Y), *(tf.data.Dataset.from_tensor_slices(lo) for lo in logits))):
        #         adv = temp[0]
        #         y = temp[1]
        #         ps = temp[2:]
        #         # y = _Y[idx]  # type: ignore
        #         for j, p in enumerate(ps):
        #             actor_loss.assign_add(
        #                 DenseActorCritic._actor_loss(p, y[j], adv))

        # gradient = tape.gradient(
        #     critic_loss, self._model.trainable_variables)
        # self._model.optimizer.apply_gradients(
        #     (grad, var)
        #     for (grad, var) in zip(gradient, self._model.trainable_variables)
        #     if grad is not None)

        # gradient = tape.gradient(
        #     actor_loss, self._model.trainable_variables)
        # self._model.optimizer.apply_gradients(
        #     (grad, var)
        #     for (grad, var) in zip(gradient, self._model.trainable_variables)
        #     if grad is not None)






        # -----------------------------------------
        # with tf.GradientTape(persistent=True) as tape:
        #     logits, values = self._model(tf.squeeze(_X))  # type: ignore

        #     values = tf.squeeze(values)

        #     critic_loss = tf.keras.losses.mean_squared_error(
        #         y_true=G, y_pred=values)

        #     advantage = G - values
        #     advantage -= np.mean(advantage)  # type: ignore
        #     advantage /= np.std(advantage) or 1.0  # type: ignore

        #     actor_loss = 0
        #     for idx in range(logits[0].shape[0]):
        #         adv = advantage[idx]
        #         ps = [lo[idx] for lo in logits]
        #         y = _Y[idx]
        #         for j, p in enumerate(ps):
        #             action_probs = tf.compat.v1.distributions.Categorical(
        #                 logits=p)
        #             log_prob = action_probs.log_prob(y[j])

        #             actor_loss += -adv * tf.squeeze(log_prob)

        #     # for idx, temp in enumerate(zip(advantage, *logits)):
        #     #     adv = temp[0]
        #     #     ps = temp[1:]
        #     #     y = _Y[idx]
        #     #     for j, p in enumerate(ps):
        #     #         action_probs = tf.compat.v1.distributions.Categorical(
        #     #             logits=p)
        #     #         log_prob = action_probs.log_prob(y[j])

        #     #         actor_loss += -adv * tf.squeeze(log_prob)

        # print(
        #     len(G), '\t', float(actor_loss), float(critic_loss),
        #     float(actor_loss + critic_loss))

        # gradient = tape.gradient(
        #     critic_loss, self._model.trainable_variables)
        # self._model.optimizer.apply_gradients(
        #     (grad, var)
        #     for (grad, var) in zip(gradient, self._model.trainable_variables)
        #     if grad is not None)

        # gradient = tape.gradient(
        #     actor_loss, self._model.trainable_variables)
        # self._model.optimizer.apply_gradients(
        #     zip(gradient, self._model.trainable_variables))

        # # with tf.GradientTape() as tape:
        # #     values = tf.squeeze(self._critic(_X))

        # #     critic_loss = tf.keras.losses.mean_squared_error(
        # #         y_true=G, y_pred=values)

        # # gradient = tape.gradient(
        # #     critic_loss, self._critic.trainable_variables)
        # # self._critic.optimizer.apply_gradients(
        # #     zip(gradient, self._critic.trainable_variables))

        # # with tf.GradientTape() as tape:
        # #     logits = self._actor(_X)

        # #     advantage = G - values
        # #     advantage -= np.mean(advantage)  # type: ignore
        # #     advantage /= np.std(advantage) or 1.0  # type: ignore

        # #     actor_loss = 0
        # #     for idx, (adv, p) in enumerate(zip(advantage, logits)):
        # #         action_probs = tf.compat.v1.distributions.Categorical(
        # #             logits=p)
        # #         log_prob = action_probs.log_prob(_Y[idx])

        # #         actor_loss += -adv * tf.squeeze(log_prob)

        # # gradient = tape.gradient(
        # #     actor_loss, self._actor.trainable_variables)
        # # self._actor.optimizer.apply_gradients(
        # #     zip(gradient, self._actor.trainable_variables))
