# -*- coding: utf-8 -*-
'''
DQNAgent class
=================

A Q-learning agent with Neural Network Q-function approximator

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''


import os
from pickle import HIGHEST_PROTOCOL, dump, load
from random import choice, random
from time import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .agent import Agent
from ..rldata import RLData


class DQNAgent(Agent):
    '''
    A Q-learning agent with deep neural network Q-function approximator.

    Constructor Arguments
    ---------------------
        gamma: discount factor in TD equation. (Default = 1)
        epsilon: exploration probability. (Default = 0)
        default_actions: list of default actions.
        learning_rate: learning rate for ANN. (Default = 1e-3)
        hidden_layer_sizes: tuple containing hidden layer sizes.
        input_length: size of the input vector. (Default = 1)
        buffer_size: DQN stores buffer_size observations and samples from it for training. (Default = 50)
        batch_size: the number of samples to choose randomly from the buffer for training. (Default = 10)
        validation_split: proportion of sampled observations set for validation. (Default = 0.3)
        clear_buffer: whether to clear buffer after sampling (True: clear buffer, False: only discard old observations). (Default = False)

        Note: Although input_length has a default value, but it should be specified in object construction.
        Note: This class doesn't have any data_collectors.
    Methods
    -------
        act: return an action based on the given state.
        learn: learn using either history or action, reward, and state.
        reset: Not Implemented Yet!
    '''

    def __init__(self, **kwargs):
        '''
        Initialize a Q-Learning agent with deep neural network Q-function approximator.
        '''
        Agent.__init__(self, **kwargs)
        Agent.set_defaults(self, gamma=1, epsilon=0, default_actions={},
                           learning_rate=1e-3, hidden_layer_sizes=(1,), input_length=1,
                           training_x=np.array([], ndmin=2), training_y=np.array([], ndmin=2),
                           buffer_size=50, batch_size=10, validation_split=0.3, clear_buffer=False, tensorboard_path=None)
        Agent.set_params(self, **kwargs)
        self.data_collector.available_statistics = {}
        self.data_collector.active_statistics = []

        if 'filename' in kwargs:
            if 'path' in kwargs:
                self.load(filename=kwargs['filename'], path=kwargs['path'])
            else:
                self.load(filename=kwargs['filename'])
            return

        self._generate_network()

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._gamma, self._epsilon = 1, lambda x: 0
            self._default_actions = {}
            self._learning_rate, self._hidden_layer_sizes, self._input_length = 1e-5, (1,), 1
            self._batch_size, self._buffer_size, self._validation_split, self._clear_buffer = 10, 0, 0.3, False
            self._training_x, self._training_y = np.array([], ndmin=2), np.array([], ndmin=2)
            self._tensorboard_path = None

    def _generate_network(self):
        '''
        Generate a tensorflow ANN network.
        '''

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._session = tf.Session()

            self._model = keras.models.Sequential()
            self._model.add(keras.layers.Dense(
                self._hidden_layer_sizes[0], activation='relu', name='layer_01', input_shape=(self._input_length,)))
            for i, v in enumerate(self._hidden_layer_sizes[1:]):
                self._model.add(keras.layers.Dense(
                    v, activation='relu', name='layer_{:0>2}'.format(i+2)))

            self._model.add(keras.layers.Dense(
                1, name='output'))  # activation='sigmoid', 

            self._model.compile(optimizer='adam', loss='mae')  # , metrics=['accuracy'])

            if self._tensorboard_path is None:
                self._tensorboard_path = 'logs/' + '_'.join(('gma', str(self._gamma), 'eps', 'func' if callable(self._epsilon) else str(self._epsilon),
                                                            'lrn', str(self._learning_rate), 'hddn', str(
                                                                self._hidden_layer_sizes),
                                                            'btch', str(self._batch_size), 'vld', str(self._validation_split)))
            else:
                self._tensorboard_path = 'logs/' + self._tensorboard_path
            self._tensorboard = keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_path, histogram_freq=1)  #, write_images=True)

            # self._session = tf.get_default_session()

    def _q(self, state, action):
        '''
        Return the Q-value of a state action pair.

        Arguments
        ---------
            state: the state for which Q-value is returned.
            action: the action for which Q-value is returned.
        '''
        if isinstance(state, RLData):
            state = [state]
        if isinstance(action, RLData):
            action = [action]

        len_state = len(state)
        len_action = len(action)
        if len_state == len_action:
            X = np.stack([np.append(state[i].normalize().as_nparray(), action[i].normalize().as_nparray()) for i in range(len_state)], axis=0)
        elif len_action == 1:
            action_np = action[0].normalize().as_nparray()
            X = np.stack([np.append(state[i].normalize().as_nparray(), action_np) for i in range(len_state)], axis=0)
        elif len_state == 1:
            state_np = state[0].normalize().as_nparray()
            X = np.stack([np.append(state_np, action[i].normalize().as_nparray()) for i in range(len_action)], axis=0)
        else:
            raise ValueError('State and action should be of the same size or at least one should be of size one.')

        with self._session.as_default():
            with self._graph.as_default():
                result = self._model.predict(X)

        return result

    def _max_q(self, state):
        '''
        Return MAX(Q) of a state.

        Arguments
        ---------
            state: the state for which MAX(Q) is returned.
        '''
        try:
            q_values = self._q(state, self._default_actions)
            max_q = np.max(q_values)
        except ValueError:
            max_q = 0
        return max_q

    def learn(self, **kwargs):
        '''
        Learn based on history.

        Arguments:
            history: a list consisting of state, action, reward of an episode.

        Note: Learning actually occurs every batch_size iterations.

        Raises ValueError if the agent is not in 'training' mode.
        '''
        if not self._training_flag:
            raise ValueError('Not in training mode!')
        try:
            history = kwargs['history']

            for i in range(len(history.index)):
                state = history.at[i, 'state']
                action = history.at[i, 'action']
                reward = history.at[i, 'reward']
                try:
                    max_q = self._max_q(history.at[i+1, 'state'])
                    new_q = reward + self._gamma*max_q
                except KeyError:
                    new_q = reward

                state_action = np.append(state.normalize().as_nparray(),
                                         action.normalize().as_nparray())
                try:
                    self._training_x = np.vstack((self._training_x, state_action))
                    self._training_y = np.vstack((self._training_y, new_q))
                except ValueError:
                    self._training_x = state_action
                    self._training_y = np.array(new_q)

            buffered_size = len(self._training_x)
            if buffered_size >= self._buffer_size:
                index = np.random.choice(buffered_size, self._batch_size, replace=False)
                with self._session.as_default():
                    with self._graph.as_default():
                        self._model.fit(self._training_x[index], self._training_y[index],
                                        epochs=1, callbacks=[self._tensorboard], validation_split=self._validation_split)

                if self._clear_buffer:
                    self._training_x = np.array([], ndmin=2)
                    self._training_y = np.array([], ndmin=2)
                else:
                    self._training_x = np.delete(self._training_x,
                                                range(buffered_size-self._buffer_size), axis=0)
                    self._training_y = np.delete(self._training_y,
                                                range(buffered_size-self._buffer_size), axis=0)

            return
        except KeyError:
            raise RuntimeError('ANNAgent only works using \'history\'')

    def act(self, state, **kwargs):
        '''
        return the best action for a given state.

        Arguments
        ---------
            state: the state for which an action is chosen.
            actions: a set of possible actions. If not provided, default actions are used.

        Note: If in 'training', the action is chosen randomly with probability of epsilon. In in 'test', the action is greedy.
        '''
        self._previous_state = state
        possible_actions = kwargs.get('actions', self._default_actions)
        episode = kwargs.get('episode', 0)
        try:
            epsilon = self._epsilon(episode)
        except TypeError:
            epsilon = self._epsilon

        if (self._training_flag) & (random() < epsilon):
            result = possible_actions
        else:
            q_values = self._q(state, possible_actions)
            max_q = np.max(q_values)
            result = tuple(possible_actions[i] for i in np.nonzero(q_values==max_q)[0])
            # result = max(((possible_actions[i], q_values[i]) for i in range(len(possible_actions))), key=lambda x: x[1])
            # action = result[0]

        action = choice(result)

        self._previous_action = action
        return action

    def load(self, **kwargs):
        '''
        Load an object from a file.

        Arguments
        ---------
            filename: the name of the file to be loaded.

        Note: tensorflow part is saved in filename.tf folder

        Raises ValueError if the filename is not specified.
        '''
        Agent.load(self, **kwargs)

        # self._session = tf.Session()
        # keras.backend.set_session(self._session)
        # self._model = keras.models.load_model(kwargs.get(
        #     'path', self._path) + '/' + kwargs['filename'] + '.tf/' + kwargs['filename'])
        # self._tensorboard = keras.callbacks.TensorBoard(
        #     log_dir=self._tensorboard_path, histogram_freq=1)  # , write_images=True)
        # self._graph = tf.get_default_graph()
        # tf.reset_default_graph()

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._session = keras.backend.get_session()
            # self._session = tf.Session()
            self._model = keras.models.load_model(kwargs.get(
                'path', self._path) + '/' + kwargs['filename'] + '.tf/' + kwargs['filename'])
            self._tensorboard = keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_path, histogram_freq=1)  # , write_images=True)

    def save(self, **kwargs):
        '''
        Save the object to a file.

        Arguments
        ---------
            filename: the name of the file to be saved.

        Note: tensorflow part should be in filename.tf folder

        Raises ValueError if the filename is not specified.
        '''

        pickle_data = tuple(key for key in self.__dict__ if key not in [
                            '_graph', '_session', '_model', '_tensorboard', 'data_collector'])
        path, filename = Agent.save(self, **kwargs, data=pickle_data)
        try:
            # with self._session.as_default():
            #     with self._graph.as_default():
            #         self._model.save(kwargs.get('path', self._path) + '/' +
            #                         kwargs['filename'] + '.tf/' + kwargs['filename'])
            with self._graph.as_default():
                self._model.save(kwargs.get('path', self._path) + '/' +
                                kwargs['filename'] + '.tf/' + kwargs['filename'])
        except OSError:
            os.makedirs(kwargs.get('path', self._path) +
                        '/' + kwargs['filename'] + '.tf/')
            # with self._session.as_default():
            #     with self._graph.as_default():
            #         self._model.save(kwargs.get('path', self._path) + '/' +
            #                         kwargs['filename'] + '.tf/' + kwargs['filename'])
            with self._graph.as_default():
                self._model.save(kwargs.get('path', self._path) + '/' +
                                kwargs['filename'] + '.tf/' + kwargs['filename'])
        return path, filename

    def _report(self, **kwargs):
        '''
        generate and return the requested report.

        Arguments
        ---------
            statistic: the list of items to report.

        Note: this function is not implemented yet!
        '''
        raise NotImplementedError

    # def __del__(self):
    #     self._tf['log_writer'].close()
    #     self._tf['session'].close()

    def __repr__(self):
        return 'DQNAgent'
