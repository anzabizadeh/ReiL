# -*- coding: utf-8 -*-
'''
DQNAgent class
=================

A Q-learning agent with Neural Network Q-function approximator

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''


import os
from dill import HIGHEST_PROTOCOL, dump, load
from random import choice, random
from time import time

from collections import deque
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
        super().__init__(**kwargs)
        super().set_defaults(gamma=1, epsilon=0, default_actions={},
                           learning_rate=1e-3, hidden_layer_sizes=(1,), input_length=1, method='forward',
                           training_x=deque(), training_y=deque(), buffer_index=-1, buffer_ready=False,
                           # np.array([], ndmin=2), training_y=np.array([], ndmin=2),
                           buffer_size=50, batch_size=10, validation_split=0.3, clear_buffer=False, tensorboard_path=None)
        super().set_params(**kwargs)
        self.data_collector.available_statistics = {}
        self.data_collector.active_statistics = []

        if 'filename' in kwargs:
            if 'path' in kwargs:
                self.load(filename=kwargs['filename'], path=kwargs['path'])
            else:
                self.load(filename=kwargs['filename'])
            return

        self._training_x = deque([0.0]*self._buffer_size)
        self._training_y = deque([0.0]*self._buffer_size)
        self._generate_network()

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._gamma, self._epsilon = 1, lambda x: 0
            self._default_actions = {}
            self._learning_rate, self._hidden_layer_sizes, self._input_length = 1e-5, (1,), 1
            self._batch_size, self._buffer_size, self._validation_split, self._clear_buffer = 10, 0, 0.3, False
            self._training_x, self._training_y, self._buffer_index, self._buffer_ready = deque(), deque(), -1, False
            self._tensorboard_path = None

        self._normalized_action_list = [a.normalize().as_list() for a in self._default_actions]

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
                    v, activation='relu', name=f'layer_{i+2:0>2}'))

            self._model.add(keras.layers.Dense(
                1, name='output'))

            self._model.compile(optimizer='adam', loss='mae')

            if self._tensorboard_path is None:
                self._tensorboard_path = os.path.join('logs', '_'.join(('gma', str(self._gamma), 'eps', 'func' if callable(self._epsilon) else str(self._epsilon),
                                                            'lrn', str(self._learning_rate), 'hddn', str(
                                                                self._hidden_layer_sizes),
                                                            'btch', str(self._batch_size), 'vld', str(self._validation_split))))
            else:
                self._tensorboard_path = os.path.join('logs', self._tensorboard_path)
            self._tensorboard = keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_path)  # , histogram_freq=1)  #, write_images=True)

    def _q(self, state, action=None):
        '''
        Return the Q-value of a state action pair.

        Arguments
        ---------
            state: the state for which Q-value is returned.
            action: the action for which Q-value is returned. 'None' uses default_actions.
        '''
        if isinstance(state, RLData):
            state = [state]
        state_list = [s.normalize().as_list() for s in state]
        len_state = len(state)

        if action is None:
            action_list = self._normalized_action_list
        else:
            if isinstance(action, RLData):
                action = [action]
            action_list = [a.normalize().as_list() for a in action]
        len_action = len(action_list)

        if len_state == len_action:
            X = np.array([state_list[i] + action_list[i] for i in range(len_state)])
        elif len_action == 1:
            X = np.array([state_list[i] + action_list[0] for i in range(len_state)])
        elif len_state == 1:
            X = np.array([state_list[0] + action_list[i] for i in range(len_action)])
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
            q_values = self._q(state)
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

            if self._method == 'forward':
                for i in range(len(history)):
                    state = history[i]['state']
                    action = history[i]['action']
                    reward = history[i]['reward']
                    try:
                        max_q = self._max_q(history[i+1]['state'])
                        new_q = reward + self._gamma*max_q
                    except IndexError:
                        new_q = reward

                    try:
                        self._buffer_index += 1
                        self._training_x[self._buffer_index] = state.normalize().as_list() + action.normalize().as_list()
                        self._training_y[self._buffer_index] = [new_q]
                    except IndexError:
                        self._buffer_ready = True
                        self._training_x[0] = state.normalize().as_list() + action.normalize().as_list()
                        self._training_y[0] = [new_q]
                        self._buffer_index = 1
            
            else:  # backward
                q_list = [0] * len(history)
                for i in range(len(history)-1, -1, -1):
                    state = history[i]['state']
                    action = history[i]['action']
                    reward = history[i]['reward']
                    try:
                        new_q = reward + self._gamma*q_list[i+1]
                    except IndexError:
                        new_q = reward
                    q_list[i] = new_q

                    try:
                        self._buffer_index += 1
                        self._training_x[self._buffer_index] = state.normalize().as_list() + action.normalize().as_list()
                        self._training_y[self._buffer_index] = [new_q]
                    except IndexError:
                        self._buffer_ready = True
                        self._training_x[0] = state.normalize().as_list() + action.normalize().as_list()
                        self._training_y[0] = [new_q]
                        self._buffer_index = 1

            if self._buffer_ready:
                index = np.random.choice(self._buffer_size, self._batch_size, replace=False)
                with self._session.as_default():
                    with self._graph.as_default():
                        self._model.fit(np.array(self._training_x)[index], np.array(self._training_y)[index],
                                        epochs=1, callbacks=[self._tensorboard], validation_split=self._validation_split)

                if self._clear_buffer:
                    self._buffer_index = 0
                    self._buffer_ready = False

            return

        except KeyError:
            raise RuntimeError('DQNAgent only works using \'history\'')

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
            q_values = self._q(state, None if possible_actions == self._default_actions else possible_actions)  # None is used to avoid redundant normalization of default_actions
            max_q = np.max(q_values)
            result = tuple(possible_actions[i] for i in np.nonzero(q_values==max_q)[0])

        action = choice(result)

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

        # To resolve a compatibility issue
        if not hasattr(self, '_normalized_action_list'):
            self._normalized_action_list = [a.normalize().as_list() for a in self._default_actions]

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._session = keras.backend.get_session()
            self._model = keras.models.load_model(kwargs.get(
                'path', self._path) + '/' + kwargs['filename'] + '.tf/' + kwargs['filename'])
            self._tensorboard = keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_path)  #, histogram_freq=1)  # , write_images=True)

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
            with self._session.as_default():
                with self._graph.as_default():
                    self._model.save(kwargs.get('path', self._path) + '/' +
                                    kwargs['filename'] + '.tf/' + kwargs['filename'])
        except OSError:
            os.makedirs(kwargs.get('path', self._path) +
                        '/' + kwargs['filename'] + '.tf/')
            with self._session.as_default():
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

    def __repr__(self):
        return 'DQNAgent'
