# -*- coding: utf-8 -*-
'''
DQNAgent class
=================

A Q-learning agent with Neural Network Q-function approximator

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import os
from collections import deque
from logging import WARNING
from pathlib import Path
from random import choice, random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from dill import HIGHEST_PROTOCOL, dump, load
from tensorflow import keras

from ..rldata import RLData
from .agent import Agent


class DQNAgent(Agent):
    '''
    A Q-learning agent with deep neural network Q-function approximator.

    Constructor Arguments
    ---------------------
        gamma: discount factor in TD equation. (Default = 1)
        epsilon: exploration probability. (Default = 0)
        default_actions: list of default actions.
        lr_initial: learning rate for ANN. (Default = 1e-3)
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

    def __init__(self,
                 filename: Optional[str] = None,
                 path: Optional[Union[str, Path]] = None,
                 gamma: float = 1.0, epsilon: Union[Callable, float] = 0.0,
                 default_actions: Sequence[RLData] = [],
                 lr_initial: float = 1e-3,
                 lr_scheduler: Callable[[int, float], float] = lambda epoch, lr: lr,
                 hidden_layer_sizes: Sequence[int] = (1,),
                 input_length: Optional[int] = None,
                 method: str = 'backward',
                 buffer_size: int = 50,
                 batch_size: int = 10,
                 validation_split: float = 0.3,
                 clear_buffer: bool = False,
                 tensorboard_path: Optional[Union[str, Path]] = None,
                 name: str = 'DQN_agent',
                 version: float = 0.5,
                 ex_protocol_current: Dict[str, str] = {'mode': 'training'},
                 ex_protocol_options: Dict[str, List[str]] = {'mode': ['training', 'test']},
                 stats_list: Sequence[str] = [],
                 logger_name: str = __name__,
                 logger_level: int = WARNING,
                 logger_filename: Optional[str] = None,
                 persistent_attributes: List[str] = []):
        '''
        Initialize a Q-Learning agent with deep neural network Q-function approximator.
        '''

        if filename is not None:
            if path is not None:
                self.load(filename=filename, path=path)
            else:
                self.load(filename=filename)
            return

        super().__init__(name=name,
                         version=version,
                         path=path,
                         ex_protocol_current=ex_protocol_current,
                         ex_protocol_options=ex_protocol_options,
                         stats_list=stats_list,
                         logger_name=logger_name,
                         logger_level=logger_level,
                         logger_filename=logger_filename,
                         persistent_attributes=persistent_attributes)

        self._gamma = min(gamma, 1.0)
        self._epsilon = epsilon
        self._default_actions = default_actions
        self._normalized_action_list = [a.normalize().as_list() for a in self._default_actions]

        self._lr_initial = lr_initial
        self._lr_scheduler = lr_scheduler

        self._hidden_layer_sizes = hidden_layer_sizes

        if input_length is None:
            raise ValueError('input_length is not specified.')
        self._input_length = input_length

        if method not in ('backward', 'forward'):
            self._logger.warning(f'method {method} is not acceptable. Should be either "forward" or "backward". Will use "backward".')
            self._method = 'backward'
        else:
            self._method = method.lower()

        self._buffer_size = max(buffer_size, 1)  # at least size 1
        self._batch_size = min(batch_size, buffer_size)  # at most as much as buffer_size

        if not 0.0 < validation_split < 1.0:
            raise ValueError('validation split should be in (0.0, 1.0).')
        self._validation_split = validation_split

        self._clear_buffer = clear_buffer
        self._tensorboard_path = tensorboard_path

        self.data_collector.available_statistics = {}
        self.data_collector.active_statistics = []

        self._training_x = deque([0.0]*self._buffer_size)
        self._training_y = deque([0.0]*self._buffer_size)
        self._generate_network()

        self._epoch = 0
        self._buffer_index = -1
        self._buffer_ready = False

    def _generate_network(self) -> None:
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

            self._model.compile(optimizer=keras.optimizers.Adam(learning_rate=self._lr_initial), loss='mae')

            self._callbacks = []
            if self._tensorboard_path is not None:
                self._tensorboard_path = Path('logs', self._tensorboard_path)
                self._tensorboard = keras.callbacks.TensorBoard(
                    log_dir=self._tensorboard_path)  # , histogram_freq=1)  #, write_images=True)
                self._callbacks.append(self._tensorboard)

            if self._lr_scheduler is not None:
                self._learning_rate_scheduler = keras.callbacks.LearningRateScheduler(self._lr_scheduler, verbose=1)
                self._callbacks.append(self._learning_rate_scheduler)

    def _q(self, state: Union[List[RLData], RLData], action: Optional[Union[List[RLData], RLData]] = None) -> float:
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

    def _max_q(self, state: Union[List[RLData], RLData]) -> float:
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

    def learn(self, history: List[Dict[str, Any]]) -> None:
        '''
        Learn based on history.

        Arguments:
            history: a list consisting of state, action, reward of an episode.

        Note: Learning actually occurs every batch_size iterations.

        Raises ValueError if the agent is not in 'training' mode.
        '''
        if not self.training_mode:
            raise ValueError('Not in training mode!')
        try:
            if self._method == 'forward':
                for i in range(len(history)):
                    state = history[i]['state']
                    action = history[i]['action']
                    reward = history[i]['reward'][0]
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
                    reward = history[i]['reward'][0]
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
                                        initial_epoch=self._epoch, epochs=self._epoch+1,
                                        callbacks=self._callbacks,
                                        validation_split=self._validation_split,
                                        verbose=2)

                if self._clear_buffer:
                    self._buffer_index = 0
                    self._buffer_ready = False

        except KeyError:
            raise RuntimeError('DQNAgent only works using \'history\'')

    def act(self, state: RLData, actions: Optional[List[RLData]] = None, episode: Optional[int] = 0) -> RLData:
        '''
        return the best action for a given state.

        Arguments
        ---------
            state: the state for which an action is chosen.
            actions: a set of possible actions. If not provided, default actions are used.

        Note: If in 'training', the action is chosen randomly with probability of epsilon. In in 'test', the action is greedy.
        '''
        self._previous_state = state
        possible_actions = actions
        try:
            epsilon = self._epsilon(episode)
        except TypeError:
            epsilon = self._epsilon

        if (self.training_mode) & (random() < epsilon):
            result = possible_actions
        else:
            q_values = self._q(state, None if possible_actions == self._default_actions else possible_actions)  # None is used to avoid redundant normalization of default_actions
            max_q = np.max(q_values)
            result = tuple(possible_actions[i] for i in np.nonzero(q_values==max_q)[0])

        action = choice(result)

        return action

    def load(self, filename: str, path: Optional[str] = None) -> None:
        '''
        Load an object from a file.

        Arguments
        ---------
            filename: the name of the file to be loaded.

        Note: tensorflow part is saved in filename.tf folder

        Raises ValueError if the filename is not specified.
        '''
        Agent.load(self, filename, path)

        # To resolve a compatibility issue
        if not hasattr(self, '_normalized_action_list'):
            self._normalized_action_list = [a.normalize().as_list() for a in self._default_actions]

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._session = keras.backend.get_session()
            self._model = keras.models.load_model(Path(path if path is not None else self._path,
                f'{filename}.tf', filename))
            self._tensorboard = keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_path)  #, histogram_freq=1)  # , write_images=True)
            self._learning_rate_scheduler = keras.callbacks.LearningRateScheduler(self._lr_scheduler)

    def save(self, filename: Optional[str] = None, path: Optional[str] = None, data_to_save: Optional[Sequence[str]] = None) -> Tuple[Union[str, Path], str]:
        '''
        Save the object to a file.

        Arguments
        ---------
            filename: the name of the file to be saved.

        Note: tensorflow part should be in filename.tf folder

        Raises ValueError if the filename is not specified.
        '''

        pickle_data = tuple(key for key in self.__dict__ if key not in [
                            '_graph', '_session', '_model', '_tensorboard', '_learning_rate_scheduler', 'data_collector'])
        _path, filename = Agent.save(self, filename, path, data_to_save=pickle_data)
        try:
            with self._session.as_default():
                with self._graph.as_default():
                    self._model.save(Path(_path, f'{filename}.tf', filename))
        except OSError:
            os.makedirs(Path(_path, f'{filename}.tf'))
            with self._session.as_default():
                with self._graph.as_default():
                    self._model.save(Path(_path, f'{filename}.tf', filename))

        return _path, filename

    def reset(self) -> None:
        if self.training_mode:
            self._buffer_index = 0
            self._buffer_ready = False
            self._epoch += 1

    def _report(self, **kwargs) -> None:
        '''
        generate and return the requested report.

        Arguments
        ---------
            statistic: the list of items to report.

        Note: this function is not implemented yet!
        '''
        raise NotImplementedError

    def __repr__(self) -> str:
        return 'DQNAgent'
