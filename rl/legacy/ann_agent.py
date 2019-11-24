# -*- coding: utf-8 -*-
'''
ANNAgent class
=================

A Q-learning agent with Neural Network Q-function approximator

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from dill import HIGHEST_PROTOCOL, dump, load
from random import choice, random
from time import time

import numpy as np
import tensorflow as tf

from ..agents import Agent
from ..rldata import RLData


class ANNAgent(Agent):
    '''
    A Q-learning agent with neural network Q-function approximator.

    Constructor Arguments
    ---------------------
        alpha: learning rate for TD equation. (Default = 0.1)
        gamma: discount factor in TD equation. (Default = 1)
        epsilon: exploration probability. (Default = 0)
        default_actions: list of default actions.
        learning_rate: learning rate for ANN. (Default = 1e-3)
        hidden_layer_sizes: tuple containing hidden layer sizes.
        input_length: size of the input vector. (Default = 1)
        batch_size: the learning method stores inputs for batch_size iterations and then runs one ANN training. (Default = 10)

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
        Initialize a Q-Learning agent with neural network Q-function approximator.
        '''
        super().__init__(**kwargs)
        super().set_defaults(gamma=1, alpha=0.1, epsilon=0, default_actions={},
                           learning_rate=1e-3, hidden_layer_sizes=(), input_length=1,
                           training_x=np.array([], ndmin=2), training_y=np.array([], ndmin=2), buffer_size=50, batch_size=10)
        super().set_params(**kwargs)
        self.data_collector.available_statistics = {}
        self.data_collector.active_statistics = []

        self._tf = {}
        self._generate_network()

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._gamma, self._alpha, self._epsilon = 1, 0.1, 0
            self._default_actions = {}
            self._learning_rate, self._hidden_layer_sizes, self._input_length = 1e-5, (), 1
            self._batch_size, self._buffer_size, self._training_x, self._training_y = 10, 0, np.array(
                [], ndmin=2), np.array([], ndmin=2)

    def _generate_network(self):
        '''
        Generate a tensorflow ANN network.
        '''
        self._tf['graph'] = tf.Graph()
        with self._tf['graph'].as_default():
            # , config=tf.ConfigProto(log_device_placement=True))
            self._tf['session'] = tf.Session(graph=self._tf['graph'])
            self._tf['inputs'] = tf.placeholder(
                tf.float32, [None, self._input_length], name='inputs')

            layer = [self._tf['inputs']]
            for i, v in enumerate(self._hidden_layer_sizes):
                layer.append(tf.layers.dense(
                    layer[i], v, activation=tf.nn.relu, name=f'layer_{i+1:0>2}'))

            self._tf['output'] = tf.layers.dense(layer[-1], 1, name='output')

            self._tf['labels'] = tf.placeholder(
                tf.int32, [None, 1], name='labels')
            self._tf['loss'] = tf.losses.mean_squared_error(
                labels=self._tf['labels'], predictions=self._tf['output'])
            self._tf['global_step'] = tf.Variable(
                0, trainable=False, name="step")
            self._tf['train_opt'] = tf.train.AdamOptimizer(learning_rate=self._learning_rate) \
                .minimize(self._tf['loss'], global_step=self._tf['global_step'], name='train_opt')

            hparam = '_'.join(('gma', str(self._gamma), 'alf', str(self._alpha), 'eps', 'func' if callable(self._epsilon) else str(self._epsilon),
                                                        'lrn', str(self._learning_rate), 'hddn', str(
                                                            self._hidden_layer_sizes),
                                                        'btch', str(self._batch_size)))

            self._tf['log_writer'] = tf.summary.FileWriter("logs/" + hparam)
            self._tf['log_writer'].add_graph(self._tf['session'].graph)
            tf.summary.scalar("MSE_Loss", self._tf['loss'])

            self._tf['merged'] = tf.summary.merge_all()

            init_g = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            self._tf['session'].run(init_g)
            self._tf['session'].run(init_l)
            self._tf['saver'] = tf.train.Saver()

    def _q(self, state, action):
        '''
        Return the Q-value of a state action pair.

        Arguments
        ---------
            state: the state for which Q-value is returned. (Can be a list of states)
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

        feed_dict = {self._tf['inputs']: X}
        result = self._tf['session'].run(
            self._tf['output'], feed_dict=feed_dict)
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
            previous_state = history.at[0, 'state']
            for i in range(len(history.index)):
                previous_action = history.at[i, 'action']
                reward = history.at[i, 'reward']
                try:
                    state = history.at[i+1, 'state']
                    max_q = self._max_q(state)
                    new_q = reward + self._gamma*max_q
                except KeyError:
                    new_q = reward

                state_action = np.append(previous_state.normalize(
                ).as_nparray(), previous_action.normalize().as_nparray())
                try:
                    self._training_x = np.vstack(
                        (self._training_x, state_action))
                    self._training_y = np.vstack((self._training_y, new_q))
                except ValueError:
                    self._training_x = state_action
                    self._training_y = np.array(new_q)
                previous_state = state

            buffered_size = len(self._training_x)
            if buffered_size >= self._buffer_size:
                index = np.random.choice(buffered_size, self._batch_size)
                feed_dict = {self._tf['inputs']: self._training_x[index],
                             self._tf['labels']: self._training_y[index]}
                self._train_step = self._tf['session'].run(
                    self._tf['global_step'])
                self._tf['session'].run(
                    self._tf['train_opt'], feed_dict=feed_dict)
                summary = self._tf['session'].run(
                    self._tf['merged'], feed_dict=feed_dict)
                self._tf['log_writer'].add_summary(summary, self._train_step)

                # self._training_x = np.array([], ndmin=2)
                # self._training_y = np.array([], ndmin=2)
                self._training_x = np.delete(self._training_x, range(
                    buffered_size-self._buffer_size), axis=0)
                self._training_y = np.delete(self._training_y, range(
                    buffered_size-self._buffer_size), axis=0)

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
            # result = tuple(possible_actions[i] for i in range(len(possible_actions)) if q_values[i]==max_q)

        action = choice(result)
            # result = max(((possible_actions[i], q_values[i]) for i in range(len(possible_actions))), key=lambda x: x[1])
            # action_q = ((action, self._q(state, action))
            #             for action in possible_actions)
            # result = max(action_q, key=lambda x: x[1])
            # action = result[0]
        # print(result)
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
        self._tf = {}
        self._generate_network()
        self._tf['saver'].restore(self._tf['session'], kwargs.get(
            'path', self._path) + '/' + kwargs['filename'] + '.tf/' + kwargs['filename'])

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
                            '_tf', 'data_collector'])
        path, filename = Agent.save(self, **kwargs, data=pickle_data)
        self._tf['saver'].save(self._tf['session'], kwargs.get(
            'path', self._path) + '/' + kwargs['filename'] + '.tf/' + kwargs['filename'])
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

    def __del__(self):
        self._tf['log_writer'].close()
        self._tf['session'].close()

    def __repr__(self):
        try:
            return 'ANNAgent: ' + '_'.join(('gma', str(self._gamma), 'alf', str(self._alpha), 'eps', str(self._epsilon),
                                            'lrn', str(self._learning_rate), 'hddn', str(
                self._hidden_layer_sizes),
                'btch', str(self._batch_size)))
        except AttributeError:
            return 'ANNAgent: New'
