# -*- coding: utf-8 -*-
'''
ANNAgent class
=================

A Q-learning agent with Neural Network Q-function approximator

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''


from pickle import load, dump, HIGHEST_PROTOCOL
from random import choice, random
from time import time

import numpy as np
import tensorflow as tf

from rl.agents.agent import Agent


def main():
    from rl.valueset import ValueSet
    from random import randint

    print('This is a simple game. a random number is generated and the agent should move left or right to get to target.')

    action_set = ValueSet([-1, 0, 1]).as_valueset_array()
    sample_agent = ANNAgent(epsilon=0.1, hidden_layer_sizes=(100,), default_actions=action_set, state_size=3)
    sample_agent.save(filename='test')
    sample_agent.load(filename='test')
    sample_agent.status = 'training'
    state = ValueSet()
    state.min = 0
    state.max = 100
    action = ValueSet()
    action.min = -1
    action.max = 1
    target = ValueSet(randint(1, 100))
    print('target=', target.to_list())
    for i in range(30):
        state.value = randint(1, target.to_list()[0])
        print('game ', i, ' state= ', state.to_list())
        max_iteration = 50
        while state != target:
            history = []
            itr = 0
            while itr <= max_iteration:
                itr += 1
                history.append(state)
                action.value = sample_agent.act(state).value
                history.append(action)
                print(state.to_list()[0], ' -> ', action.to_list()[0], end=', ')
                state.value = min(max(state.value[0] + action.value[0], 0), target.value[0])
                reward = state.to_list()[0]-target.to_list()[0]
                history.append(reward)
        # sample_agent.learn(state=state, reward=1)
            sample_agent.learn(history=history)
    
    for state in range(1, target):
        print(sample_agent.act(state, method=''))


class ANNAgent(Agent):
    '''
    A Q-learning agent with neural network Q-function approximator.

    Methods
    -------
        act: return an action based on the given state.
        learn: learn using either history or action, reward, and state.
        reset: Not Implemented Yet!
    '''
    def __init__(self, **kwargs):
        '''
        Initialize a Q-Learning agent with neural network Q-function approximator.

        Arguments
        ---------
            alpha: learning rate. (Default = 1e-5)
            gamma: discount factor. (Default = 1)
            hidden_layer_sizes: tuple containintg hidden layer sizes
            random_state: random state. (Default = 1)
            default_actions: list of default actions
        '''
        Agent.__init__(self, **kwargs)
        Agent.set_defaults(self, gamma=1, alpha=0.1, epsilon=0, default_actions={},
                           learning_rate=1e-5, hidden_layer_sizes=(10,),
                           training_x=np.array([], ndmin=2), training_y=np.array([], ndmin=2), current_run=0, batch_size=10)
        Agent.set_params(self, **kwargs)
        self.data_collector.available_statistics = {'report': [True, self._report, '_dummy']}
        self.data_collector.active_statistics = ['report']

        self._input_length = kwargs['state_size']

        self._tf = {}
        self._generate_network()

        # NOTE: here dummy is a dummy variable to pass to data_collector
        self._dummy=0

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._gamma, self._alpha, self._epsilon = 1, 0.1, 0
            self._default_actions = {}
            self._learning_rate, self._hidden_layer_sizes=1e-5, (10,)
            self._batch_size, self._current_run, self._training_x, self._training_y = 10, 0, np.array([], ndmin=2), np.array([], ndmin=2)

    def _generate_network(self):
        self._tf['graph'] = tf.Graph()
        with self._tf['graph'].as_default():
            self._tf['session'] = tf.Session(graph=self._tf['graph'])  # , config=tf.ConfigProto(log_device_placement=True))
            self._tf['inputs'] = tf.placeholder(tf.float32, [None, self._input_length], name='inputs')

            layer = [self._tf['inputs']]
            for i, v in enumerate(self._hidden_layer_sizes):
                layer.append(tf.layers.dense(layer[i], v, activation=tf.nn.relu, name='layer_{:0>2}'.format(i+1)))

            self._tf['output'] = tf.layers.dense(layer[-1], 1, name='output')

            self._tf['labels'] = tf.placeholder(tf.int32, [None, 1], name='labels')
            self._tf['loss'] = tf.losses.mean_squared_error(labels=self._tf['labels'], predictions=self._tf['output'])
            self._tf['global_step'] = tf.Variable(0, trainable=False, name="step")
            self._tf['train_opt'] = tf.train.AdamOptimizer(learning_rate=self._learning_rate) \
                .minimize(self._tf['loss'], global_step=self._tf['global_step'], name='train_opt')

            hparam = '_'.join(('gma', str(self._gamma), 'alf', str(self._alpha), 'eps', str(self._epsilon),
                               'lrn', str(self._learning_rate), 'hddn', str(self._hidden_layer_sizes),
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
            state: the state for which Q-value is returned.
            action: the action for which Q-value is returned.
        '''
        X = np.append(state.binary_representation().to_nparray(), action.binary_representation().to_nparray())
        X = X.reshape(1, -1)
        feed_dict = {self._tf['inputs']: X}
        result = self._tf['session'].run(self._tf['output'], feed_dict=feed_dict)
        return result

    def _max_q(self, state):
        '''
        Return MAX(Q) of a state.
        
        Arguments
        ---------
            state: the state for which MAX(Q) is returned.
        '''
        try:
            max_q = max(self._q(state, action) for action in self._default_actions)
        except ValueError:
            max_q = 0
        return max_q

    def learn(self, **kwargs):
        '''
        Learn either based on state reward pair or history.
        
        Arguments:
            history: a list consisting of state, action, reward of an episode. If both history and state reward provided, only history is used.
            state: state resulted from the previous action on the previous state.
            reward: the reward of the previous action.

        Raises ValueError if the agent is not in 'training' mode.
        '''
        if not self._training_flag:
            raise ValueError('Not in training mode!')
        # X = np.array([], ndmin=2)
        # y = np.array([], ndmin=2)
        try:  # history
            history = kwargs['history']
            previous_state = history[0]
            for i in range(1, len(history), 3):
                previous_action = history[i]
                reward = history[i+1]
                q_sa = self._q(previous_state, previous_action)
                try:
                    state = history[i+2]
                    max_q = self._max_q(state)
                    new_q = q_sa + self._alpha*(reward+self._gamma*max_q-q_sa)
                except IndexError:
                    new_q = reward
                    # new_q = q_sa + self._alpha*(reward-q_sa)
                state_action = np.append(previous_state.binary_representation().to_nparray(), previous_action.binary_representation().to_nparray())
                try:
                    self._training_x = np.vstack((self._training_x, state_action))
                    self._training_y = np.vstack((self._training_y, new_q))
                except ValueError:
                    self._training_x = state_action
                    self._training_y = np.array(new_q)
                previous_state = state

            self._current_run += 1
            if self._current_run == self._batch_size:
                feed_dict = {self._tf['inputs']: self._training_x, self._tf['labels']: self._training_y}
                self._train_step = self._tf['session'].run(self._tf['global_step'])
                self._tf['session'].run(self._tf['train_opt'], feed_dict=feed_dict)
                summary = self._tf['session'].run(self._tf['merged'], feed_dict=feed_dict)
                self._tf['log_writer'].add_summary(summary, self._train_step)
                self._training_x = np.array([], ndmin=2)
                self._training_y = np.array([], ndmin=2)
                self._current_run = 0

            return
        except KeyError:
            raise RuntimeError('ANNAgent only works using \'history\'')

        # try:  # state
        #     state = kwargs['state']
        # except KeyError:
        #     state = None
        # try:  # reward
        #     reward = kwargs['reward']
        # except KeyError:
        #     reward = None

        # q_sa = self._q(self._previous_state, self._previous_action)
        # max_q = self._max_q(state)
        # new_q = q_sa + self._alpha*(reward+self._gamma*max_q-q_sa)

        # state_action = np.append(self._previous_state.binary_representation().to_nparray(), 
        #                             self._previous_action.binary_representation().to_nparray())
        # X = state_action.reshape(1, -1)
        # y = np.array([new_q])
        # self._previous_state = state

        # feed_dict = {self._inputs: X, self._labels: y}
        # self._train_step = self._sess.run(self._global_step)
        # self._sess.run(self._train_opt, feed_dict=feed_dict)
        # summary = self._sess.run(self._merged, feed_dict=feed_dict)
        # self._log_writer.add_summary(summary, self._train_step)

    def act(self, state, **kwargs):
        '''
        return the best action for a given state.

        Arguments
        ---------
            state: the state for which an action is chosen.
            actions: a set of possible actions. If not provided, default actions are used.
        '''
        self._previous_state = state
        try:  # possible actions
            possible_actions = kwargs['actions']
        except KeyError:
            possible_actions = self._default_actions


        if (self._training_flag) & (random() < self._epsilon):
            action = choice(possible_actions)
        else:
            action_q = ((action, self._q(state, action))
                        for action in possible_actions)
            result = max(action_q, key=lambda x: x[1])
            action = result[0]
        # print(result)
        self._previous_action = action
        return action

    def load(self, **kwargs):
        '''
        Load an object from a file.

        Arguments
        ---------
            filename: the name of the file to be loaded.

        Raises ValueError if the filename is not specified.
        '''
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')

        with open(filename + '.pkl', 'rb') as f:
            self.__dict__ = load(f)

        self._tf = {}
        self._generate_network()
        self._tf['saver'].restore(self._tf['session'], './tf/'+filename)


    def save(self, **kwargs):
        '''
        Save the object to a file.

        Arguments
        ---------
            filename: the name of the file to be saved.

        Raises ValueError if the filename is not specified.
        '''
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')
        
        pickle_data = dict((key, value) for key, value in self.__dict__.items() if key not in ['_tf', 'data_collector'])
        # for k, v in self.__dict__.items():
        #     if not isinstance(v, (tf.Tensor, tf.SparseTensor, tf.Variable, tf.Graph)):
        #         pickle_data[k] = v

        with open(filename + '.pkl', 'wb+') as f:
            dump(pickle_data, f, HIGHEST_PROTOCOL)

        self._tf['saver'].save(self._tf['session'], './tf/'+filename)

    def _report(self, **kwargs):
        '''
        generate and return the requested report.

        Arguments
        ---------
            statistic: the list of items to report.
        '''
        # try:
        #     item = kwargs['statistic']
        # except KeyError:
        #     return

        # if item.lower() == 'diff-coef':
        #     import numpy as np
        #     rep = 0
        #     for i in range(np.size(kwargs['old']['_clf.coefs_'])):
        #         rep += np.sum(np.subtract(kwargs['old']['_clf.coefs_'][i], kwargs['new']['_clf.coefs_'][i]))

        summary = self._tf['session'].run(self._tf['merged'], feed_dict={})
        self._tf['log_writer'].add_summary(summary, self._train_step)

        return summary

    def __del__(self):
        self._tf['log_writer'].close()

    def __repr__(self):
        return 'ANNAgent'

if __name__ == '__main__':
    main()
