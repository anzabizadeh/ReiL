# -*- coding: utf-8 -*-
'''
PGAgent class
=================

A Policy Gradient agent with Neural Network action approximator

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''


from random import choice, random

import numpy as np
import tensorflow as tf

from rl.agents.agent import Agent
from rl.valueset import ValueSet


def main():
    from random import randint

    print('This is a simple game. a random number is generated and the agent should move left or right to get to target.')

    action_set = ValueSet([-1, 0, 1]).as_valueset_array()
    sample_agent = PGAgent(epsilon=0.1, hidden_layer_sizes=(100,), default_actions=action_set)
    sample_agent.training_mode = True
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


class PGAgent(Agent):
    '''
    A policy gradient agent with neural network action approximator.

    Methods
    -------
        act: return an action based on the given state.
        learn: learn using either history or action, reward, and state.
        reset: Not Implemented Yet!
    '''
    def __init__(self, **kwargs):
        raise NotImplementedError
        '''
        Initialize a policy gradient-based agent.

        Arguments
        ---------
            solver: the solver method. (Default = 'sgd')
            alpha: learning rate. (Default = 1e-5)
            gamma: discount factor. (Default = 1)
            hidden_layer_sizes: tuple containintg hidden layer sizes
            random_state: random state. (Default = 1)
            default_actions: list of default actions
            state_size: size of the state in its binary format (len(state.binary_representation()))
        '''
        super().__init__(**kwargs)
        super().set_defaults(gamma=1, alpha=1e-5, epsilon=0, default_actions={},
                           solver='sgd', hidden_layer_sizes=(10,), max_iter=1, random_state=None)
        super().set_params(**kwargs)
        self.data_collector.available_statistics = {'diff-coef': [True, self._report, '_clf.coefs_']}
        self.data_collector.active_statistics = ['diff-coef']

        self._input_length = kwargs['state_size']

        self._generate_network()

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._gamma, self._alpha, self._epsilon = 1, 1e-5, 0
            self._default_actions = []
            self._solver, self._hidden_layer_sizes, self._max_iter, self._random_state='sgd', (10,), 1, None


    def _generate_network(self):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._sess = tf.Session(graph=self._graph)
            layer = [0]*(len(self._hidden_layer_sizes)+1)
            self._inputs = tf.placeholder(tf.float32, [None, self._input_length], name='inputs')
            layer[0] = self._inputs
            for i, v in enumerate(self._hidden_layer_sizes):
                layer[i+1] = tf.layers.dense(layer[i], v, activation=tf.nn.relu)

            self._output = tf.layers.dense(layer[-1], len(self._default_actions))

            self._labels = tf.placeholder(tf.int32, [None, 1], name='labels')
            self._loss = tf.losses.sparse_softmax_cross_entropy(labels=self._labels, logits=self._output)
            self._global_step = tf.Variable(0, trainable=False, name="step")
            self._train_opt = tf.train.AdamOptimizer(learning_rate=self._alpha) \
                .minimize(self._loss, global_step=self._global_step)

            # log_writer = tf.summary.FileWriter("logs/" + str(time()))
            # log_writer.add_graph(self._sess.graph)
            # tf.summary.scalar("MSE_Loss", loss)

            # merged = tf.summary.merge_all()

            init = tf.global_variables_initializer()
            self._sess.run(init)

            # self._clf = MLPClassifier(solver=self._solver, alpha=self._alpha,
            #                       hidden_layer_sizes=self._hidden_layer_sizes,
            #                       random_state=self._random_state, max_iter=self._max_iter, warm_start=True)

    def learn(self, **kwargs):
        '''
        Learn either based on state reward pair or history.
        
        Arguments:
            history: a list consisting of state, action, reward of an episode. If both history and state reward provided, only history is used.
            state: state resulted from the previous action on the previous state.
            reward: the reward of the previous action.

        Raises ValueError if the agent is not in 'training' mode.
        '''
        if not self.training_mode:
            raise ValueError('Not in training mode!')
        X = np.array([], ndmin=2)
        y = np.array([], ndmin=2)
        try:  # history
            history = kwargs['history']
            if history[-1]>=0:
                previous_state = history[0]
                for i in range(1, len(history), 3):
                    previous_action = history[i]
                    # reward = history[i+1]
                    try:
                        state = history[i+2]
                    except IndexError:
                        state = None

                    state_array = previous_state.binary_representation().to_nparray()
                    action_array = self._default_actions.index(previous_action)
                    try:
                        X = np.vstack((X, state_array))
                        y = np.vstack((y, action_array))
                    except ValueError:
                        X = state_array
                        y = action_array
                    previous_state = state

                feed_dict = {self._inputs: X, self._labels: y}
                train_step = self._sess.run(self._global_step)
                self._sess.run(self._train_opt, feed_dict=feed_dict)
            else:
                # if my moves doesn't result in victory or draw, make all other moves more probable!
                previous_state = history[0]
                for i in range(1, len(history), 3):
                    previous_action = history[i]
                    # reward = history[i+1]
                    try:
                        state = history[i+2]
                    except IndexError:
                        state = None

                    state_array = [previous_state.binary_representation().to_nparray()]*(len(self._default_actions)-1)
                    action_array = list(range(len(self._default_actions)))
                    action_array.remove(self._default_actions.index(previous_action))
                    try:
                        X = np.vstack((X, state_array))
                        y = np.append(y, action_array)
                    except ValueError:
                        X = state_array
                        y = action_array
                    previous_state = state

                feed_dict = {self._inputs: X, self._labels: y.reshape(-1, 1)}
                train_step = self._sess.run(self._global_step)
                self._sess.run(self._train_opt, feed_dict=feed_dict)

            return
        except KeyError:
            pass

        try:  # state
            state = kwargs['state']
        except KeyError:
            state = None
        try:  # reward
            reward = kwargs['reward']
        except KeyError:
            reward = None

        if reward>=0:
            state_array = self._previous_state.binary_representation().to_nparray()
            action_array = self._previous_action.binary_representation().to_nparray()
            action_index = action_array.dot(1 << np.arange(action_array.size)[::-1])
            X = state_array.reshape(1, -1)
            y = np.array(action_index)
        else:
            # if my moves doesn't result in victory or draw, make all other moves more probable!
            state_array = [self._previous_state.binary_representation().to_nparray()]*(len(self._default_actions)-1)
            action_array = list(range(len(self._default_actions)))
            action_array.remove(self._default_actions.index(self._previous_action))
            X = state_array
            y = action_array

        feed_dict = {self._inputs: X, self._labels: y}
        train_step = self._sess.run(self._global_step)
        self._sess.run(self._train_opt, feed_dict=feed_dict)

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

        X = state.binary_representation().to_nparray().reshape(1, -1)
        feed_dict = {self._inputs: X}
        action_index = np.argmax(self._sess.run(self._output, feed_dict=feed_dict))
        action = self._default_actions[action_index]
        if (action not in possible_actions) | \
            ((self.training_mode) & (random() < self._epsilon)):
            action = choice(possible_actions)

        self._previous_action = action
        return action

    def _report(self, **kwargs):
        '''
        generate and return the requested report.

        Arguments
        ---------
            statistic: the list of items to report.
        '''
        try:
            item = kwargs['statistic']
        except KeyError:
            return

        if item.lower() == 'diff-coef':
            import numpy as np
            rep = 0
            for i in range(np.size(kwargs['old']['_clf.coefs_'])):
                rep += np.sum(np.subtract(kwargs['old']['_clf.coefs_'][i], kwargs['new']['_clf.coefs_'][i]))

        return rep

    def load(self, **kwargs):
        '''
        Load an object from a file.

        Arguments
        ---------
            filename: the name of the file to be loaded.

        Raises ValueError if the filename is not specified.
        '''
        Agent.load(self, **kwargs)
        self._tf = {}
        self._generate_network()
        self._tf['saver'].restore(self._tf['session'], kwargs.get('path', self._path) + '/' + kwargs['filename'] + '.tf/' + kwargs['filename'])

    def save(self, **kwargs):
        '''
        Save the object to a file.

        Arguments
        ---------
            filename: the name of the file to be saved.

        Raises ValueError if the filename is not specified.
        '''
        
        pickle_data = tuple(key for key in self.__dict__ if key not in ['_tf', 'data_collector'])
        path, filename = Agent.save(self, **kwargs, data=pickle_data)
        self._tf['saver'].save(self._tf['session'], kwargs.get('path', self._path) + '/' + kwargs['filename'] + '.tf/' + kwargs['filename'])
        return path, filename


    def __repr__(self):
        return 'PGAgent'

if __name__ == '__main__':
    main()
