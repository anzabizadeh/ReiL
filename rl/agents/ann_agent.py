# -*- coding: utf-8 -*-
"""
The :mod:`agents` provides the super class and some classes for reinforcement learning.

Classes
-------
    - `Agent`: the super class of all agent classes
    - `UserAgent`: an agent that shows asks for user's choice of action
    - `RandomAgent`: an agent that randomly chooses an action
    - `RLAgent`: the Q-learning agent
    - `NeuralAgent`: the agent with neural network as a  (Not Implemented Yet)

@author: Sadjad Anzabi Zadeh
-----------------------------------

Classes:
    Agent (Super Class)
"""

# KNOWN ISSUES:
# Implement NeuralAgent.

import numpy as np
from sklearn import exceptions
from sklearn.neural_network import MLPRegressor

from .agent import Agent


def main():
    from ..valueset import ValueSet
    from random import randint

    print('This is a simple game. a random number is generated and the agent should move left or right to get to target.')
    # sample_agent = RLAgent(gamma=.1, alpha=0.5, epsilon=0.2, Rplus=0, Ne=2,
    #                   default_actions=[-1, 0, 1])
    # sample_agent.status = 'training'
    sample_agent = ANNAgent(hidden_layer_sizes=(100,), default_actions=[ValueSet(-1), ValueSet(0), ValueSet(1)])
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
    This class should implement a Neural Network learner. NOT FULLY IMPLEMENTED YET!
    '''
    def __init__(self, **kwargs):
        '''
        Initializes an Reinforcement Learning Agent.
        \nArguments:
        \n    solver: the solver method. Default = 'lbfgs'
        \n    alpha: learning rate. Default = 1e-5
        \n    gamma: discount factor. Default = 1
        \n    hidden_layer_sizes: tuple containintg hidden layer sizes
        \n    random_state: random state. Default = 1
        \n    default_actions: list of default actions
        \n    state_action_list: list from a previous training
        '''
        Agent.__init__(self, **kwargs)
        try:
            self._solver = kwargs['solver']
        except KeyError:
            self._solver = 'lbfgs'
        try:
            self._alpha = kwargs['alpha']
        except KeyError:
            self._alpha = 1e-5
        try:
            self._gamma = kwargs['gamma']
        except KeyError:
            self._gamma = 1
        try:
            self._hidden_layer_sizes = kwargs['hidden_layer_sizes']
        except KeyError:
            self._hidden_layer_sizes = (10)
        try:
            self._random_state = kwargs['random_state']
        except KeyError:
            self._random_state = 1
        try:
            self._default_actions = kwargs['default_actions']
        except KeyError:
            raise ValueError('default_actions should be specified for Neural Agent')
        try:
            self._state_action_list = kwargs['state_action_list']
        except KeyError:
            self._state_action_list = {}
        self._clf = MLPRegressor(solver=self._solver, alpha=self._alpha, hidden_layer_sizes=self._hidden_layer_sizes,
                                 random_state=self._random_state, max_iter=2, warm_start=True)

    def _q(self, state, action):
        X = np.append(state.binary_representation().to_nparray(), action.binary_representation().to_nparray())
        X = X.reshape(1, -1)
        try:
            result = self._clf.predict(X)
            # print(result, end=' ')
            return result
        except exceptions.NotFittedError:
            return 0

    def _max_q(self, state):
        '''
        Returns MAX(Q) of a state.
        \nArguments:
        \n    state: the state for which MAX(Q) is returned.
        '''
        try:
            max_q = max(self._q(state, action) for action in self._default_actions)
        except ValueError:
            max_q = 0
        return max_q

    def learn(self, **kwargs):
        '''
        Learns either based on state reward pair or history. Agent should be in 'training' mode. Otherwise, a ValueError exception is raised.
        \nArguments:
        \n    history: a list consisting of state, action, reward of an episode. If both history and state reward provided, only history is used.
        \n    state: state resulted from the previous action on the previous state.
        \n    reward: the reward of the previous action.
        '''
        if not self._training_flag:
            raise ValueError('Not in training mode!')
        X = np.array([], ndmin=2)
        y = np.array([], ndmin=2)
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
                    X = np.vstack((X, state_action))
                    y = np.append(y, new_q)
                except ValueError:
                    X = state_action
                    y = np.array([new_q])
                previous_state = state

            # try:
            #     old_q = self._clf.predict(X)
            #     old_coef = self._clf.coefs_
            #     self._clf.fit(X, y)
            #     print('before training:')
            #     print(old_coef)
            #     print('after training:')
            #     print(self._clf.coefs_)
            #     # diff_q = old_q - self._clf.predict(X)
            #     # print('{: 2.2f}'.format(sum(diff_q)), end='\t')
            # except AttributeError:
            self._clf.fit(X, y)

            return
        except KeyError:
            pass

        # try:  # state
        #     state = tuple(kwargs['state'])
        # except TypeError:
        #     state = tuple([kwargs['state']])
        # except KeyError:
        #     state = None
        # try:  # reward
        #     reward = kwargs['reward']
        # except KeyError:
        #     reward = None

        # q_sa = self._q(self._previous_state, self._previous_action)
        # max_q = self._max_q(state)

        # new_N = self._N(self._previous_state, self._previous_action) + 1
        # new_q = q_sa + self._alpha*(reward+self._gamma*max_q-q_sa)

        # self._state_action_list.update({
        #         (self._previous_state, self._previous_action):
        #         (new_q, new_N)})

    def act(self, state, **kwargs):
        '''
        Gets a state and returns the best action.
        \nArguments:
        \n    actions: a set of possible actions.
        '''
        self._previous_state = state.value
        try:  # possible actions
            possible_actions = kwargs['actions']
        except KeyError:
            possible_actions = self._default_actions

        action_q = ((action, self._q(state, action))
                    for action in possible_actions)
        result = max(action_q, key=lambda x: x[1])
        action = result[0]
        # print(result)
        self._previous_action = action
        return action


if __name__ == '__main__':
    main()
