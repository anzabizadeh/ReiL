# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 19:01:29 2018

@author: Sadjad

rlagent(**kwargs): gamma, alpha, epsilon, Rplus, Ne, state_action_list
    previous_action
    act(state, **kwargs): actions, method={*'e-greedy', ''}
    learn(state, reward)
    reset()


"""

import random as rand
import pickle
from contextlib import contextmanager


def main():
    print('This is a simple game. a random number is generated and the agent should move left or right to get to target.')
    myAgent = RLAgent(gamma=.1, alpha=0.5, epsilon=0.2, Rplus=0, Ne=2,
                      default_actions=[-1, 0, 1])
    myAgent.status = 'training'
    target = rand.randint(1, 100)
    print('target={: d}'.format(target))
    for i in range(30):
        state = rand.randint(1, target)
        print('game {: d} state={: d}'.format(i, state))
        while state != target:
            action = myAgent.act(state)
            print(action)
            state = min(max(state + action, 0), target)
            myAgent.learn(state=state, reward=state-target)
        myAgent.learn(state=state, reward=1)
    print(myAgent._state_action_list)

    for state in range(1, target):
        print(myAgent.act(state, method=''))


class Agent:
    '''Super class of all agent classes. This class provides basic methods including act, learn, reset, load, save, and report Load and save are implemented.'''
    def __init__(self, **kwargs):
        self._training_flag = True

    @property
    def status(self):
        ''' Returns the status of the agent as 'training' or 'testing' '''
        if self._training_flag:
            return 'training'
        else:
            return 'testing'

    @status.setter
    def status(self, value):
        ''' Sets the status of the agent as 'training' or 'testing' '''
        self._training_flag = (value == 'training')

    def act(self, state, **kwargs):
        ''' This function gets the state and returns agent's action.
            If state is 'training' (_training_flag=false), then this function should not return any random move due to exploration.'''
        pass

    def learn(self, **kwargs):
        ''' Shoud get state, action, reward or history to learn. ''' 
        pass

    def reset(self):
        ''' Resets the agent at the end of a learning episode. ''' 
        pass

    def load(self, **kwargs):
        ''' uses pickle to load an agent using filename. '''
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')
        with open(filename + '.pkl', 'rb') as f:
            self.__dict__ = pickle.load(f)

    def save(self, **kwargs):
        ''' uses pickle to save the agent to a file. '''
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')
        with open(filename + '.pkl', 'wb+') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def report(self, **kwargs):
        ''' Should generate a report and return the string. ''' 
        raise NotImplementedError


class UserAgent(Agent):
    def act(self, state, **kwargs):
        ''' Displays current state and asks user for input. '''
        try:
            state = '\n'+kwargs['printable']+'\n'
        except KeyError:
            pass
        action = None
        while action is None:
            action = input('Choose action for this state: {}'.format(state))
        return action


class RandomAgent(Agent):
    ''' This agent acts randomly ''' 
    def act(self, state, **kwargs):
        ''' states and a set of possible actions are given to the function and it chooses an action randomly. ''' 
        try:  # possible actions
            return rand.choice(kwargs['actions'])
        except KeyError:
            return []


class NeuralAgent(Agent):
    def __init__(self, **kwargs):
        pass


class RLAgent(Agent):
    def __init__(self, **kwargs):
        '''Initializes an Reinforcement Learning Agent. \n
           Parameters:
               gamma: discount factor
               alpha: learning rate
               epsilon: exploration rate
               Rplus: optimistic estimate of the reward at each state
               Ne: the least number of visits
               default_actions: list of default actions
               state_action_list: list from a previous training
        '''
        Agent.__init__(self, **kwargs)
        try:  # discount factor
            self._gamma = kwargs['gamma']
        except KeyError:
            self._gamma = 1
        try:  # learning rate
            self._alpha = kwargs['alpha']
        except KeyError:
            self._alpha = 0.2
        try:  # exploration rate
            self._epsilon = kwargs['epsilon']
        except KeyError:
            self._epsilon = 0.0
        try:  # optimistic estimate of the reward at each state
            self._r_plus = kwargs['Rplus']
        except KeyError:
            self._r_plus = 0.0
        try:  # the least number of visits
            self._n_e = kwargs['Ne']
        except KeyError:
            self._n_e = 0.0
        try:  # list of default actions
            self._default_actions = kwargs['default_actions']
        except KeyError:
            self._default_actions = []
        try:  # state action list
            self._state_action_list = kwargs['state_action_list']
        except KeyError:
            self._state_action_list = {}

    def _q(self, state, action):
        if self._training_flag:
            try:
                if self._state_action_list[(state, action)][1] > self._n_e:
                    return self._state_action_list[(state, action)][0]
                else:
                    return self._r_plus
            except KeyError:
                return self._r_plus

        try:
            return self._state_action_list[(state, action)][0]
        except KeyError:
            return 0

    def state_q(self, state):
        return list(self._q(sa[0], sa[1]) for sa in self._state_action_list
                    if sa[0] == state)

    def _max_q(self, state):
        try:
            max_q = max(self._q(sa[0], sa[1]) for sa in self._state_action_list
                        if sa[0] == state)
        except ValueError:
            max_q = 0
        return max_q

    def _N(self, state, action):
        try:
            return self._state_action_list[(state, action)][1]
        except KeyError:
            return 0

    @property
    def previous_action(self):
        return self._previous_action

    def act(self, state, **kwargs):
        try:
            state = tuple(state)
        except TypeError:
            state = tuple([state])
        self._previous_state = state
        try:  # possible actions
            possible_actions = kwargs['actions']
        except KeyError:
            possible_actions = self._default_actions

        try:  # method
            method = kwargs['method'].lower()
        except KeyError:
            if self._epsilon > 0:
                method = 'e-greedy'
            else:
                method = ''

        if (self._training_flag) & \
           (method == 'e-greedy') & (rand.random() < self._epsilon):
            action = rand.choice(possible_actions)
        else:
            action_q = ((action, self._q(state, action))
                        for action in possible_actions)
            action = max(action_q, key=lambda x: x[1])[0]
        self._previous_action = action
        return action

    def learn(self, **kwargs):
        if not self._training_flag:
            raise ValueError('Not in training mode!')
        try:  # history
            history = kwargs['history']
            previous_state = tuple(history[0])
            for i in range(1, len(history)-2, 3):
                previous_action = history[i]
                reward = history[i+1]
                state = tuple(history[i+2])
                q_sa = self._q(previous_state, previous_action)
                max_q = self._max_q(state)

                new_N = self._N(previous_state, previous_action) + 1
                new_q = q_sa + self._alpha*(reward+self._gamma*max_q-q_sa)

                self._state_action_list.update({
                        (previous_state, previous_action): (new_q, new_N)})
                previous_state = state

            return
        except KeyError:
            pass

        try:  # state
            state = tuple(kwargs['state'])
        except TypeError:
            state = tuple([kwargs['state']])
        except KeyError:
            state = None
        try:  # reward
            reward = kwargs['reward']
        except KeyError:
            reward = None

        q_sa = self._q(self._previous_state, self._previous_action)
        max_q = self._max_q(state)

        new_N = self._N(self._previous_state, self._previous_action) + 1
        new_q = q_sa + self._alpha*(reward+self._gamma*max_q-q_sa)

        self._state_action_list.update({
                (self._previous_state, self._previous_action):
                (new_q, new_N)})

    def reset(self):
        self._previous_state = None
        self._previous_action = None


if __name__ == '__main__':
    main()
