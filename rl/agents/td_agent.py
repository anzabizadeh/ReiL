# -*- coding: utf-8 -*-
'''
TD0Agent class
=================

A basic temporal difference agent

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from random import choice, random

from ..valueset import ValueSet
from .agent import Agent


class TD0Agent(Agent):
    '''
    A basic temporal difference learning agent. (a.k.a. SARSA)

    Atributes
    ---------

    Methods
    -------
        act: return an action based on the given state.
        learn: learn using either history or action, reward, and state.
        reset: reset the agent.
    '''
    def __init__(self, **kwargs):
        '''
        Initialize a TD(0) learning agent.

        Arguments
        ---------
            gamma: discount factor. (Default = 1.0)
            alpha: learning rate. (Default = 0.2)
            epsilon: exploration rate. (Default = 0.0)
            default_actions: list of default actions. (Default = empty ValueSet)
            state_action_list: state action list from a previous training. (Default = {}})
        '''
        Agent.__init__(self, **kwargs)
        Agent.set_defaults(self, gamma=1, alpha=1, epsilon=0, default_actions=ValueSet(), state_action_list={})
        Agent.set_params(self, **kwargs)

        self.data_collector.available_statistics = {'states q': [False, self.__report, '_state_action_list'],
                                                    'states action': [False, self.__report, '_state_action_list'],
                                                    'state-actions q': [False, self.__report, '_state_action_list'],
                                                    'diff-q': [True, self.__report, '_state_action_list']}
        self.data_collector.active_statistics = ['states q', 'states action', 'state-actions q', 'diff-q']

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._gamma, self._alpha, self._epsilon = 1, 0.2, 0.0
            self._default_actions, self._state_action_list = ValueSet(), {}

    def _q(self, state, action):
        '''
        Return the Q-value of a state action pair.

        Arguments
        ---------
            state: the state for which Q-value is returned.
            action: the action for which Q-value is returned.
        '''
        try:
            return self._state_action_list[(state, action)]
        except KeyError:
            return 0

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

        if random() < self._epsilon:
            action = choice(possible_actions)
        else:
            action_q = ((action, self._q(state, action))
                        for action in possible_actions)
            action = max(action_q, key=lambda x: x[1])[0]
        self._previous_action = action
        return action

    def learn(self, **kwargs):
        '''
        Learn either based on state reward pair or history.
        
        Arguments:
            history: a list consisting of state, action, reward of an episode. If both history and state reward provided, only history is used.
            state: state resulted from the previous action on the previous state.
            reward: the reward of the previous action.

        Raises ValueError if the agent is not in 'training' mode.
        '''
        try:  # history
            history = kwargs['history']
            previous_state = history[0]
            for i in range(1, len(history), 3):
                previous_action = history[i]
                reward = history[i+1]
                try:
                    state = history[i+2]
                    action = history[i+3]
                except IndexError:
                    state = None
                    action = None

                q_sa = self._q(previous_state, previous_action)
                q_sa2 = self._q(state, action)

                new_q = q_sa + self._alpha*(reward + self._gamma*q_sa2 - q_sa)

                self._state_action_list.update({
                        (previous_state, previous_action): new_q})
                previous_state = state

            return
        except KeyError:
            pass

        try:
            state = kwargs['state']
        except KeyError:
            state = None
        try:
            action = kwargs['action']
        except KeyError:
            action = None
        try:  # reward
            reward = kwargs['reward']
        except KeyError:
            reward = None

        q_sa = self._q(self._previous_state, self._previous_action)
        q_sa2 = self._q(state, action)

        new_q = q_sa + self._alpha*(reward + self._gamma*q_sa2 - q_sa)

        self._state_action_list.update({
                (self._previous_state, self._previous_action): new_q})

    def reset(self):
        '''
        Resets the agent.
        
        Note: reset should be called at the end of an episode of learning when state reward pair is used in learning  
        '''
        self._previous_state = None
        self._previous_action = None

    def _max_q(self, state):
        '''
        Return MAX(Q) of a state.
        
        Arguments
        ---------
            state: the state for which MAX(Q) is returned.
        '''
        try:
            max_q = max(self._q(sa[0], sa[1]) for sa in self._state_action_list
                        if sa[0] == state)
        except ValueError:
            max_q = 0
        return max_q

    def __report(self, **kwargs):
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

        if item.lower() == 'states q':
            rep = dict((sa[0], self._max_q(sa[0])) for sa in kwargs['data']['_state_action_list'])
        if item.lower() == 'states action':
            rep = {}
            all_states = set(sa[0] for sa in kwargs['data']['_state_action_list'])
            for state in all_states:
                action_q = ((sa[1], self._q(sa[0], sa[1]))
                            for sa in kwargs['data']['_state_action_list'] if sa[0] == state)
                action = max(action_q, key=lambda x: x[1])
                rep[state] = action
        if item.lower() == 'state-actions q':
            rep = dict(((sa[0], sa[1]), self._q(sa[0], sa[1])) for sa in kwargs['data']['_state_action_list'])

        if item.lower() == 'diff-q':
            list_old = dict((sa[0], self._q(sa[0], sa[1])) for sa in kwargs['old']['_state_action_list'])
            list_new = dict((sa[0], self._q(sa[0], sa[1])) for sa in kwargs['new']['_state_action_list'])
            rep = 0
            for item in set(list(list_old.keys())+list(list_new.keys())):
                rep += abs(list_new.get(item, 0) - list_old.get(item, 0))

        return rep
