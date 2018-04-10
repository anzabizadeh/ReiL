# -*- coding: utf-8 -*-
'''
QAgent class
=================

A Q-learning agent

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from random import choice, random

from ..valueset import ValueSet
from .agent import Agent


class QAgent(Agent):
    '''
    A Q-learning agent.

    Atributes
    ---------

    Methods
    -------
        act: return an action based on the given state.
        learn: learn using either history or action, reward, and state.
        reset: reset the agent.
        report: report the requested data.
    '''
    def __init__(self, **kwargs):
        '''
        Initialize a Q-Learning agent.

        Arguments
        ---------
            gamma: discount factor. (Default = 1.0)
            alpha: learning rate. (Default = 0.2)
            epsilon: exploration rate. (Default = 0.0)
            Rplus: optimistic estimate of the reward at each state. (Default = 0.0)
            Ne: the least number of visits. (Default = 0.0)
            default_actions: list of default actions. (Default = empty ValueSet)
            state_action_list: state action list from a previous training. (Default = {}})
        '''
        Agent.__init__(self, **kwargs)
        Agent.set_defaults(self, gamma=1, alpha=1, epsilon=0, r_plus=0, n_e=0,
                           default_actions=ValueSet(), state_action_list={})
        Agent.set_params(self, **kwargs)
        self.data_collector.available_statistics = {'states q': [False, self._report, '_state_action_list'],
                                                    'states action': [False, self._report, '_state_action_list'],
                                                    'state-actions q': [False, self._report, '_state_action_list'],
                                                    'state-actions n': [False, self._report, '_state_action_list'],
                                                    'diff-q': [True, self._report, '_state_action_list'],
                                                    'total q': [False, self._report, '_state_action_list']}
        self.data_collector.active_statistics = list(self.data_collector.available_statistics.keys())

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._gamma, self._alpha, self._epsilon, self._r_plus, self._n_e = 1, 1, 0, 0, 0
            self._default_actions, self._state_action_list = ValueSet(), {}

    def _q(self, state, action):
        '''
        Return the Q-value of a state action pair. In training mode, exploration using Ne and R+ are used, otherwise exact Q estimate is returned.

        Arguments
        ---------
            state: the state for which Q-value is returned.
            action: the action for which Q-value is returned.
        '''
        if self._training_flag:
            try:
                return self._state_action_list[state][action][0] \
                        if self._state_action_list[state][action][1] > self._n_e else self._r_plus
            except KeyError:
                return self._r_plus

        try:
            return self._state_action_list[state][action][0]
        except KeyError:
            return 0

    def _max_q(self, state):
        '''
        Return MAX(Q) of a state.
        
        Arguments
        ---------
            state: the state for which MAX(Q) is returned.
        '''
        try:
            max_q = max(self._q(state, a) for a in self._state_action_list[state])
        except (ValueError, KeyError):
            max_q = 0
        return max_q

    def _N(self, state, action):
        '''
        Return the number of times a state action pair is visited.
        Arguments
        ---------
            state: the state for which N is returned.
            action: the action for which N is returned.
        '''
        try:
            return self._state_action_list[state][action][1]
        except KeyError:
            return 0

    def act(self, state, **kwargs):
        '''
        return the best action for a given state.

        Arguments
        ---------
            state: the state for which an action is chosen.
            actions: a set of possible actions. If not provided, default actions are used.
            method: 'e-greedy' allows for exploration epsilon percent of times during training mode.
        '''
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
           (method == 'e-greedy') & (random() < self._epsilon):
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
        if not self._training_flag:
            raise ValueError('Not in training mode!')
        try:  # history
            history = kwargs['history']
            previous_state = history[0]
            for i in range(1, len(history), 3):
                previous_action = history[i]
                reward = history[i+1]
                try:
                    state = history[i+2]
                except IndexError:
                    state = None
                q_sa = self._q(previous_state, previous_action)
                max_q = self._max_q(state)

                new_N = self._N(previous_state, previous_action) + 1
                new_q = q_sa + self._alpha*(reward+self._gamma*max_q-q_sa)

                try:
                    self._state_action_list[previous_state].update({previous_action: (new_q, new_N)})
                except KeyError:
                    self._state_action_list.update({previous_state: {previous_action: (new_q, new_N)}})

                previous_state = state

            return
        except KeyError:
            pass

        try:
            state = kwargs['state']
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

        try:
            self._state_action_list[self._previous_state].update({self._previous_action: (new_q, new_N)})
        except KeyError:
            self._state_action_list.update({self._previous_state: {self._previous_action: (new_q, new_N)}})

    def reset(self):
        '''
        Resets the agent.
        
        Note: reset should be called at the end of an episode of learning when state reward pair is used in learning  
        '''
        self._previous_state = None
        self._previous_action = None

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

        if item.lower() == 'states q':
            data = kwargs['data']['_state_action_list']
            rep = {}
            for state in data:
                rep[state] = max(data[state][action][0] for action in data[state])

        if item.lower() == 'states action':
            data = kwargs['data']['_state_action_list']
            rep = {}
            all_states = list(s for s in data)
            for state in all_states:
                action_q = ((action, data[state][action][0])
                            for action in data[state])
                action = max(action_q, key=lambda x: x[1])
                rep[state] = action

        if item.lower() == 'state-actions q':
            data = kwargs['data']['_state_action_list']
            rep = dict(((state, action), data[state][action][0])
                        for state in data for action in data[state])

        if item.lower() == 'state-actions n':
            data = kwargs['data']['_state_action_list']
            rep = dict(((state, action), data[state][action][1])
                        for state in data for action in data[state])

        if item.lower() == 'diff-q':
            new = kwargs['new']['_state_action_list']
            old = kwargs['old']['_state_action_list']
            rep = 0
            state_list = set(list(new.keys()) + list(old.keys()))
            for s in state_list:
                action_list = set(list(new.get(s, {}).keys()) + list(old.get(s, {}).keys()))
                for a in action_list:
                    rep += abs(new.get(s, {}).get(a, (0, ))[0] - old.get(s, {}).get(a, (0, ))[0])

        if item.lower() == 'total q':
            data = kwargs['data']['_state_action_list']
            rep = 0
            for state in data:
                rep += sum(data[state][action][0] for action in data[state])
 
        return rep
