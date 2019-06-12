# -*- coding: utf-8 -*-
'''
QAgent class
=================

A Q-learning agent for warfarin model

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from random import choice, random

from ..valueset import ValueSet
from .agent import Agent
from ..agents.q_learning import QAgent


class WarfarinQAgent(QAgent):
    '''
    A Q-learning agent for warfarin dosing. It experiences fixed trajectories first, then moves to the actual Q-learning.

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
            fixed_policy_number: number of times each fixed policy should be explored. Used when 'fixed policy first' is chosen in learn() method. (Default = 0) 
        '''
        QAgent.__init__(self, **kwargs)
        QAgent.set_defaults(self, gamma=1, alpha=1, epsilon=0, r_plus=0, n_e=0,
                           default_actions=ValueSet(), state_action_list={},
                           fixed_policy_attempts=10, current_fixed_policy_index=0, current_fixed_policy_attempts=0,
                           method='e-greedy')
        QAgent.set_params(self, **kwargs)

        if False:
            self._gamma, self._alpha, self._epsilon, self._r_plus, self._n_e = 1, 1, 0, 0, 0
            self._default_actions, self._state_action_list = ValueSet(), {}
            self._fixed_policy_attempts, self._current_fixed_policy_index, self._current_fixed_policy_attempts = 10, 0, 0
            self._method = 'e-greedy'


    def act(self, state, **kwargs):
        '''
        return the best action for a given state.

        Arguments
        ---------
            state: the state for which an action is chosen.
            actions: a set of possible actions. If not provided, default actions are used.
            method: 'e-greedy' allows for exploration epsilon percent of times during training mode.
                    'fixed policy first': check fixed policies first and then other methods.
        '''
        self._previous_state = state
        try:  # possible actions
            possible_actions = kwargs['actions']
        except KeyError:
            possible_actions = self._default_actions

        try:  # method
            method = kwargs['method'].lower()
        except KeyError:
            method = self._method

        action = None
        if (self._training_flag):
            if (self._current_fixed_policy_index>=len(self._default_actions)) \
                and (self._current_fixed_policy_index>=len(possible_actions)) \
                and (method == 'fixed policy first'):
                method = 'e-greedy'

            if (method == 'fixed policy first') and (self._current_fixed_policy_attempts<self._fixed_policy_attempts):
                try:
                    action = self._default_actions[self._current_fixed_policy_index]
                except IndexError:
                    action = possible_actions[self._current_fixed_policy_index]

            elif (method == 'e-greedy') and (random() < self._epsilon):
                action = choice(possible_actions)
        if action is None:
            action_q = ((action, self._q(state, action))
                        for action in possible_actions)
            action = max(action_q, key=lambda x: x[1])[0]
        self._previous_action = action
        return action

    def reset(self):
        '''
        Resets the agent.
        
        Note: reset should be called at the end of an episode of learning when state reward pair is used in learning  
        '''
        self._previous_state = None
        self._previous_action = None
        self._current_fixed_policy_attempts += 1
        if self._current_fixed_policy_attempts >= self._fixed_policy_attempts:
            self._current_fixed_policy_index += 1
            self._current_fixed_policy_attempts = 0


    def __repr__(self):
        return 'Warfarin QAgent'