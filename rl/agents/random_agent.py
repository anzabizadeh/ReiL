# -*- coding: utf-8 -*-
'''
RandomAgent class
=================

An agent that randomly chooses an action

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from random import choice

from rl.agents import Agent


class RandomAgent(Agent):
    '''
    An agent that acts randomly.

    Methods
    -------
        act: return an action randomly.
    ''' 
    def act(self, state, **kwargs):
        '''
        Return a random action.

        Arguments
        ---------
            state: the state for which the action should be returned. This argument is solely to keep the method's signature unified.
            actions: the set of possible actions to choose from. If not provided, an empty list is returned. 
        '''
        try:  # possible actions
            return choice(kwargs['actions'])
        except KeyError:
            return []

