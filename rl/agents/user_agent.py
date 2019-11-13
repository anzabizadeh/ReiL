# -*- coding: utf-8 -*-
'''
UserAgent class
===============

An agent that prints the state and asks the user for action.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''


from ..legacy.valueset import ValueSet
from .agent import Agent


class UserAgent(Agent):
    '''
    An agent that acts based on user input.

    Methods
    -------
        act: return user's chosen action.
    ''' 
    def act(self, state, **kwargs):
        '''
        Displays current state and asks user for input. The result is a ValueSet.
        Arguments
        ---------
            state: the state for which the action should be returned.
        '''
        action = None
        while action is None:
            action = input(f'Choose action for this state: {s}')
        return ValueSet(action)

    def __repr__(self):
        return 'UserAgent'