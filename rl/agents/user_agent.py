# -*- coding: utf-8 -*-
'''
UserAgent class
===============

An agent that prints the state and asks the user for action.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

# KNOWN ISSUES:
# NOTE TO MYSELF: This implementation is not good! It should get either the state or the printable.


from ..valueset import ValueSet
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
            printable: user friendly format of the state. If not supplied, the state is printed as a list.
        '''
        try:
            s = '\n'+kwargs['printable']+'\n'
        except KeyError:
            s = state.to_list()
        action = None
        while action is None:
            action = input('Choose action for this state: {}'.format(s))
        return ValueSet(action)
