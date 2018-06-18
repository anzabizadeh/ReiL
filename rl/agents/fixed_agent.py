# -*- coding: utf-8 -*-
'''
FixedAgent class
=================

An agent that always chooses a fixed action

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from rl.agents import Agent


class FixedAgent(Agent):
    '''
    An agent that acts a specific action.

    Methods
    -------
        act: return the prespecified action.
    ''' 

    def __init__(self, action):
        self._action = action

    def act(self, state, **kwargs):
        '''
        Return a random action.

        Arguments
        ---------
            state: the state for which the action should be returned. This argument is solely to keep the method's signature unified.
            actions: the set of possible actions to choose from. If not provided, an empty list is returned. 
        '''
        return self._action

    def __repr__(self):
        try:
            return 'FixedAgent' & str(self._action)
        except:
            return 'FixedAgent'
