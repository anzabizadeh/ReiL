# -*- coding: utf-8 -*-
'''
Agent class
=============

This `agent` class is the super class of all agent classes. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from ..rlbase import RLBase


def main():
    pass

class Agent(RLBase):
    '''
    Super class of all agent classes.
    
    Attributes
    ----------
        status: return the status of the agent

    Methods
    -------
        act: return an action based on the given state.
        learn: learn using either history or action, reward, and state.
        reset: reset the agent.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        super().set_defaults(training_flag=True)
        super().set_params(**kwargs)

    @property
    def status(self):
        '''Return the status of the agent as 'training' or 'testing'.'''
        if self._training_flag:
            return 'training'
        else:
            return 'testing'

    @status.setter
    def status(self, value):
        '''
        Set the status of the agent as 'training' or 'testing'.
        '''
        self._training_flag = (value == 'training')

    def act(self, state, **kwargs):
        '''
        Return an action based on the given state.

        Arguments
        ---------
            state: the state for which the action should be returned.
            actions: the set of possible actions to choose from.

        Note: If state is 'training' (_training_flag=false), then this function should not return any random move due to exploration.
        '''
        pass

    def learn(self, **kwargs):
        '''Learn using either history or action, reward, and state.''' 
        pass

    def reset(self):
        '''Reset the agent at the end of a learning episode.''' 
        pass

    def __repr__(self):
        return 'Agent'