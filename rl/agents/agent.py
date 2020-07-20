# -*- coding: utf-8 -*-
'''
Agent class
=============

This `agent` class is the super class of all agent classes. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from typing import Dict, List, Optional

from rl import rlbase


class Agent(rlbase.RLBase):
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

    def __init__(self,
                 ex_protocol_current: Dict[str, str] = {'mode': 'training'},
                 ex_protocol_options: Dict[str, List[str]] = {'mode': ['training', 'test']},
                 **kwargs):

        super().__init__(name=kwargs.get('name', __name__),
                         logger_name=kwargs.get('logger_name', __name__),
                         ex_protocol_current=ex_protocol_current,
                         ex_protocol_options=ex_protocol_options,
                         **kwargs)

    @property
    def status(self):
        '''Return the status of the agent as 'training' or 'test'.'''
        if self.training_mode:
            return 'training'
        else:
            return 'test'

    @status.setter
    def status(self, value):
        '''
        Set the status of the agent as 'training' or 'test'.
        '''
        self.training_mode = (value == 'training')

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

    def learn(self, observation: Optional[rlbase.Observation] = None,
              history: Optional[rlbase.History] = None):
        '''Learn using either history or action, reward, and state.'''
        pass

    def reset(self):
        '''Reset the agent at the end of a learning episode.'''
        pass

    def __repr__(self):
        return 'Agent'
