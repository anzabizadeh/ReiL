# -*- coding: utf-8 -*-
'''
Agent class
=============

This `agent` class is the super class of all agent classes. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from logging import WARNING
from typing import Dict, List, Optional, Sequence

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
    def __init__(self,
                 name: str = 'agent',
                 version: float = 0.5,
                 path: str = '.',
                 ex_protocol_current: Dict[str, str] = {'mode': 'training'},
                 ex_protocol_options: Dict[str, List[str]] = {'mode': ['training', 'test']},
                 stats_list: Sequence[str] = [],
                 logger_name: str = __name__,
                 logger_level: int = WARNING,
                 logger_filename: Optional[str] = None,
                 persistent_attributes: List[str] = []):

        super().__init__(name=name,
                         version=version,
                         path=path,
                         ex_protocol_current=ex_protocol_current,
                         ex_protocol_options=ex_protocol_options,
                         stats_list=stats_list,
                         logger_name=logger_name,
                         logger_level=logger_level,
                         logger_filename=logger_filename,
                         persistent_attributes=persistent_attributes)

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

    def learn(self, **kwargs):
        '''Learn using either history or action, reward, and state.''' 
        pass

    def reset(self):
        '''Reset the agent at the end of a learning episode.''' 
        pass

    def __repr__(self):
        return 'Agent'
