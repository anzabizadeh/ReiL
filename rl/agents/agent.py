# -*- coding: utf-8 -*-
'''
Agent class
=============

This `agent` class is the super class of all agent classes. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from rl import rldata
from typing import Any, Dict, Optional, Sequence

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
                 ex_protocol_options: Dict[str, Sequence[str]] = {'mode': ['training', 'test']},
                 **kwargs: Any):

        kwargs['name'] = kwargs.get('name', __name__)
        kwargs['logger_name'] = kwargs.get('logger_name', __name__)

        self.training_mode: bool = kwargs.get('training_mode', False)

        super().__init__(ex_protocol_current=ex_protocol_current,
                         ex_protocol_options=ex_protocol_options,
                         **kwargs)

    @property
    def status(self) -> str:
        '''Return the status of the agent as 'training' or 'test'.'''
        if self.training_mode:
            return 'training'
        else:
            return 'test'

    @status.setter
    def status(self, value: str) -> None:
        '''
        Set the status of the agent as 'training' or 'test'.
        '''
        self.training_mode = (value == 'training')

    def act(self,
        state: rldata.RLData,
        actions: Optional[Sequence[rldata.RLData]] = None,
        episode: Optional[int] = 0) -> rldata.RLData:
        '''
        Return an action based on the given state.

        Arguments
        ---------
            state: the state for which the action should be returned.
            actions: the set of possible actions to choose from.

        Note: If state is 'training' (_training_flag=false), then this function should not return any random move due to exploration.
        '''
        return rldata.RLData({'reward': {'value': 0.0}})

    def learn(self, history: Optional[rlbase.History] = None,
        observation: Optional[rlbase.Observation] = None) -> None:
        '''Learn using either history or action, reward, and state.'''
        pass

    def reset(self):
        '''Reset the agent at the end of a learning episode.'''
        pass

    def __repr__(self):
        return 'Agent'
