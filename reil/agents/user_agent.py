# -*- coding: utf-8 -*-
'''
UserAgent class
===============

An agent that prints the state and asks the user for action.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''
from typing import Any, Optional, Tuple

from reil import agents
from reil.datatypes.reildata import ReilData
from reil.utils import functions


class UserAgent(agents.NoLearnAgent):
    '''
    An agent that acts based on user input.

    ### Methods
    act: return user's chosen action.
    ''' 
    def __init__(self,
                 default_actions: Tuple[ReilData, ...] = (),
                 **kwargs: Any):
        super().__init__(default_actions=default_actions, **kwargs)

    def act(self,
            state: ReilData,
            actions: Optional[Tuple[ReilData, ...]] = None,
            epoch: int = 0) -> ReilData:
        '''
        Return an action based user input.

        ### Arguments
        state: the state for which the action should be returned.

        actions: the set of possible actions to choose from.
        '''
        possible_actions = functions.get_argument(
            actions, self._default_actions)

        action = None
        while action is None:
            for i, a in enumerate(possible_actions):
                print(f'{i}. {a.value}')
            action = input(f'Choose action for this state: {state.value}')

        return possible_actions[action]
