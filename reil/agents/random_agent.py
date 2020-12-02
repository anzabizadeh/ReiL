# -*- coding: utf-8 -*-
'''
RandomAgent class
=================

An agent that randomly chooses an action

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import random
from typing import Any, Optional, Tuple

from reil import agents
from reil.datatypes.reildata import ReilData
from reil.utils import functions


class RandomAgent(agents.NoLearnAgent):
    '''
    An agent that acts randomly.

    ### Methods
    act: return an action randomly.
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
        Return a random action.

        ### Arguments
        state: the state for which the action should be returned.

        actions: the set of possible actions to choose from.
        '''
        return random.choice(functions.get_argument(
            actions, self._default_actions))
