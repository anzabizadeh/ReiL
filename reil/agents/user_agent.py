# -*- coding: utf-8 -*-
'''
UserAgent class
===============

An agent that prints the state and asks the user for action.
'''
from typing import Any, Optional, Tuple

from reil import agents
from reil.datatypes import FeatureArray


class UserAgent(agents.NoLearnAgent):
    '''
    An agent that acts based on user input.
    '''

    def __init__(self,
                 default_actions: Tuple[FeatureArray, ...] = (),
                 **kwargs: Any):
        super().__init__(default_actions=default_actions, **kwargs)

    def act(self,
            state: FeatureArray,
            actions: Optional[Tuple[FeatureArray, ...]] = None,
            epoch: int = 0) -> FeatureArray:
        '''
        Return a random action.

        Arguments
        ---------
        state:
            The state for which the action should be returned.

        actions:
            The set of possible actions to choose from.

        epoch:
            The epoch in which the agent is acting.

        Returns
        -------
        :
            The action
        '''
        possible_actions = actions or self._default_actions

        action = None
        while action is None:
            for i, a in enumerate(possible_actions):
                print(f'{i}. {a.value}')
            action = int(input(
                f'Choose action number for this state: {state.value}'))

        return possible_actions[action]
