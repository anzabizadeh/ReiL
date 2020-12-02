# -*- coding: utf-8 -*-
'''
NoLearnAgent class
==================

This `NoLearnAgent` class is the base class of all agent classes. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import random
from typing import Any, List, Optional, Tuple, TypeVar

from reil import stateful
from reil.datatypes.reildata import ReilData
from reil.utils import functions
from typing_extensions import Literal

TrainingData = Tuple[List[ReilData], List[float]]
T = TypeVar('T')

class NoLearnAgent(stateful.Stateful):
    '''
    The base class of all `agent` classes. This class does not support any
    `learner`.

    ### Methods
    act: returns an `action` based on the given `state`.

    best_actions: returns a list of best `action`s for a given `state`.
    '''

    def __init__(self,
                 default_actions: Tuple[ReilData, ...] = (),
                 tie_breaker: Literal['first', 'last', 'random'] = 'random',
                 **kwargs: Any):
        '''
        Initializes the `agent`.

        ### Arguments
        default_actions: a tuple of default actions.

        tie_breaker: how to choose the `action` if more than one is candidate
        to be chosen.
        '''
        super().__init__(**kwargs)

        self._default_actions = default_actions

        self.training_mode = False
        if tie_breaker not in ['first', 'last', 'random']:
            raise ValueError(
                'Tie breaker should be one of first, last, or random options.')
        self._tie_breaker = tie_breaker

    def act(self,
            state: ReilData,
            actions: Optional[Tuple[ReilData, ...]] = None,
            epoch: int = 0) -> ReilData:
        '''
        Returns an `action` based on the given `state`.

        ### Arguments
        state: the state for which the action should be returned.

        actions: the set of possible actions to choose from. If not supplied,
        `default_actions` is used.

        epoch: the epoch in which the agent is acting.
        '''
        possible_actions = functions.get_argument(
            actions, self._default_actions)

        result = self.best_actions(state, possible_actions)

        if len(result) > 1:
            action = self._break_tie(result, self._tie_breaker)  # type: ignore
        else:
            action = result[0]

        return action

    def best_actions(self,
                     state: ReilData,
                     actions: Tuple[ReilData, ...],
                     ) -> Tuple[ReilData, ...]:
        '''
        Returns a tuple of best `action`s based on the given `state`.

        ### Arguments
        state: the state for which the action should be returned.

        actions: the set of possible actions to choose from.
        '''
        raise NotImplementedError

    @staticmethod
    def _break_tie(input_tuple: Tuple[T, ...],
                   method: Literal['first', 'last', 'random']) -> T:
        '''
        Chooses one item from the supplied list of options, based on the method.
        ### Arguments
        input_tuple: the set of options to choose from.

        method: method of choosing an item from `input_tuple`.
        '''
        if method == 'first':
            action = input_tuple[0]
        elif method == 'last':
            action = input_tuple[-1]
        else:
            action = random.choice(input_tuple)

        return action
