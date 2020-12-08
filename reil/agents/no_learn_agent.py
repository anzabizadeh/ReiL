# -*- coding: utf-8 -*-
'''
NoLearnAgent class
==================

The base class of all `agent` classes.
'''

import random
from typing import Any, Optional, Tuple, TypeVar

from reil import stateful
from reil.datatypes.reildata import ReilData
from reil.utils import functions
from typing_extensions import Literal

TrainingData = Tuple[Tuple[ReilData, ...], Tuple[float, ...]]
T = TypeVar('T')

class NoLearnAgent(stateful.Stateful):
    '''
    The base class of all `agent` classes. This class does not support any
    `learner`.
    '''

    def __init__(self,
                 default_actions: Tuple[ReilData, ...] = (),
                 tie_breaker: Literal['first', 'last', 'random'] = 'random',
                 **kwargs: Any):
        '''
        Arguments
        ---------
        default_actions:
            A list of default actions.

        tie_breaker:
            How to choose the `action` if more than one is candidate
            to be chosen.

        Raises
        ------
        ValueError:
            if `tie_breaker` is not one of 'first', 'last', and 'random'.
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
        Return an action based on the given state.

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
        Find the best `action`s for the given `state`.

        Arguments
        ---------
        state:
            The state for which the action should be returned.

        actions:
            The set of possible actions to choose from.

        Returns
        -------
        :
            A list of best actions.
        '''
        raise NotImplementedError

    @staticmethod
    def _break_tie(input_tuple: Tuple[T, ...],
                   method: Literal['first', 'last', 'random']) -> T:
        '''
        Choose one item from the supplied list of options, based on the method.

        Arguments
        ---------
        input_tuple:
            The set of options to choose from.

        method:
            Method of choosing an item from `input_tuple`.

        Returns
        -------
        :
            One of the items from the list
        '''
        if method == 'first':
            action = input_tuple[0]
        elif method == 'last':
            action = input_tuple[-1]
        else:
            action = random.choice(input_tuple)

        return action
