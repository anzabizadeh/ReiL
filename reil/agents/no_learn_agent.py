# -*- coding: utf-8 -*-
'''
NoLearnAgent class
==================

The base class of all `agent` classes.
'''

import random
from typing import Any, Generator, Optional, Tuple, TypeVar, Union

from reil import stateful
from reil.datatypes import FeatureArray
from typing_extensions import Literal

TrainingData = Tuple[Tuple[FeatureArray, ...], Tuple[float, ...]]
T = TypeVar('T')


class NoLearnAgent(stateful.Stateful):
    '''
    The base class of all `agent` classes. This class does not support any
    `learner`.
    '''

    def __init__(self,
                 default_actions: Tuple[FeatureArray, ...] = (),
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
            `tie_breaker` is not one of 'first', 'last', and 'random'.
        '''
        super().__init__(**kwargs)

        self._default_actions = default_actions
        self._training_trigger = 'none'

        if tie_breaker not in ['first', 'last', 'random']:
            raise ValueError(
                'Tie breaker should be one of first, last, or random options.')
        self._tie_breaker: Literal['first', 'last', 'random'] = tie_breaker

    def act(self,
            state: FeatureArray,
            subject_id: int,
            actions: Optional[Tuple[FeatureArray, ...]] = None,
            epoch: int = 0) -> FeatureArray:
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
        possible_actions = actions or self._default_actions

        result = self.best_actions(state, possible_actions)

        if len(result) > 1:
            action = self._break_tie(result, self._tie_breaker)
        else:
            action = result[0]

        return action

    def best_actions(self,
                     state: FeatureArray,
                     actions: Tuple[FeatureArray, ...],
                     ) -> Tuple[FeatureArray, ...]:
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

    def observe(self, subject_id: int, stat_name: str,
                ) -> Generator[Union[FeatureArray, None], Any, None]:
        '''
        Create a generator to interact with the subject (`subject_id`).

        This method creates a generator for `subject_id` that
        receives `state`, yields `action` and receives `reward`
        until it is closed. When `.close()` is called on the generator,
        `statistics` are calculated.

        Arguments
        ---------
        subject_id:
            the ID of the `subject` on which action happened.

        stat_name:
            The name of the `statistic` that should be computed at the end of
            each trajectory.

        Raises
        ------
        ValueError
            Subject with `subject_id` not found.
        '''
        if subject_id not in self._entity_list:
            raise ValueError(f'Subject with ID={subject_id} not found.')

        history: stateful.History = []
        new_observation = stateful.Observation()
        while True:
            try:
                new_observation = stateful.Observation()
                temp = yield
                new_observation.state = temp['state']
                actions: Tuple[FeatureArray, ...] = temp['actions']
                epoch: int = temp['epoch']

                if actions is not None:
                    new_observation.action = self.act(
                        state=new_observation.state,  # type: ignore
                        subject_id=subject_id,
                        actions=actions, epoch=epoch)

                    new_observation.reward = (yield new_observation.action)

                    history.append(new_observation)

            except GeneratorExit:
                if new_observation.reward is None:  # terminated early!
                    history.append(new_observation)

                # self._computed_stats.append(
                #     self.statistic(stat_name, subject_id))

                return

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


        :meta public:
        '''
        if method == 'first':
            action = input_tuple[0]
        elif method == 'last':
            action = input_tuple[-1]
        else:
            action = random.choice(input_tuple)

        return action
