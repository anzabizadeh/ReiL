# -*- coding: utf-8 -*-
'''
subject class
=============

This `subject` class is the base class of all subject classes.
'''

import pathlib
from typing import Any, Optional, Tuple, Union

from reil import stateful
from reil.datatypes import Feature, FeatureArray, SecondayComponent


class Subject(stateful.Stateful):
    '''
    The base class of all subject classes.
    '''

    def __init__(self,
                 sequential_interaction: bool = True,
                 **kwargs: Any):
        '''
        Arguments
        ---------
        sequential_interaction:
            If `True`, `agents` can only act on the `subject` in the order they
            are added.

        Notes
        -----
        `sequential_interaction` is not enforced (implemented) yet!
        '''
        super().__init__(**kwargs)

        self._sequential_interaction = sequential_interaction
        self.reward = SecondayComponent(
            name='reward',
            primary_component=self.state,
            default_definition=self._default_reward_definition,
            enabled=False)

        self.possible_actions = SecondayComponent(
            name='action',
            primary_component=self.state,
            default_definition=self._default_action_definition,
            enabled=True)

    def _default_reward_definition(
            self, _id: Optional[int] = None) -> float:
        return 0.0

    def _default_action_definition(
            self, _id: Optional[int] = None) -> Tuple[FeatureArray, ...]:
        return (FeatureArray(Feature(name='default_action')),)

    def is_terminated(self, _id: Optional[int] = None) -> bool:
        '''
        Determine if the `subject` is terminated for the given `agent` ID.

        Arguments
        ---------
        _id:
            ID of the agent that checks termination. In a multi-agent setting,
            e.g. an RTS game, one agent might die and another agent might still
            be alive.

        Returns
        -------
        :
            `False` as long as the subject can accept new actions from the
            `agent`. If `_id` is `None`, then returns `True` if no `agent`
            can act on the `subject`.
        '''
        raise NotImplementedError

    # def possible_actions(self, _id: int = 0) -> Tuple[FeatureArray, ...]:
    #     '''
    #     Generate the list of possible actions.

    #     Arguments
    #     ---------
    #     _id:
    #         ID of the `agent` that wants to act on the `subject`.

    #     Returns
    #     -------
    #     :
    #         A list of possible actions for the `agent` with ID=_id.
    #     '''
    #     return (FeatureArray(Feature(name='default_action')),)

    def take_effect(self, action: FeatureArray, _id: int = 0) -> None:
        '''
        Receive an `action` from `agent` with ID=`_id` and transition to
        the next state.

        Arguments
        ---------
        action:
            The action sent by the `agent` that will affect this `subject`.

        _id:
            ID of the `agent` that has sent the `action`.
        '''
        self.reward.enable()

    def reset(self) -> None:
        '''Reset the `subject`, so that it can resume accepting actions.'''
        self.reward.disable()

    def load(self, filename: str,
             path: Optional[Union[str, pathlib.PurePath]]) -> None:

        super().load(filename, path=path)

        self.reward.set_primary_component(self.state)
        self.reward.set_default_definition(
            self._default_reward_definition)

    def save(self,
             filename: Optional[str] = None,
             path: Optional[Union[str, pathlib.PurePath]] = None,
             data_to_save: Optional[Tuple[str, ...]] = None
             ) -> Tuple[pathlib.PurePath, str]:

        prim_comp, self.reward._primary_component = (  # type: ignore
            self.reward._primary_component, None)
        reward_default, self.reward._default = (
            self.reward._default, None)
        try:
            f, p = super().save(filename, path=path, data_to_save=data_to_save)
        finally:
            self.reward._primary_component = prim_comp
            self.reward._default = reward_default

        return f, p
