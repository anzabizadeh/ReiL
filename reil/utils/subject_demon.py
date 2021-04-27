# -*- coding: utf-8 -*-
'''
subject class
=============

This `subject` class is the base class of all subject classes.
'''

import dataclasses
import pathlib
from typing import Any, Callable, Optional, Tuple, Union

from reil.datatypes import FeatureArray
from reil.reilbase import ReilBase
from reil.subjects import Subject


@dataclasses.dataclass
class Modifier:
    name: str
    state_definition: str
    condition: Callable[[FeatureArray], bool]
    function: Callable[[FeatureArray], Any]


class SubjectDemon(ReilBase):
    '''
    This class accepts a regular subject, and intervenes in its interaction
    with the agents. It can modify `state` representation or change
    the `possible_actions`.
    '''

    def __init__(
            self,
            subject: Subject,
            action_modifier: Optional[Modifier] = None,
            state_modifier: Optional[Modifier] = None,
            **kwargs: Any):
        '''
        Arguments
        ---------
        subject:
            The `subject` that needs to be intervened with.

        action_modifier:
            A modifier instance for action.

        state_modifier:
            A modifier instance for state.

        '''
        super().__init__(**kwargs)

        self._subject = subject
        self.reward = subject.reward
        self.statistic = subject.statistic
        self.is_terminated = subject.is_terminated
        self.take_effect = subject.take_effect
        self.reset = subject.reset
        self.load = subject.load
        self.save = subject.save

        self._action_modifier = action_modifier
        self._state_modifier = state_modifier

    def state(self, name: str, _id: Optional[int] = None) -> FeatureArray:
        '''
        Generate the component based on the specified `name` for the
        specified caller.

        Parameters
        ----------
        name:
            The name of the component definition.

        _id:
            ID of the caller.

        Returns
        -------
        :
            The component with the specified definition `name`.

        Raises
        ------
        ValueError
            Definition not found.
        '''
        original_state = self._subject.state(name, _id)
        if self._state_modifier is None:
            return original_state

        state = self._subject.state(self._state_modifier.state_definition, _id)
        if self._state_modifier.condition(state):
            return self._state_modifier.function(original_state)

        return original_state

    def possible_actions(self,
                         name: str,
                         _id: Optional[int] = None) -> Any:
        '''
        Generate the component based on the specified `name` for the
        specified caller.

        Parameters
        ----------
        name:
            The name of the component definition.

        _id:
            ID of the caller.

        Returns
        -------
        :
            The component with the specified definition `name`.

        Raises
        ------
        ValueError
            Definition not found.
        '''
        original_set = self._subject.possible_actions(name, _id)
        if self._action_modifier is None:
            return original_set

        state = self._subject.state(
            self._action_modifier.state_definition, _id)
        if self._action_modifier.condition(state):
            return self._action_modifier.function(original_set)

        return original_set

    def load_daemon(self, filename: str,
                    path: Optional[Union[str, pathlib.PurePath]]) -> None:
        super().load(filename, path)

    def save_daemon(self,
                    filename: Optional[str] = None,
                    path: Optional[Union[str, pathlib.PurePath]] = None,
                    data_to_save: Optional[Tuple[str, ...]] = None
                    ) -> Tuple[pathlib.PurePath, str]:
        return super().save(filename, path, data_to_save)
