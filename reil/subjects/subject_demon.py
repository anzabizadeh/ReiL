# -*- coding: utf-8 -*-
'''
SubjectDemon class
==================

`SubjectDemon` class changes the behavior of a given subject.
'''
from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from reil.datatypes.components import Reward, Statistic
from reil.datatypes.feature import FeatureGeneratorType, FeatureSet
from reil.reilbase import ReilBase
from reil.subjects.subject import Subject

T = TypeVar('T', FeatureSet, FeatureGeneratorType)


@dataclasses.dataclass
class Modifier(Generic[T]):
    name: str
    cond_state_def: str | None
    condition_fn: Callable[[FeatureSet], bool] | None
    modifier_fn: Callable[[T], T]

    def __post_init__(self):
        if self.condition_fn is not None:
            if self.cond_state_def is None:
                raise ValueError(
                    '`condition_fn` cannot be declared with '
                    '`cond_state_def=None`.')


class SubjectDemon(ReilBase):
    '''
    This class accepts a regular subject, and intervenes in its interaction
    with the agents. It can modify `state` representation or change
    the `possible_actions`.
    '''

    def __init__(
            self,
            subject: Subject | None = None,
            action_modifier: Modifier[FeatureGeneratorType] | None = None,
            state_modifier: Modifier[FeatureSet] | None = None,
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

        self._subject: Subject
        self.reward: Reward
        self.statistic: Statistic

        if subject:
            self.__call__(subject)

        self._action_modifier = action_modifier
        self._state_modifier = state_modifier

    # @classmethod
    # def _empty_instance(cls):
    #     return cls(Subject())

    def __call__(self, subject: Subject) -> SubjectDemon:
        self._subject = subject
        self.reward = subject.reward
        self.statistic = subject.statistic
        self.is_terminated = subject.is_terminated
        self.take_effect = subject.take_effect
        self.reset = subject.reset

        return self

    def state(
            self, name: str,
            _id: int | None = None) -> FeatureSet:
        '''
        Generate the component based on the specified `name` for the
        specified caller.

        Arguments
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
        modifier = self._state_modifier
        if (modifier is not None and
            (modifier.condition_fn is None
                or modifier.condition_fn(self._subject.state(
                    modifier.cond_state_def, _id)))):  # type: ignore
            return modifier.modifier_fn(original_state)

        return original_state

    def possible_actions(
            self, name: str,
            _id: int | None = None
    ) -> FeatureGeneratorType | None:
        '''
        Generate the component based on the specified `name` for the
        specified caller.

        Arguments
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
        original_gen = self._subject.possible_actions(name, _id)
        modifier = self._action_modifier
        if (original_gen is not None and
            modifier is not None and
            (modifier.condition_fn is None
                or modifier.condition_fn(self._subject.state(
                    modifier.cond_state_def, _id)))):  # type: ignore
            return modifier.modifier_fn(original_gen)

        return original_gen

    # def load(
    #         self, filename: str,
    #         path: Union[str, pathlib.PurePath] | None) -> None:
    #     super().load(filename, path)

    # def save(
    #         self,
    #         filename: str | None = None,
    #         path: Union[str, pathlib.PurePath] | None = None
    # ) -> pathlib.PurePath:
    #     return super().save(filename, path)

    @property
    def _entity_list(self):
        return self._subject._entity_list

    @_entity_list.setter
    def _entity_list(self, arg):
        self._subject._entity_list = arg

    def register(self, entity_name: str, _id: int | None = None) -> int:
        return self._subject.register(entity_name, _id)

    def deregister(self, entity_id: int) -> None:
        return self._subject.deregister(entity_id)

    def copy(
        self, perturb: bool = False, n: int | None = None
    ) -> 'SubjectDemon' | list['SubjectDemon']:
        '''
        Returns a copy of the subject.

        Arguments
        ---------
        perturb:
            `False` should return an exact copy. `True` should be a subject
            with the same current `state`, but other attributes might change
            depending on the implementation.
        '''
        if perturb:
            raise NotImplementedError

        if n is None:
            return copy.deepcopy(self)

        return [copy.deepcopy(self) for _ in range(n)]
