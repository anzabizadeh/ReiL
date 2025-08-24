# -*- coding: utf-8 -*-
'''
AgentDemon class
================

`AgentDemon` class changes the behavior of a given agent.
'''
from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any, Literal, Union

from reil import reilbase
from reil.agents.base_agent import BaseAgent
from reil.datatypes import History
from reil.datatypes.components import State, Statistic
from reil.datatypes.entity_register import EntityRegister
from reil.datatypes.feature import FeatureGeneratorType, FeatureSet


class AgentDemon(BaseAgent):
    '''
    This class accepts a regular `agent`, and intervenes in its interaction
    with the subjects. A substitute `agent` acts whenever a condition is
    satisfied.
    '''

    def __init__(
            self,
            sub_agent: BaseAgent,
            condition_fn: Callable[[FeatureSet, int], bool],
            main_agent: BaseAgent | None = None,
            **kwargs: Any):
        '''
        Arguments
        ---------
        sub_agent:
            An `agent` that acts instead of the `main_agent`.

        condition_fn:
            A function that accepts the current state and ID of the subject
            and decides whether the `main_agent` should act or the `sub_agent`.

        main_agent:
            The `agent` that needs to be intervened with.
        '''
        reilbase.ReilBase.__init__(self, **kwargs)

        self.state: State
        self.statistic: Statistic
        self._entity_list: EntityRegister
        self._training_trigger: Literal[
            'none', 'termination', 'state', 'action', 'reward']

        self._main_agent: BaseAgent | None = main_agent
        self._sub_agent: BaseAgent = sub_agent
        self._condition_fn = condition_fn

        if main_agent is not None:
            self.__call__(main_agent)

    @classmethod
    def _empty_instance(cls):
        '''
        Return an empty instance of the class.
        '''
        return cls(BaseAgent(), lambda f, i: True, None)

    def __call__(self, main_agent: BaseAgent) -> AgentDemon:
        '''
        Set the `main_agent` to be intervened with.

        Arguments
        ---------
        main_agent:
            The `agent` that needs to be intervened with.

        Returns
        -------
        :
            self
        '''
        self._main_agent = main_agent
        self.state = main_agent.state
        self.statistic = main_agent.statistic
        self._entity_list = main_agent._entity_list
        self._training_trigger = main_agent._training_trigger

        return self

    def load(
            self, filename: str,
            path: Union[str, pathlib.PurePath] | None = None) -> None:
        _path = pathlib.Path(path or self._path)
        '''
        Load the agent from a file.

        Arguments
        ---------
        filename:
            The name of the file to load the agent from.

        path:
            The path to the file to load the agent from.
        '''
        super().load(filename, _path)

        self._main_agent = (self._main_agent or BaseAgent).from_pickle(
            filename, _path / 'main_agent')
        self._sub_agent = self._sub_agent.from_pickle(
            filename, _path / 'sub_agent')

        self.__call__(self._main_agent)

    def save(
            self,
            filename: str | None = None,
            path: Union[str, pathlib.PurePath] | None = None
    ) -> pathlib.PurePath:
        '''
        Save the agent to a file.

        Arguments
        ---------
        filename:
            The name of the file to save the agent to.

        path:
            The path to the file to save the agent to.

        Returns
        -------
        :
            The path to the file where the agent was saved
        '''
        full_path = super().save(filename, path)
        if self._main_agent:
            self._main_agent.save(
                full_path.name, full_path.parent / 'main_agent')
        self._sub_agent.save(
            full_path.name, full_path.parent / 'sub_agent')

        return full_path

    def register(self, entity_name: str, _id: int | None = None) -> int:
        '''
        Register a new entity.

        Arguments
        ---------
        entity_name:
            The name of the entity to register.

        _id:
            The ID to register the entity with.

        Returns
        -------
        :
            The ID of the entity.

        Raises
        ------
        ValueError
            If the main agent is not set.

        RuntimeError
            If the ID from the main agent does not match the ID from the sub
            agent.
        '''
        if self._main_agent is None:
            raise ValueError('main_agent is not set.')

        from_main = self._main_agent.register(entity_name, _id)
        from_sub = self._sub_agent.register(entity_name, _id)
        if from_main != from_sub:
            raise RuntimeError(f'ID from the main agent {from_main} does not '
                               f'match the ID from the sub agent {from_sub}.')

        return from_main

    def deregister(self, entity_id: int) -> None:
        '''
        Deregister an entity.

        Arguments
        ---------
        entity_id:
            The ID of the entity to deregister.

        Raises
        ------
        ValueError
            If the main agent is not set.
        '''
        self._sub_agent.deregister(entity_id)

        if self._main_agent is None:
            raise ValueError('main_agent is not set.')
        self._main_agent.deregister(entity_id)

    def reset(self):
        '''
        Reset the agent.
        '''
        if self._main_agent:
            self._main_agent.reset()
        self._sub_agent.reset()

    def act(self,
            state: FeatureSet,
            subject_id: int,
            actions: FeatureGeneratorType,
            iteration: int = 0) -> FeatureSet:
        '''
        Return an action based on the given state.

        Arguments
        ---------
        state:
            the state for which the action should be returned.

        subject_id:
            the ID of the `subject` on which action should occur.

        actions:
            the set of possible actions to choose from.

        iteration:
            the iteration in which the agent is acting.

        Raises
        ------
        ValueError
            Subject with `subject_id` not found.

        Returns
        -------
        :
            the action
        '''
        if self._condition_fn(state, subject_id):
            return self._sub_agent.act(state, subject_id, actions, iteration)

        if self._main_agent is None:
            raise ValueError('main_agent is not set.')

        return self._main_agent.act(state, subject_id, actions, iteration)

    def learn(self, history: History) -> None:
        '''
        Learn using history.

        Arguments
        ---------
        subject_id:
            the ID of the `subject` whose history is being used for learning.

        next_state:
            The new `state` of the `subject` after taking `agent`'s action.
            Some methods
        '''
        if self._main_agent is None:
            raise ValueError('main_agent is not set.')

        if self._training_trigger != 'none':
            self._main_agent.learn(history)  # type: ignore

    def __getstate__(self):
        '''
        Get the state of the agent.

        Returns
        -------
        :
            The state of the agent.
        '''
        state = super().__getstate__()

        state['_main_agent'] = type(self._main_agent)
        state['_sub_agent'] = type(self._sub_agent)
        del state['state']
        del state['statistic']

        return state
