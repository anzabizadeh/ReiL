# -*- coding: utf-8 -*-
'''
AgentDemon class
================

`AgentDemon` class changes the behavior of a given agent.
'''
from __future__ import annotations

import pathlib
from typing import Any, Callable, Optional, Tuple, Union

from reil import reilbase, stateful
from reil.agents.agent import Agent
from reil.agents.no_learn_agent import NoLearnAgent
from reil.datatypes.components import PrimaryComponent, Statistic
from reil.datatypes.feature import FeatureArray
from reil.subjects.subject import Subject


class AgentDemon(Agent):
    '''
    This class accepts a regular `agent`, and intervenes in its interaction
    with the subjects. A substitute `agent` acts whenever a condition is
    satisfied.
    '''

    def __init__(
            self,
            sub_agent: NoLearnAgent,
            condition_fn: Callable[[FeatureArray, int], bool],
            main_agent: Optional[Agent] = None,
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

        self.state: PrimaryComponent
        self.statistic: Statistic
        self._entity_list: stateful.EntityRegister
        self._training_trigger: str

        self._main_agent = main_agent
        self._sub_agent: NoLearnAgent = sub_agent
        self._condition_fn = condition_fn

        if main_agent is not None:
            self.__call__(main_agent)

    @classmethod
    def _empty_instance(cls):
        return cls(Subject(), None, None)  # type: ignore

    def __call__(self, main_agent: Agent) -> AgentDemon:
        self._main_agent = main_agent
        self.state = main_agent.state
        self.statistic = main_agent.statistic
        self._entity_list = main_agent._entity_list
        self._training_trigger = main_agent._training_trigger

        return self

    def load(self, filename: str,
             path: Optional[Union[str, pathlib.PurePath]]) -> None:
        _path = pathlib.Path(path or self._path)
        super().load(filename, _path)

        self._main_agent = self._main_agent.from_pickle(  # type: ignore
            filename, _path / 'main_agent')
        self._sub_agent = self._sub_agent.from_pickle(  # type: ignore
            filename, _path / 'sub_agent')

        self.__call__(self._main_agent)

    def save(self,
             filename: Optional[str] = None,
             path: Optional[Union[str, pathlib.PurePath]] = None,
             data_to_save: Optional[Tuple[str, ...]] = None
             ) -> Tuple[pathlib.PurePath, str]:

        data = list(data_to_save or self.__dict__)
        save_main = '_main_agent' in data
        save_sub = '_sub_agent' in data
        if save_main:
            temp_main, self._main_agent = (  # type: ignore
                self._main_agent, type(self._main_agent))

        if save_sub:
            temp_sub, self._sub_agent = (  # type: ignore
                self._sub_agent, type(self._sub_agent))

        if 'state' in data:
            data.remove('state')

        if 'statistic' in data:
            data.remove('statistic')

        _path = pathlib.Path(path or self._path)
        _filename = filename or self._name

        try:
            super().save(
                _filename, _path, data_to_save=tuple(data))
            if save_main:
                self._main_agent = temp_main  # type: ignore
                self._main_agent.save(_filename, _path / 'main_agent')
            if save_sub:
                self._sub_agent = temp_sub  # type: ignore
                self._sub_agent.save(_filename, _path / 'sub_agent')
        finally:
            if save_main:
                self._main_agent = temp_main  # type: ignore
                self._sub_agent = temp_sub  # type: ignore

        return _path, _filename

    def register(self, entity_name: str, _id: Optional[int] = None) -> int:
        from_main = self._main_agent.register(entity_name, _id)
        from_sub = self._sub_agent.register(entity_name, _id)
        if from_main != from_sub:
            raise RuntimeError(f'ID from the main agent {from_main} does not '
                               f'match the ID from the sub agent {from_sub}.')

        return from_main

    def deregister(self, entity_id: int) -> None:
        self._sub_agent.deregister(entity_id)
        self._main_agent.deregister(entity_id)

    def reset(self):
        self._main_agent.reset()
        self._sub_agent.reset()

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
            the state for which the action should be returned.

        subject_id:
            the ID of the `subject` on which action should occur.

        actions:
            the set of possible actions to choose from.

        epoch:
            the epoch in which the agent is acting.

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
            return self._sub_agent.act(state, subject_id, actions, epoch)

        return self._main_agent.act(state, subject_id, actions, epoch)

    def learn(self, history: stateful.History) -> None:
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
        self._main_agent.learn(history)
