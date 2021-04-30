# -*- coding: utf-8 -*-
'''
AgentDemon class
================

`AgentDemon` class changes the behavior of a given agent.
'''

import pathlib
from reil.datatypes.feature import FeatureArray
from typing import Any, Callable, Generator, Optional, Tuple, Union

from reil import reilbase, agents, stateful


class AgentDemon(reilbase.ReilBase):
    '''
    This class accepts a regular `agent`, and intervenes in its interaction
    with the subjects. A substitute `agent` acts whenever a condition is
    satisfied.
    '''

    def __init__(
            self,
            main_agent: agents.Agent,
            sub_agent: agents.NoLearnAgent,
            condition_fn: Callable[[FeatureArray, int], bool],
            **kwargs: Any):
        '''
        Arguments
        ---------
        main_agent:
            The `agent` that needs to be intervened with.

        sub_agent:
            An `agent` that acts instead of the `main_agent`.

        condition_fn:
            A function that accepts the current state and ID of the subject
            and decides whether the `main_agent` should act or the `sub_agent`.

        '''
        super().__init__(**kwargs)

        self._main_agent = main_agent
        self._sub_agent = sub_agent
        self._condition_fn = condition_fn

        self.state = main_agent.state
        self.statistic = main_agent.statistic
        self.load = main_agent.load
        self.save = main_agent.save

    @classmethod
    def _empty_instance(cls):
        return cls(None, None, None)  # type: ignore

    def load_daemon(self, filename: str,
                    path: Optional[Union[str, pathlib.PurePath]]) -> None:
        super().load(filename, path)

    def save_daemon(self,
                    filename: Optional[str] = None,
                    path: Optional[Union[str, pathlib.PurePath]] = None,
                    data_to_save: Optional[Tuple[str, ...]] = None
                    ) -> Tuple[pathlib.PurePath, str]:
        return super().save(filename, path, data_to_save)

    def register(self, entity_name: str, _id: Optional[int] = None) -> int:
        from_main = self._main_agent.register(entity_name, _id)
        from_sub = self._sub_agent.register(entity_name, _id)
        if from_main != from_sub:
            raise RuntimeError(f'ID from the main agent {from_main} does not '
                               f'match the ID from the sub agent {from_sub}.')

        return self._main_agent.register(entity_name, _id)

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

    def observe(self, subject_id: int, stat_name: Optional[str],  # noqa: C901
                ) -> Generator[Union[FeatureArray, None], Any, None]:
        '''
        Create a generator to interact with the subject (`subject_id`).
        Extends `NoLearnAgent.observe`.

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
        if subject_id not in self._main_agent._entity_list:
            raise ValueError(f'Subject with ID={subject_id} not found.')

        trigger = self._main_agent._training_trigger
        trigger_on_state = trigger == 'state'
        trigger_on_action = trigger == 'action'
        trigger_on_reward = trigger == 'reward'
        trigger_on_termination = trigger == 'termination'

        history: stateful.History = []
        new_observation = stateful.Observation()
        while True:
            try:
                new_observation = stateful.Observation()
                temp = yield
                new_observation.state = temp['state']
                actions: Tuple[FeatureArray, ...] = temp['actions']
                epoch: int = temp['epoch']

                if trigger_on_state:
                    self._main_agent.learn([history[-1], new_observation])

                if actions is not None:
                    new_observation.action = self.act(
                        state=new_observation.state,  # type: ignore
                        subject_id=subject_id,
                        actions=actions, epoch=epoch)

                    if trigger_on_action:
                        self._main_agent.learn([history[-1], new_observation])

                    new_observation.reward = (yield new_observation.action)

                    history.append(new_observation)

                    if trigger_on_reward:
                        self._main_agent.learn(history[-2:])
                else:
                    yield

            except GeneratorExit:
                if new_observation.reward is None:  # terminated early!
                    history.append(new_observation)

                if trigger_on_termination:
                    self._main_agent.learn(history)

                if stat_name is not None:
                    self.statistic.append(stat_name, subject_id)

                return
