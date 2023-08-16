# -*- coding: utf-8 -*-
'''
TwoPhaseAgent class
===================

A class that allows combining two `agent`s.
'''

from typing import Any

from reil.agents.base_agent import BaseAgent
from reil.datatypes.feature import FeatureGeneratorType, FeatureSet


class TwoPhaseAgent(BaseAgent):
    def __init__(
        self, first_agent: BaseAgent, second_agent: BaseAgent,
        switch_feature: str, switch_value: Any,
        init_state_comps: tuple[str, ...], main_state_comps: tuple[str, ...],
        **kwargs: Any
    ):
        '''
        Arguments
        ---------
        first_agent:
            The first agent to be used.
        second_agent:
            The second agent to be used.
        switch_feature:
            The name of the feature that will be used to switch between the
            two agents.
        switch_value:
            The value of the switch feature that will trigger the switch between
            agents.
        init_state_comps:
            The components of the state that will be used by the first agent.
        main_state_comps:
            The components of the state that will be used by the second agent.
        **kwargs:
            Additional keyword arguments.
        '''
        super().__init__(**kwargs)
        self._first_agent = first_agent
        self._second_agent = second_agent
        self._switch_feature = switch_feature
        self._switch_value = switch_value
        self._init_state_comps = init_state_comps
        self._main_state_comps = main_state_comps

    def act(
            self, state: FeatureSet, subject_id: int,
            actions: FeatureGeneratorType, iteration: int = 0) -> FeatureSet:
        '''
        Arguments
        ---------
        state:
            The current state.
        subject_id:
            The id of the subject to be acted upon.
        actions:
            The actions to be performed.
        iteration:
            The current iteration.

        Returns
        -------
        :
            The actions to be performed.
        '''
        val = state.value
        if val[self._switch_feature] < self._switch_value:  # type: ignore
            for f in set(val.keys()).difference(self._init_state_comps):
                state.pop(f)

            action = self._first_agent.act(
                state, subject_id, actions, iteration)
            return action

        for f in set(val.keys()).difference(self._main_state_comps):
            state.pop(f)

        return self._second_agent.act(state, subject_id, actions, iteration)
