# -*- coding: utf-8 -*-
'''
LRAgent class
=============

An agent that uses a given `scikitlearn` model, and parameter list, and
specifies action using that.
'''
from collections.abc import Callable
from typing import Any

from sklearn.linear_model import LinearRegression

from reil.agents.base_agent import BaseAgent
from reil.datatypes.feature import Feature, FeatureGeneratorType, FeatureSet


class LRAgent(BaseAgent):
    '''
    An agent that acts based on user input.
    '''

    def __init__(
            self,
            default_actions: tuple[FeatureSet, ...] = (),
            models: dict[int, LinearRegression] = {0: LinearRegression()},
            feature_sequence: tuple[str, ...] = (),
            value_extractor_fn: Callable[
                [Feature], float] = lambda x: x.value,  # type: ignore
            **kwargs: Any):
        super().__init__(default_actions=default_actions, **kwargs)
        self._models = models
        self._feature_sequence = feature_sequence
        self._value_extractor_fn = value_extractor_fn

    def act(self,
            state: FeatureSet,
            subject_id: int,
            actions: FeatureGeneratorType,
            iteration: int = 0) -> FeatureSet:
        '''
        Return a random action.

        Arguments
        ---------
        state:
            The state for which the action should be returned.

        actions:
            The set of possible actions to choose from.

        iteration:
            The iteration in which the agent is acting.

        Returns
        -------
        :
            The action
        '''
        values = {n: v for n, v in (
            (f.name, f.value) if f.is_numerical
            else (f'{f.name}_{f.value}', 1)
            for f in state)
        }

        X = [[
            values.get(feature_name, 0.0)
            for feature_name in self._feature_sequence]
        ]

        action_temp: float = self._models[
            subject_id].predict(X)  # type: ignore

        possible_actions = actions

        # find the closest allowed action
        actions_dict: dict[float, FeatureSet] = {
            abs(a.value - action_temp): a  # type: ignore
            for a in possible_actions}

        return actions_dict[min(actions_dict)]
