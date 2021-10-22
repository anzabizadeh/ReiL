# -*- coding: utf-8 -*-
'''
LRAgent class
=============

An agent that uses a given `scikitlearn` model, and parameter list, and
specifies action using that.
'''
from typing import Any, Callable, Dict, Optional, Tuple

from reil.agents.agent_base import AgentBase
from reil.datatypes.feature import Feature, FeatureArray
from sklearn.linear_model import LinearRegression


class LRAgent(AgentBase):
    '''
    An agent that acts based on user input.
    '''

    def __init__(
            self,
            default_actions: Tuple[FeatureArray, ...] = (),
            models: Dict[int, LinearRegression] = {0: LinearRegression()},
            feature_sequence: Tuple[str, ...] = (),
            value_extractor_fn: Callable[
                [Feature], float] = lambda x: x.value,  # type: ignore
            **kwargs: Any):
        super().__init__(default_actions=default_actions, **kwargs)
        self._models = models
        self._feature_sequence = feature_sequence
        self._value_extractor_fn = value_extractor_fn

    def act(self,
            state: FeatureArray,
            subject_id: int,
            actions: Optional[Tuple[FeatureArray, ...]] = None,
            iteration: int = 0) -> FeatureArray:
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

        possible_actions = actions or self._default_actions

        # find the closest allowed action
        actions_dict: Dict[float, FeatureArray] = {
            abs(a.value - action_temp): a  # type: ignore
            for a in possible_actions}

        return actions_dict[min(actions_dict)]
