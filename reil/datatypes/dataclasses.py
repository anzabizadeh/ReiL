# -*- coding: utf-8 -*-
'''
InteractionProtocol class
=========================

A datatype that accepts initial value and feature generator, and generates
new values. This datatype uses `Entity` to specify an `agent` or a `subject`.
'''

from __future__ import annotations

import dataclasses
from typing import Literal

from reil.datatypes.feature import FeatureGeneratorType, FeatureSet


@dataclasses.dataclass(frozen=True)
class Entity:
    '''
    The datatype to specify an `agent` or a `subject`.
    Used in `InteractionProtocol`.
    '''
    name: str
    demon_name: str | None = None
    statistic_name: str | None = None
    groupby: tuple[str, ...] | None = None
    aggregators: tuple[str, ...] | None = None
    trajectory_name: str | None = None

    def __post_init__(self):
        if self.groupby is not None:
            self.__dict__['groupby'] = tuple(self.groupby)
        if self.aggregators is not None:
            self.__dict__['aggregators'] = tuple(self.aggregators)


@dataclasses.dataclass(frozen=True)
class LookaheadPlan:
    '''
    The datatype to specify lookahead plan.
    Used in `InteractionProtocol`.
    '''
    reward_name: str
    action_type: Literal['fixed', 'training', 'optimal'] = 'fixed'
    steps: int = 1
    subject_count: int = 1
    perturb_subject: bool = False
    # Removed 'branch_on_each_node', because it makes the implementation
    # too complicated
    # branch_on_each_node: bool = False


@dataclasses.dataclass
class InteractionProtocol:
    '''
    The datatype to specify how an `agent` should interact with a `subject` in
    an `environment`.
    '''
    agent: Entity
    subject: Entity
    state_name: str
    action_name: str
    reward_name: str
    n: int
    unit: Literal['interaction', 'instance', 'iteration']
    lookahead: LookaheadPlan | None = None


@dataclasses.dataclass
class Observation:
    # the state received from the subject
    state: FeatureSet | None = None
    # the action generator available to the agent
    possible_actions: FeatureGeneratorType | None = None
    # the action chosen by the agent
    action: FeatureSet | None = None
    # the action taken by the subject
    action_taken: FeatureSet | None = None
    # lookahead data
    lookahead: 'LookaheadData' | None = None
    # the reward of taking the action at the state
    reward: float | None = None


History = list[Observation]
LookaheadData = list[History]
