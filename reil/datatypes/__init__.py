# -*- coding: utf-8 -*-
'''
datatypes module for reinforcement learning
===========================================

This module contains datatypes used in `reil`

Submodules
----------
buffers:
    A submodule that contains different types of `Buffers` used in `ReiL`
    package.

Classes
-------
Feature:
    The main datatype in `ReiL`. Any value, such as components of a `state` or
    an `action` are instances of `Feature`.

FeatureSet:
    The main datatype used to communicate `state`s and `action`s
    between objects in `reil`. Under the hood, `FeatureSet` is a dictionary that
    contains instances of `Feature`.

FeatureGenerator:
    A class that generates `Feature` instances. If provided with the necessary
    information, the generated `Feature` will have the normalized form to be
    used which saves computation time.

FeatureGeneratorSet:
    A set of `FeatureGenerator`s that can generate `FeatureSet` objects. `agent`
    and `subject` classes use `FeatureGeneratorSet` for the properties that are
    needed for `state` and `action` definitions.

FeatureSetDumper:
    A class to dump a `FeatureSet` to a file.

State:
    A datatype that is being used mostly by children of `Stateful` to include
    a `State`, e.g. a state. It allows defining different
    definitions for the component, and call the instance to calculate them.

SecondayComponent:
    A datatype that is being used mostly by children of `Subject` to include
    a `SecondayComponent`, e.g. a statistic or a reward. It allows defining
    different definitions for the component, and call the instance to calculate
    them.

ActionSet:
    A class of type `SecondayComponent[FeatureGeneratorType]`. It is used to
    define different possible actions for a `Subject`.

Reward:
    A class of type `SecondaryComponent[float]`. It is used to define different
    reward functions for a `Subject`.

Statistic:
    A class similar to `SecondaryComponent`, but with `append` and `aggregate`
    methods, that allow for collecting data and computing statistics.

Entity:
    A datatype to specify `agent` or `subject` information. Used in
    `InteractionProtocol`.

EntityRegister:
    A class to register and maintain a list of entities. Used by `Stateful` objects.

LookaheadPlan:
    A datatype to specify how lookahead should be performed. Used in
    `InteractionProtocol`.

InteractionProtocol:
    A datatype to specifies how an `agent` and a `subject`
    interact in an `environment`.

Observation:
    A datatype to record the observation of an `agent` and a `subject` in an
    `environment`.

History:
    A list of `Observation`s.

LookaheadData:
    A list of `History`s. Used in `InteractionProtocol`.

'''

from __future__ import annotations

import dataclasses
from typing import Literal

from . import buffers  # noqa: W601
from .components import (ActionSet, Reward, SecondayComponent,  # noqa: W601
                         State, Statistic)
from .entity_register import EntityRegister  # noqa: W601
from .feature import (Feature, FeatureGenerator,  # noqa: W601
                      FeatureGeneratorSet, FeatureGeneratorType, FeatureSet,
                      FeatureSetDumper)


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
