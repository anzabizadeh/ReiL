# -*- coding: utf-8 -*-
'''
InteractionProtocol class
=========================

A datatype that accepts initial value and feature generator, and generates
new values. This datatype uses `Entity` to specify an `agent` or a `subject`.
'''
import dataclasses
from typing import Optional, Tuple

from typing_extensions import Literal


@dataclasses.dataclass(frozen=True)
class Entity:
    '''
    The datatype to specify an `agent` or a `subject`.
    Used in `InteractionProtocol`.
    '''
    name: str
    statistic_name: Optional[str] = None
    groupby: Optional[Tuple[str, ...]] = None
    aggregators: Optional[Tuple[str, ...]] = None


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
    reward_function_name: str
    n: int
    unit: Literal['interaction', 'instance', 'epoch']
