# -*- coding: utf-8 -*-
'''
datatypes module for reinforcement learning
===========================================

This module contains datatypes used in `reil`

Submodules
----------
buffers:
    A module that contains different types of `Buffers` used in `reil`
    package.

Classes
-------
Feature:
    A datatype that accepts initial value and feature generator and
    generates new values.

Entity:
    A datatype to specify `agent` or `subject` information. Used in
    `InteractionProtocol`.

InteractionProtocol:
    A datatype to specifies how an `agent` and a `subject`
    interact in an `environment`.

ReilData:
    The main datatype used to communicate `state`s, `action`s, and `reward`s,
    between objects in `reil`. `ReilData` is basically a tuple that contains
    instances of `BaseData`, `CategoricalData`, and `NumericalData`.

BaseData:
    An immutable dataclass that accepts name, value, and normalizer.

CategoricalData:
    Extends `BaseData` by adding a list of categories.

NumericalData:
    Extends `BaseData` by adding lower and upper bounds for the value.

PrimaryComponent:
    A datatype that is being used mostly by children of `Stateful` to include
    a `PrimaryComponent`, e.g. a state. It allows defining different
    definitions for the component, and call the instance to calculate them.

SecondayComponent:
    A datatype that is being used mostly by children of `Subject` to include
    a `SecondayComponent`, e.g. a statistic or a reward. It allows defining
    different definitions for the component, and call the instance to calculate
    them.
'''

from .reildata import (BaseData, CategoricalData, NumericalData,  # noqa: W0611
                       ReilData)

from .feature import Feature, FeatureType  # noqa: W0611
from .interaction_protocol import Entity, InteractionProtocol  # noqa: W0611

from .components import (PrimaryComponent, SecondayComponent,  # noqa: W0611
                         Statistic, MockStatistic, SubComponentInfo)

import reil.datatypes.buffers  # noqa: W0611
