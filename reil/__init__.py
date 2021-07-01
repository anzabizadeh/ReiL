# -*- coding: utf-8 -*-
'''
A Reinforcement Learning Module for Python
==========================================

This module provides a framework for training and test of different
reinforcement learning methods.

Modules
-------
agents
    Entities that act on one or more subject via an environment and observing
    the reward. `Agents` can learn or just be actors.

subjects
    Entities with an internal state that get one or more
    `agents`' action via an `environment` and return new state and reward.

environments
    Classes that connect `agents` and `subjects`, and simulate their
    interactions.

learners
A set of learning techniques used as the learner of an `agent`.

stats
    Compute statistics.

reilbase
    Base class for all `reil` classes.

stateful
    Based class for all stateful objects.

datatypes
    All custom datatypes used in `reil`.

utils
    Classes and functions that are utilities used in `reil`.

legacy
    All classes that are no longer supported.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from ._version import get_versions
__version__ = get_versions()['version']  # type: ignore
del get_versions
