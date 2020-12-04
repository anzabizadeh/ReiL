# -*- coding: utf-8 -*-
'''
Reinforcement learning module for Python
========================================

This module provides a framework for training and test of different
reinforcement learning methods.

submodules
----------
agents: objects that learn by acting on one or more subject
    via an environment and observing the reward.

subjects: objects with an internal state that get one or
    more agent's action via environment and return new
    state and reward.

environments: objects that connect agents to subjects and
    elapse time.

learners: objects that are used as the learner of an `agent`.

stats: objects that compute statistics.

reilbase: base class for all `reil` objects.

stateful: based class for all stateful objects.

datatypes: all custom datatypes used in `reil`.

utils: all classes that are not part of the agent, subject, environment framework.

legacy: all classes that are no longer supported.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''


__all__ = ['agents',
           'datatypes',
           'environments',
           'learners',
           'legacy',
           'reilbase',
           'stateful',
           'stats',
           'subjects',
           'utils']
