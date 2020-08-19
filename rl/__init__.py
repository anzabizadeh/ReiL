# -*- coding: utf-8 -*-
'''
Reinforcement learning module for Python
========================================

This module provides different reinforcement learning methods.

submodules
----------
    agents: objects that learn by acting on one or more subject
        via an environment and observing the reward.
    subjects: objects with an internal state that get one or
        more agent's action via environment and return new
        state and reward.
    environments: objects that connect agents to subjects and
        elapse time.
    stats: objects that compute statistics.
    rlbase: base class for all rl objects.
    rldata: a data type to store states and actions.
    utils: all classes that are not part of the agent, subject, environment framework.
    legacy: all classes that are no longer supported.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''


__all__ = ['agents',
           'data_collector',
           'environments',
           'legacy',
           'rlbase',
           'rldata',
           'stats',
           'subjects',
           'utils']
