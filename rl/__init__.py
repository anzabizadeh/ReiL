# -*- coding: utf-8 -*-
'''
Reinforcement learning module for Python
========================================

This module provides different reinforcement learning methods.

submodules
----------
    subjects: objects with an internal state that get one or
        more agent's action via environment and return new
        state and reward.
    agents: objects that learn by acting on one or more subject
        via an environment and observing the reward.
    environments: objects that connect agents to subjects and
        elapse time.
    valueset: a data type to store state and action data.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''


__all__ = ['environments', 'agents', 'subjects', 'valueset', 'rldata', 'data_collector']
