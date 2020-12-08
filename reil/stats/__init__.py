# -*- coding: utf-8 -*-
'''
stats module for reinforcement learning
========================================

This module provides classes that comupte statistics.

Classes
-------
    WarfarinStats: a statistic class for warfarin analysis.


'''
from .stats import Stats
from .custom.warfarin_stats import WarfarinStats
from .rl_functions import RLFunction, NormalizedSquareDistance, PercentInRange, Delta
