# -*- coding: utf-8 -*-
'''
stats module for reinforcement learning
========================================

This module provides classes that comupte statistics.

Classes
-------
WarfarinStats:
    A statistic class for warfarin analysis.


'''

from .stats import Stats  # noqa: W0611
from .custom.warfarin_stats import WarfarinStats  # noqa: W0611

from .reil_functions import (Arguments, Delta,  # noqa: W0611
                           NormalizedSquareDistance, PercentInRange,
                           ReilFunction)
