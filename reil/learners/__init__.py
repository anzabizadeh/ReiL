# -*- coding: utf-8 -*-
'''
learners module for reinforcement learning
==========================================

This module provides different learners in reinforcement learning context.

Classes
-------
Learner:
    The base class of all `learner` classes

Dense:
    A fully-connected neural net

QLookupTable:
    A simple lookup table for Q-learning

LearningRateScheduler:
    Base class for learning rate schedulers

ConstantLearningRate:
    A class that returns a constant learning rate
'''

from .learning_rate_schedulers import (  # noqa: W0611
    LearningRateScheduler, ConstantLearningRate)

from .learner import Learner  # noqa: W0611
from .lookup_table import QLookupTable, TableEntry  # noqa: W0611
from .dense import Dense  # noqa: W0611
from .dense_softmax import DenseSoftMax  # noqa: W0611
