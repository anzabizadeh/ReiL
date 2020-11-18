# -*- coding: utf-8 -*-
'''
learners module for reinforcement learning
==========================================

This module provides different learners in reinforcement learning context.

Classes
-------
    Learner: the super class of all learner classes
    Dense: a fully-connected neural net
    LookupTable: a simple lookup table

    LearningRateScheduler: base class for learning rate schedulers
    ConstantLearningRate: a class that returns a constant learning rate

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from reil.learners.learning_rate_schedulers import (ConstantLearningRate,
                                                    LearningRateScheduler)

from reil.learners.learner import Learner
from reil.learners.fully_connected_neural_net import Dense
from reil.learners.lookup_table import LookupTable
