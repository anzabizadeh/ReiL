# -*- coding: utf-8 -*-
'''
agents module for reinforcement learning
========================================

This module provides different agents in reinforcement learning context.

Classes
-------
    Agent: the super class of all agent classes
    UserAgent: an agent that shows current state and asks for
        user's choice of action
    RandomAgent: an agent that randomly chooses an action
    QAgent: the Q-learning agent
    ANNAgent: the agent with neural network as a Q-function
        estimator (Not Implemented Yet)

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from .agent import Agent
from .q_learning import QAgent
from .random_agent import RandomAgent
from .user_agent import UserAgent
from .ann_agent import ANNAgent
from .td_agent import TD0Agent
