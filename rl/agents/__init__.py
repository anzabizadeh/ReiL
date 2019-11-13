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
    FixedAgent: an agent that outputs a constant action
    QAgent: the Q-learning agent
    TD0Agent: a TD(0) agent (might not work! Uses legacy ValueSet instead of RLData)
    DQNAgent: the agent with neural network as a Q-function
        estimator
    WarfarinAgent: an agent based on Ravvaz et al (2016) paper for Warfarin Dosing
    WarfarinClusterAgent: an agent whose actions are based on clustering of observations

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from .agent import Agent
from .user_agent import UserAgent
from .random_agent import RandomAgent
from .fixed_agent import FixedAgent
from .q_learning import QAgent
from .td_agent import TD0Agent
from .dqn_agent import DQNAgent
from .warfarin_agent import WarfarinAgent
from .warfarin_cluster_based_agent import WarfarinClusterAgent