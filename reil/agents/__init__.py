# -*- coding: utf-8 -*-
'''
agents module for reinforcement learning
========================================

This module provides different agents in reinforcement learning context.

Classes
-------
    Agent: the base class of all agent classes
    DeepQLearning: the agent with neural network as a learner (derived from QLearning class)
    FixedAgent: an agent that outputs a constant action
    UserAgent: an agent that shows current state and asks for
        user's choice of action
    RandomAgent: an agent that randomly chooses an action
    QLearning: the Q-learning agent that can accept any learner
    TD0Agent: a TD(0) agent (might not work! Uses legacy ValueSet instead of RLData)
    WarfarinAgent: an agent based on Ravvaz et al (2016) paper for Warfarin Dosing
    WarfarinClusterAgent: an agent whose actions are based on clustering of observations

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''
# TODO: update FixedAgent
# TODO: update UserAgent
# TODO: update RandomAgent
# TODO: update TD0Agent
# TODO: check if WarfarinClusterAgent is still useful!

from .agent import Agent
from .q_learning import QLearning
from .deep_q_learning import DeepQLearning
from .random_agent import RandomAgent
from .td_agent import TD0Agent
from .user_agent import UserAgent
from .warfarin_agent import WarfarinAgent
from .warfarin_cluster_based_agent import WarfarinClusterAgent
