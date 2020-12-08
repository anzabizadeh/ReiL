# -*- coding: utf-8 -*-
'''
agents module for reinforcement learning
========================================

This module provides different agents in reinforcement learning context.

Classes
-------
NoLearnAgent
    the base class of all agent classes

Agent
    the base class of all agent classes that learn from history

QLearning
    the Q-learning agent that can accept any learner

DeepQLearning
    the agent with neural network as a learner (derived from
    `QLearning` class)

RandomAgent
    an agent that randomly chooses an action

UserAgent
    an agent that shows current state and asks for user's choice
    of action

WarfarinAgent
    an agent based on Ravvaz et al (2016) paper for Warfarin Dosing

WarfarinClusterAgent
    an agent whose actions are based on clustering of observations.
    Note: This agent will no longer run due to changes in the module
    implementation.

Types
-----
TrainingData
    a type alias for training data, consisting of a tuple of
    `ReilData` for X matrix and a tuple of floats for Y vector.

AgentType
    a type variable that is bound to `Agent`.
'''
from .no_learn_agent import NoLearnAgent, TrainingData
from .agent import Agent
from .q_learning import QLearning
from .deep_q_learning import DeepQLearning
from .random_agent import RandomAgent
from .user_agent import UserAgent
from .warfarin_agent import WarfarinAgent
# from .warfarin_cluster_based_agent import WarfarinClusterAgent

from typing import TypeVar

AgentType = TypeVar('AgentType', bound=Agent)
