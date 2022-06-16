# -*- coding: utf-8 -*-
'''
agents module for reinforcement learning
========================================

This module provides different agents in reinforcement learning context.

Classes
-------
AgentBase
    the base class of all agent classes

Agent
    the base class of all agent classes that learn from history

AgentDemon:
    A class that allows an `agent` to be manipulated.

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

Types
-----
TrainingData
    a type alias for training data, consisting of a tuple of
    `FeatureSet` for X matrix and a tuple of floats for Y vector.

'''
