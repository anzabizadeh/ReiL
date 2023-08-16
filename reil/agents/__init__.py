# -*- coding: utf-8 -*-
'''
agents module for reinforcement learning
========================================

This module provides different agents in reinforcement learning context.

Classes
-------
BaseAgent
    the base class of all agent classes. It only takes actions and does not learn.

Agent
    the base class of all agent classes that learn from history.

AgentDemon:
    A class that allows manipulation of an `agent`'s behavior.

QLearningAgent
    the Q-learning agent that can accept any learner

DeepQLearningAgent
    the agent with a neural network as learner (derived from
    `QLearningAgent` class)

RandomAgent
    an agent that randomly chooses an action

TwoPhaseAgent
    an agent consisting of two agents. A switch is used to decide when to switch
    from the first agent to the second one.

UserAgent
    an agent that shows current state and asks for user's choice
    of action

Types
-----
TrainingData
    a type alias for training data, consisting of a tuple of
    `FeatureSet` for X matrix and a tuple of floats for Y vector.

'''
