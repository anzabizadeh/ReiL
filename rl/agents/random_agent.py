# -*- coding: utf-8 -*-
"""
The :mod:`agents` provides the super class and some classes for reinforcement learning.

Classes
-------
    - `Agent`: the super class of all agent classes
    - `UserAgent`: an agent that shows asks for user's choice of action
    - `RandomAgent`: an agent that randomly chooses an action
    - `RLAgent`: the Q-learning agent
    - `NeuralAgent`: the agent with neural network as a  (Not Implemented Yet)

@author: Sadjad Anzabi Zadeh
-----------------------------------

Classes:
    Agent (Super Class)
"""

from random import choice

from rl.agents import Agent


def main():
    pass


class RandomAgent(Agent):
    '''
    This agent acts randomly.
    ''' 
    def act(self, state, **kwargs):
        '''
        States and a set of possible actions are given to the function and it chooses an action randomly.
        \nArguments:
        \n    state: the state for which the action should be returned. This argument is solely to keep the method's signature unified.
        \n    actions: the set of possible actions to choose from. If not provided, an empty list is returned. 
        ''' 
        try:  # possible actions
            return choice(kwargs['actions'])
        except KeyError:
            return []


if __name__ == '__main__':
    main()
