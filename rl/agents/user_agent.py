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

# KNOWN ISSUES:
# UserAgent should be revised.


from .agent import Agent


def main():
    pass


class UserAgent(Agent):
    '''
    This class is used to get the action from the user.
    '''
    def act(self, state, **kwargs):
        # NOTE TO MYSELF: This implementation is not good! It should get either the state or the printable.
        '''
        Displays current state and asks user for input. The result is a string.
        '''
        try:
            s = '\n'+kwargs['printable']+'\n'
        except KeyError:
            s = state.to_list()
        action = None
        while action is None:
            action = input('Choose action for this state: {}'.format(s))
        return action


if __name__ == '__main__':
    main()
