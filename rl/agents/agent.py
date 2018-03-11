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

import pickle


def main():
    pass

class Agent:
    '''
    Super class of all agent classes. This class provides basic methods including act, learn, reset, load, save, and report Load and save are implemented.
    '''
    def __init__(self, **kwargs):
        self._training_flag = True

    @property
    def status(self):
        '''
        Returns the status of the agent as 'training' or 'testing'.
        '''
        if self._training_flag:
            return 'training'
        else:
            return 'testing'

    @status.setter
    def status(self, value):
        '''
        Sets the status of the agent as 'training' or 'testing'.
        '''
        self._training_flag = (value == 'training')

    def act(self, state, **kwargs):
        '''
        This function gets the state and returns agent's action.
        If state is 'training' (_training_flag=false), then this function should not return any random move due to exploration.
        '''
        pass

    def learn(self, **kwargs):
        '''
        Shoud get state, action, reward or history to learn.
        ''' 
        pass

    def reset(self):
        '''
        Resets the agent at the end of a learning episode.
        ''' 
        pass

    def load(self, **kwargs):
        '''
        Loads an agent.
        \nArguments:
        \n    filename: the name of the file to be loaded.
        '''
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')
        with open(filename + '.pkl', 'rb') as f:
            self.__dict__ = pickle.load(f)

    def save(self, **kwargs):
        '''
        Saves an agent.
        \nArguments:
        \n    filename: the name of the file to be saved.
        '''
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')
        with open(filename + '.pkl', 'wb+') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def report(self, **kwargs):
        '''
        Should generate a report and return the string.
        ''' 
        raise NotImplementedError


if __name__ == '__main__':
    main()
