# -*- coding: utf-8 -*-
'''
Agent class
=============

This `agent` class is the super class of all agent classes. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import pickle


def main():
    pass

class Agent:
    '''
    Super class of all agent classes.
    
    Attributes
    ----------
        status: return the status of the agent

    Methods
    -------
        act: return an action based on the given state.
        learn: learn using either history or action, reward, and state.
        reset: reset the agent.
        load: load an agent from a file.
        save: save the agent to a file.
        report: return a report as a string.
    '''
    def __init__(self, **kwargs):
        self._training_flag = True

    @property
    def status(self):
        '''Return the status of the agent as 'training' or 'testing'.'''
        if self._training_flag:
            return 'training'
        else:
            return 'testing'

    @status.setter
    def status(self, value):
        '''
        Set the status of the agent as 'training' or 'testing'.
        '''
        self._training_flag = (value == 'training')

    def act(self, state, **kwargs):
        '''
        Return an action based on the given state.

        Arguments
        ---------
            state: the state for which the action should be returned.
            actions: the set of possible actions to choose from.

        Note: If state is 'training' (_training_flag=false), then this function should not return any random move due to exploration.
        '''
        pass

    def learn(self, **kwargs):
        '''Learn using either history or action, reward, and state.''' 
        pass

    def reset(self):
        '''Reset the agent at the end of a learning episode.''' 
        pass

    def load(self, **kwargs):
        '''
        Load an agent from a file.

        Arguments
        ---------
            filename: the name of the file to be loaded.

        Raises ValueError if the filename is not specified.
        '''
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')
        with open(filename + '.pkl', 'rb') as f:
            self.__dict__ = pickle.load(f)

    def save(self, **kwargs):
        '''
        Save the agent to a file.

        Arguments
        ---------
            filename: the name of the file to be saved.

        Raises ValueError if the filename is not specified.
        '''
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')
        with open(filename + '.pkl', 'wb+') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def report(self, **kwargs):
        '''return a report as a string.''' 
        raise NotImplementedError


if __name__ == '__main__':
    main()
