# -*- coding: utf-8 -*-
'''
subject class
=============

This `subject` class is the super class of all subject classes. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import pickle


class Subject:
    '''
    Super class of all subject classes.
    
    Attributes
    ----------
        state: the state of the subject as a ValueSet.
        is_terminated: whether the subject is finished or not.
        possible_actions: a list of possible actions.

    Methods
    -------
        register: register a new agent and return its ID or return ID of an existing agent.
        take_effect: get an action and change the state accordingly.
        reset: reset the state and is_terminated.
        load: load a subject from a file.
        save: save the subject to a file.
        printable: return a readable format of subject's state
    '''
    def __init__(self):
        self._agent_list = {}

    @property
    def state(self):
        raise NotImplementedError

    @property
    def is_terminated(self):
        raise NotImplementedError

    @property
    def possible_actions(self):
        pass

    def register(self, agent_name):
        '''
        Registers an agent and returns its ID. If the agent is new, a new ID is generated and the agent_name is added to agent_list.
        \nArguments:
        \n    agent_name: the name of the agent to be registered.
        '''
        try:
            return self._agent_list[agent_name]
        except KeyError:
            _id = len(self._agent_list) + 1
            self._agent_list[agent_name] = _id
            return _id

    def take_effect(self, _id, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def load(self, **kwargs):
        '''
        Load a subject from a file.

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
        Save the subject to a file.

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

    def printable(self):
        pass
