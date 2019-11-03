# -*- coding: utf-8 -*-
'''
subject class
=============

This `subject` class is the super class of all subject classes. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from ..rlbase import RLBase


class Subject(RLBase):
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
        printable: return a readable format of subject's state
    '''
    def __init__(self, **kwargs):
        RLBase.__init__(self, **kwargs)
        RLBase.set_defaults(self, agent_list={})
        RLBase.set_params(self, **kwargs)

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
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
            _id = max(self._agent_list.values()) + 1
            self._agent_list[agent_name] = _id
            return _id

    def deregister(self, agent_name):
        '''
        Deegisters an agent given its name.
        \nArguments:
        \n    agent_name: the name of the agent to be registered.
        '''
        self._agent_list.pop(agent_name)

    def take_effect(self, _id, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def printable(self):
        pass

    def __repr__(self):
        return 'Subject'