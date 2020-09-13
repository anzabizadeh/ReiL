# -*- coding: utf-8 -*-
'''
subject class
=============

This `subject` class is the super class of all subject classes. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from typing import Any, Dict, Optional, Sequence

from reil import rlbase, rldata


class Subject(rlbase.RLBase):
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
    '''

    def __init__(self,
                 ex_protocol_current: Dict[str, str] = {'state': 'standard',
                                                        'possible_actions': 'standard',
                                                        'take_effect': 'standard',
                                                        'complete_state': True},
                 ex_protocol_options: Dict[str, Sequence[str]] = {'state': ['standard'],
                                                                  'possible_actions': ['standard'],
                                                                  'take_effect': ['standard', 'no_reward'],
                                                                  'complete_state': [False, True]},
                 **kwargs: Any):

        kwargs['name'] = kwargs.get('name', __name__)
        kwargs['logger_name'] = kwargs.get('logger_name', __name__)

        super().__init__(ex_protocol_current=ex_protocol_current,
                         ex_protocol_options=ex_protocol_options,
                         **kwargs)

        self._agent_list: Dict[str, int] = {}

    @property
    def state(self) -> rldata.RLData:
        raise NotImplementedError

    @property
    def complete_state(self) -> rldata.RLData:
        '''
        Returns all the information that the subject can provide.
        '''
        raise NotImplementedError

    @property
    def is_terminated(self) -> bool:
        raise NotImplementedError

    @property
    def possible_actions(self) -> Sequence[rldata.RLData]:
        return [rldata.RLData({'default_action': {'value': None}})]

    def register(self, agent_name: str) -> int:
        '''
        Registers an agent and returns its ID. If the agent is new, a new ID is generated and the agent_name is added to agent_list.
        \nArguments:
        \n    agent_name: the name of the agent to be registered.
        '''
        try:
            return self._agent_list[agent_name]
        except KeyError:
            try:
                _id = max(self._agent_list.values()) + 1
            except ValueError:
                _id = 1

            self._agent_list[agent_name] = _id
            return _id

    def deregister(self, agent_name: str) -> None:
        '''
        Deegisters an agent given its name.
        \nArguments:
        \n    agent_name: the name of the agent to be registered.
        '''
        self._agent_list.pop(agent_name)

    def take_effect(self, action: rldata.RLData, _id: Optional[int] = None) -> rldata.RLData:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return 'Subject'
