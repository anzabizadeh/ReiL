# -*- coding: utf-8 -*-
'''
subject class
=============

This `subject` class is the base class of all subject classes. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from typing import Any, Dict, Optional, Tuple

from reil import rlbase, rldata
from reil.stats import rl_functions


class Subject(rlbase.RLBase):
    '''
    The base class of all subject classes.

    Methods
    -------
    is_terminated: returns True if the subject is in the terminal state for
        the agent with the provided _id.

    possible_actions: a list of possible actions for the agent with the provided _id.

    reward: returns the reward for the agent `_id` based on reward definition
        `name`. For subjects that are turn-based, it is a good practice to
        check that an agent is retrieving the reward only when it is the
        agent's turn.

    default_reward: returns the default reward for the agent `_id`. This can
        be a more efficient implementation of the reward, when possible.

    take_effect: gets an action and changes the state accordingly.
        Note: take_effect does not return the reward. `Reward` method should
        be used afterwards to get the realization of reward.

    add_reward_definition: add a new reward definition consisting of a `name`,
        and reward function, and a state definition name.

    reset: resets the state and is_terminated.

    register: registers a new agent and returns its ID or returns ID of an existing agent.

    deregister: deregisters an agent identified by its ID.
    '''

    def __init__(self, **kwargs: Any):
        '''
        Initializes the subject.
        '''

        super().__init__(**kwargs)

        self._agent_list: Dict[str, int] = {}
        self._reward_definitions: Dict[str, Tuple[rl_functions.RLFunction, str]] = {}

    def is_terminated(self, _id: Optional[int] = None) -> bool:
        '''
        Returns False as long as the subject can accept new actions.

        ### Arguments

        _id: ID of the agent that checks termination. In a multi-agent setting,
            e.g. an RTS game, one agent might die and another agent might still
            be alive.
        '''
        raise NotImplementedError

    def possible_actions(self, _id: Optional[int] = None) -> Tuple[rldata.RLData, ...]:
        '''
        Returns a list of possible actions for the agent with ID=_id.

        ### Arguments

        _id: ID of the agent that wants to act on the subject.
        '''
        return (rldata.RLData({'default_action': {'value': None}}),)

    def reward(self, name: str = 'default', _id: Optional[int] = None) -> rldata.RLData:
        '''
        Returns the reward that agent `_id` recieves, based on the reward
        definition `name`.

        ### Arguments

        name: name of the reward definition. If omitted, output of the
            `default_reward` method will be returned. 

        _id: ID of the agent that calls the retrieves the reward.
        '''
        if name.lower() == 'default':
            return self.default_reward(_id)

        f, s = self._reward_definitions[name.lower()]
        temp = f(self.state(s, _id))

        return rldata.RLData({'name': 'reward', 'value': temp, 'lower': None, 'upper': None})

    def default_reward(self, _id: Optional[int] = None) -> rldata.RLData:
        '''
        Returns the default reward definition of the subject for agent `_id`.

        ### Arguments

        _id: ID of the agent that calls the reward method.
        '''
        return rldata.RLData({'name': 'reward', 'value': 0.0, 'lower': None, 'upper': None})

    def take_effect(self, action: rldata.RLData, _id: Optional[int] = None) -> None:
        '''
        Receive an `action` from agent `_id` and transition to the next state.

        ### Arguments

        action: the action sent by the agent that will affect this subject.

        _id: ID of the agent that has sent the action.
        '''
        raise NotImplementedError

    def add_reward_definition(self, name: str,
                              rl_function: rl_functions.RLFunction,
                              state_name: str) -> None:
        '''
        Adds a new reward definition called `name` with function `rl_function`
        that uses state `state_name`.

        ### Arguments

        name: name of the new reward definition. ValueError is raise if the reward
            already exists.

        rl_function: An instance of `RLFunction` that gets the state of the
            subject, and computes the reward. The rl_function should have the
            list if arguments from the state in its definition.

        state_name: The name of the state definition that should be used to
            compute the reward. ValueError is raise if the state_name is
            undefined.

        Note: statistic and reward are basicly doing the same thing. The
            difference is in their application: Statistic should be called at
            the end of each trajectory (sample path) to compute the necessary
            statistics about the performance of the agents and subjects. Reward,
            on the other hand, should be called after each interaction between
            an agent and the subject to guide the reinforcement learning model
            to learn the optimal policy.
        '''
        if name.lower() in self._reward_definitions:
            raise ValueError(f'Reward definition {name} already exists.')

        if state_name.lower() not in self._state_definitions:
            raise ValueError(f'Unknown state name: {state_name}.')

        self._reward_definitions[name.lower()] = (rl_function, state_name)

    def reset(self) -> None:
        ''' Resets the subject, so that it can resume accepting actions.'''
        raise NotImplementedError

    def register(self, agent_name: str) -> int:
        '''
        Registers an agent and returns its ID. If the agent is new, a new ID
        is generated and the agent_name is added to agent_list.

        ### Arguments

        agent_name: the name of the agent to be registered.
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
        Deregisters an agent given its name.

        ### Arguments

        agent_name: the name of the agent to be registered.
        '''
        self._agent_list.pop(agent_name)
