# -*- coding: utf-8 -*-
'''
subject class
=============

This `subject` class is the base class of all subject classes. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from reil import rlbase, rldata, utils
from reil.stats import rl_functions

StateComponentTuple = namedtuple('StateComponentTuple', ('func', 'kwargs'),
                            defaults=({}))

StateComponentFunction = Callable[..., Union[Dict[str, Any], rldata.RLData]]
ComponentInfo = Union[str, Tuple[str, Dict[str, Any]]]


class Subject(rlbase.RLBase):
    '''
    The base class of all subject classes.

    Methods
    -------
    is_terminated: returns True if the subject is in the terminal state for
        the agent with the provided _id.

    possible_actions: a list of possible actions for the agent with the provided _id.

    state: the state of the subject as an RLData. Different state definitions
        can be introduced using `add_state_definition` method. _id is
        available, in case in the implementation, State is agent-dependent.
        (For example in games with partial map visibility).
        For subjects that are turn-based, it is a good practice to check
        that an agent is retrieving the state only when it is the agent's
        turn.

    default_state: the default state definition provided by the subject.
        This can be a more efficient implementation of the state, when it is
        possible.

    complete_state: returns an RLData consisting of all available state
        components. _id is available, in case in the implementation, State is
        agent-dependent.

    reward: returns the reward for the agent `_id` based on reward definition
        `name`. For subjects that are turn-based, it is a good practice to
        check that an agent is retrieving the reward only when it is the
        agent's turn.

    default_reward: returns the default reward for the agent `_id`. This can
        be a more efficient implementation of the reward, when possible.

    statistic: computes the value of the given statistic for the agent `_id`
        based on the statistic definition `name`. It should normally be called
        after each sampled path (trajectory).

    default_statistic: returns the default statistic for the agent `_id`. This
        can be a more efficient implementation of the statistic, when possible.

    take_effect: gets an action and changes the state accordingly.
        Note: take_effect does not return the reward. `Reward` method should
        be used afterwards to get the realization of reward.

    add_state_definition: add a new state definition consisting of a `name`,
        and a list of state components. Each element in the list can be
        string representing component's name, a tuple representing name and
        positional arguments,  a tuple representing name and keyword
        arguments, or a tuple representing name, positional and keyword arguments.

    add_reward_definition: add a new reward definition consisting of a `name`,
        and reward function, and a state definition name.

    add_statistic_definition: add a new statistic definition consisting of a
        `name`, and statistic function, and a state definition name.

    reset: resets the state and is_terminated.

    register: registers a new agent and returns its ID or returns ID of an existing agent.

    deregister: deregisters an agent identified by its ID.

    _generate_state_components: used by the subject during the `__init__`
        to create state components.
    '''

    def __init__(self, **kwargs: Any):
        '''
        Initializes the subject.
        '''

        super().__init__(**kwargs)

        self._agent_list: Dict[str, int] = {}
        self._state_definitions: Dict[str, List[StateComponentTuple]] = {'default': []}
        self._reward_definitions: Dict[str, Tuple[rl_functions.RLFunction, str]] = {}
        self._statistic_definitions: Dict[str, Tuple[rl_functions.RLFunction, str]] = {}

        self._available_state_components: Dict[str, StateComponentFunction] = {}
        # self._generate_state_components()

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

    def state(self, name: str = 'default', _id: Optional[int] = None) -> rldata.RLData:
        '''
        Returns the current state of the subject as agent `_id` might see, based
        on the state definition `name`.

        ### Arguments

        name: name of the state definition. If omitted, output of the
            `default_state` method will be returned. 

        _id: ID of the agent that calls the state method. In a multi-agent
            setting, e.g. an RTS game with fog of war, agents would see the world
            differently.
        '''
        if name.lower() == 'default':
            return self.default_state(_id)

        return rldata.RLData([f.func(**f.kwargs)
                              for f in self._state_definitions[name.lower()]])

    def default_state(self, _id: Optional[int] = None) -> rldata.RLData:
        '''
        Returns the default state definition of the subject as agent `_id` might
        see.

        ### Arguments

        _id: ID of the agent that calls the state method. In a multi-agent
            setting, e.g. an RTS game with fog of war, agents would see the world
            differently.
        '''
        return self.complete_state(_id)

    def complete_state(self, _id: Optional[int] = None) -> rldata.RLData:
        '''
        Returns all the information that the subject can provide.

         The default implementation returns all available state components with
         their default settings. Based on the state component definition of a
         child class, this can include redundant or incomplete information.

        ### Arguments

        _id: ID of the agent that calls the complete_state method.
        '''
        return rldata.RLData([f()  # type: ignore
                              for f in self._available_state_components.values()])

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

    def statistic(self, name: str = 'default', _id: Optional[int] = None) -> rldata.RLData:
        '''
        Returns the statistic that agent `_id` requests, based on the statistic
        definition `name`.

        ### Arguments

        name: name of the statistic definition. If omitted, output of the
            `default_statistic` method will be returned. 

        _id: ID of the agent that calls the retrieves the statistic.
        '''
        if name.lower() == 'default':
            return self.default_statistic(_id)

        f, s = self._statistic_definitions[name.lower()]
        temp = f(self.state(s, _id))

        return rldata.RLData({'name': 'reward', 'value': temp, 'lower': None, 'upper': None})

    def default_statistic(self, _id: Optional[int] = None) -> rldata.RLData:
        '''
        Returns the default statistic definition of the subject for agent `_id`.

        ### Arguments

        _id: ID of the agent that calls the reward method.
        '''
        return rldata.RLData({'name': 'default_stat', 'value': 0.0, 'lower': None, 'upper': None})

    def take_effect(self, action: rldata.RLData, _id: Optional[int] = None) -> None:
        '''
        Receive an `action` from agent `_id` and transition to the next state.

        ### Arguments

        action: the action sent by the agent that will affect this subject.

        _id: ID of the agent that has sent the action.
        '''
        raise NotImplementedError

    def add_state_definition(self, name: str,
                             component_list: Tuple[ComponentInfo, ...]) -> None:
        '''
        Adds a new state definition called `name` with state components provided
        in `component_list`.

        ### Arguments

        name: name of the new state definition. ValueError is raise if the state
            already exists.

        component_list: A tuple consisting of component information. Each element
            in the list should be either (1) name of the component, or (2)
            a tuple with the name and a dict of kwargs.
        '''
        _name = name.lower()
        if _name in self._state_definitions:
            raise ValueError(f'State definition {name} already exists.')

        self._state_definitions[_name] = []
        for component in component_list:
            if isinstance(component, str):
                f = self._available_state_components[component]
                kwargs = {}
            elif isinstance(component, (tuple, list)):
                f = self._available_state_components[component[0]]
                kwargs = utils.get_argument(component[1], {})
            else:
                raise ValueError('Items in the component_list should be one of: '
                                 '(1) name of the component, '
                                 '(2) a tuple with the name and a dict of kwargs.')
            self._state_definitions[_name].append(
                StateComponentTuple(f, kwargs))

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
            difference is that statistic should be called at the end of each
            trajectory to compute the necessary statistics about the performance
            of the agents and subjects. Reward, on the other hand, should be
            called after each interaction between an agent and the subject.
        '''
        if name.lower() in self._reward_definitions:
            raise ValueError(f'Reward definition {name} already exists.')

        if state_name.lower() not in self._state_definitions:
            raise ValueError(f'Unknown state name: {state_name}.')

        self._reward_definitions[name.lower()] = (rl_function, state_name)

    def add_statistic_definition(self, name: str,
                                 rl_function: rl_functions.RLFunction,
                                 state_name: str) -> None:
        '''
        Adds a new statistic definition called `name` with function `rl_function`
        that uses state `state_name`.

        ### Arguments

        name: name of the new statistic definition. ValueError is raise if the
            statistic already exists.

        rl_function: An instance of `RLFunction` that gets the state of the
            subject, and computes the statistic. The rl_function should have the
            list if arguments from the state in its definition.

        state_name: The name of the state definition that should be used to
            compute the statistic. ValueError is raise if the state_name is
            undefined.

        Note: statistic and reward are basicly doing the same thing. The
            difference is that statistic should be called at the end of each
            trajectory to compute the necessary statistics about the performance
            of the agents and subjects. Reward, on the other hand, should be
            called after each interaction between an agent and the subject.
        '''
        if name.lower() in self._statistic_definitions:
            raise ValueError(f'Statistic definition {name} already exists.')

        if state_name.lower() not in self._state_definitions:
            raise ValueError(f'Unknown state name: {state_name}.')

        self._statistic_definitions[name.lower()] = (rl_function, state_name)

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

    def _generate_state_components(self) -> None:
        '''
        Generates all state components.

        This method should be implemented for all subjects. Each state component
        is a function/ method that computes the given state component. The
        function can have arguments with default values. It should have **kwargs
        arguments to avoid raising exceptions if unnecessary arguments are passed
        on to it.

        Finally, the function should fill a dictionary of state component names
        as keys and functions as values.

        >>> class Dummy(Subject):
        ...     some_attribute = None
        ...     def _generate_state_components(self) -> None:
        ...         def get_some_attribute(**kwargs):
        ...             return self.some_attribute
        ...         self._available_state_components = {
        ...             'some_attribute': get_some_attribute
        ...         }
        '''
        raise NotImplementedError
