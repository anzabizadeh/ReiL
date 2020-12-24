# -*- coding: utf-8 -*-
'''
subject class
=============

This `subject` class is the base class of all subject classes.
'''

from reil.datatypes.components import SecondayComponent
from typing import Any, List, Optional, Tuple, TypeVar

from reil import stateful
from reil.datatypes import reildata
# from reil.stats import reil_functions


class AgentList:
    '''
    Create and maintain a list of registered `agents` for the `subject`.


    :meta private:
    '''
    def __init__(self, min_agent_count: int, max_agent_count: int,
                 unique_agents: bool = True):
        '''
        Arguments
        ---------
        min_agent_count:
            The minimum number of `agents` needed to be registered so that the
            `subject` is ready for interaction.

        max_agent_count:
            The maximum number of `agents` that can act on the `subject`.

        unique_agents:
            If `True`, each `agent` can be registered only once.
        '''
        self._id_list: List[int] = []
        self._agent_list: List[str] = []
        self._min_agent_count = min_agent_count
        self._max_agent_count = max_agent_count
        self._unique_agents = unique_agents

    @property
    def ready(self) -> bool:
        '''
        Determine if enough `agents` are registered.

        Returns
        -------
        :
            `True` if enough agents are registered, else `False`.
        '''
        return len(self._id_list) >= self._min_agent_count

    def append(self, agent_name: str, _id: Optional[int] = None) -> int:
        '''
        Add a new `agent` to the end of the list.

        Parameters
        ----------
        agent_name:
            The name of the `agent` to add.

        _id:
            If provided, method tries to register the `agent` with the given
            ID.

        Returns
        -------
        :
            The ID assigned to the `agent`.

        Raises
        ------
        ValueError:
            Capacity is reached. No new agents can be registered.

        ValueError:
            ID is already taken.

        ValueError:
            `agent_name` is already registered with a different ID.
        '''
        if (0 < self._max_agent_count < len(self._id_list)):
            raise ValueError('Capacity is reached. No new agents can be'
                             ' registered.')

        if _id is not None:
            if _id in self._id_list:
                raise ValueError(f'{_id} is already taken.')

            if self._unique_agents:
                try:
                    current_id = self._id_list[
                        self._agent_list.index(agent_name)]
                    if _id == current_id:
                        return _id
                    else:
                        raise ValueError(
                            f'{agent_name} is already registered with '
                            f'ID: {current_id}.')
                except ValueError:
                    pass

            new_id = _id
        else:
            new_id = max(self._id_list, default=0) + 1

        self._agent_list.append(agent_name)
        self._id_list.append(new_id)

        return new_id

    def remove(self, _id: int):
        '''
        Remove the `agent` registered by ID=`_id`.

        Arguments
        ---------
        _id:
            ID of the `agent` to remove.
        '''
        agent_name = self._agent_list[self._id_list.index(_id)]
        self._agent_list.remove(agent_name)
        self._id_list.remove(_id)


class Subject(stateful.Stateful):
    '''
    The base class of all subject classes.
    '''

    def __init__(self,
                 min_agent_count: int = 1, max_agent_count: int = -1,
                 unique_agents: bool = True,
                 sequential_interaction: bool = True,
                 **kwargs: Any):
        '''
        Arguments
        ---------
        min_agent_count:
            The minimum number of `agents` needed to be registered so that the
            `subject` is ready for interaction.

        max_agent_count:
            The maximum number of `agents` that can act on the `subject`.

        unique_agents:
            If `True`, each `agent` can be registered only once.

        sequential_interaction:
            If `True`, `agents` can only act on the `subject` in the order they
            are added.
        '''
        super().__init__(**kwargs)

        self._sequential_interaction = sequential_interaction
        self._agent_list = AgentList(min_agent_count=min_agent_count,
                                     max_agent_count=max_agent_count,
                                     unique_agents=unique_agents)
        self._reward = SecondayComponent(name='reward',
                                         primary_component=self._state)
        # self._reward_definitions: Dict[
        #     str, Tuple[reil_functions.ReilFunction, str]] = {}

    def is_terminated(self, _id: Optional[int] = None) -> bool:
        '''
        Determine if the `subject` is terminated for the given `agent` ID.

        Arguments
        ---------
        _id:
            ID of the agent that checks termination. In a multi-agent setting,
            e.g. an RTS game, one agent might die and another agent might still
            be alive.

        Returns
        -------
        :
            `False` as long as the subject can accept new actions from the
            `agent`. If `_id` is `None`, then returns `True` if no `agent`
            can act on the `subject`.
        '''
        raise NotImplementedError

    def possible_actions(self, _id: int = 0) -> Tuple[reildata.ReilData, ...]:
        '''
        Generate the list of possible actions.

        Arguments
        ---------
        _id:
            ID of the `agent` that wants to act on the `subject`.

        Returns
        -------
        :
            A list of possible actions for the `agent` with ID=_id.
        '''
        return (reildata.ReilData.single_base(name='default_action'),)

    def reward(self,
               name: str = 'default', _id: int = 0) -> reildata.ReilData:
        '''
        Compute the reward that `agent` receives, based on the reward
        definition `name`.

        Arguments
        ---------
        name:
            The name of the reward definition. If omitted, output of the
            `default_reward` method will be returned.

        _id:
            The ID of the calling `agent`.

        Returns
        -------
        :
            The reward for the given `agent`.
        '''
        return self._reward(name, _id)

    def take_effect(self, action: reildata.ReilData, _id: int = 0) -> None:
        '''
        Receive an `action` from `agent` with ID=`_id` and transition to
        the next state.

        Arguments
        ---------
        action:
            The action sent by the `agent` that will affect this `subject`.

        _id:
            ID of the `agent` that has sent the `action`.
        '''
        raise NotImplementedError

    def reset(self) -> None:
        ''' Reset the `subject`, so that it can resume accepting actions.'''
        raise NotImplementedError

    def register(self, agent_name: str, _id: Optional[int] = None) -> int:
        '''
        Register an `agent` and return its ID. If the `agent` is new, a new ID
        is generated and the `agent_name` is added to the list of
        registered agents.

        Arguments
        ---------
        agent_name:
            The name of the `agent` to be registered.

        _id:
            The ID of the agent to be used. If not provided, subject will
            assign an ID to the `agent`.

        Returns
        -------
        :
            ID of the registered `agent`.

        Raises
        ------
        ValueError:
            Attempt to register an already registered `agent` with a new ID.

        ValueError:
            Attempt to register an `agent` with an already assigned ID.

        ValueError:
            Reached max capacity.
        '''
        return self._agent_list.append(agent_name=agent_name, _id=_id)

    def deregister(self, agent_id: int) -> None:
        '''
        Deregister an `agent` given its ID.

        Arguments
        ---------
        agent_id:
            The ID of the `agent` to be deregistered.
        '''
        self._agent_list.remove(agent_id)


SubjectType = TypeVar('SubjectType', bound=Subject)


# def reward(self,
#            _id: int = 0, name: Optional[str] = None) -> reildata.ReilData:
#     '''
#     Compute the reward that `agent` receives, based on the reward
#     definition `name`.

#     Arguments
#     ---------
#     _id:
#         The ID of the calling `agent`.

#     name:
#         The name of the reward definition. If omitted, output of the
#         `default_reward` method will be returned.

#     Returns
#     -------
#     :
#         The reward for the given `agent`.
#     '''
#     if name is None or name.lower() == 'default':
#         return self.default_reward(_id)

#     f, s = self._reward_definitions[name.lower()]
#     temp = f(self.state(s, _id))

#     return reildata.ReilData.single_base(name='reward', value=temp)

# def default_reward(self, _id: int = 0) -> reildata.ReilData:
#     '''
#     Compute the default reward definition of the subject for agent `_id`.

#     Arguments
#     ---------
#     _id:
#         ID of the `agent` that calls the reward method.

#     Returns
#     -------
#     :
#         The reward for the given `agent`.
#     '''
#     return reildata.ReilData.single_base(name='reward', value=0.0)

# def add_reward_definition(self, name: str,
#                           rl_function: reil_functions.ReilFunction,
#                           state_name: str) -> None:
#     '''
#     Add a new reward definition called `name` with function `rl_function`
#     that uses state `state_name`.

#     Arguments
#     ---------
#     name:
#         The name of the new reward definition.

#     rl_function:
#         An instance of `ReilFunction` that gets the state of the
#         `subject`, and computes the reward. The `rl_function` should
#         have the list of arguments from the state in its definition.

#     state_name:
#         The name of the state definition that should be used to
#         compute the reward.

#     Raises
#     ------
#     ValueError:
#         The reward `name` already exists.

#     ValueError:
#         The `state_name` is undefined.

#     Notes
#     -----
#         `statistic` and `reward` are basicly doing the same thing. The
#         difference is in their application: `statistic` should be called at
#         the end of each trajectory (sampled path) to compute the necessary
#         statistics about the performance of the `agents` and `subjects`.
#         `reward`, on the other hand, should be called after each
#         interaction between an `agent` and the `subject` to guide the
#         reinforcement learning model to learn the optimal policy.
#     '''
#     if name.lower() in self._reward_definitions:
#         raise ValueError(f'Reward definition {name} already exists.')

#     if state_name.lower() not in self._state_definitions:
#         raise ValueError(f'Unknown state name: {state_name}.')

#     self._reward_definitions[name.lower()] = (rl_function, state_name)
