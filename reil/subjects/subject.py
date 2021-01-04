# -*- coding: utf-8 -*-
'''
subject class
=============

This `subject` class is the base class of all subject classes.
'''

from reil.datatypes import ReilData, SecondayComponent
from typing import Any, Optional, Tuple, TypeVar

from reil import stateful


class Subject(stateful.Stateful):
    '''
    The base class of all subject classes.
    '''

    def __init__(self,
                 sequential_interaction: bool = True,
                 **kwargs: Any):
        '''
        Arguments
        ---------
        sequential_interaction:
            If `True`, `agents` can only act on the `subject` in the order they
            are added.

        Notes
        -----
        `sequential_interaction` is not enforced (implemented) yet!
        '''
        super().__init__(**kwargs)

        self._sequential_interaction = sequential_interaction
        self.reward = SecondayComponent(name='reward',
                                         primary_component=self.state,
                                         enabled=False)

    def _default_reward_definition(
            self, _id: Optional[int] = None) -> ReilData:
        return ReilData.single_base(name='default_reward', value=None)

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

    def possible_actions(self, _id: int = 0) -> Tuple[ReilData, ...]:
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
        return (ReilData.single_base(name='default_action'),)

    def take_effect(self, action: ReilData, _id: int = 0) -> None:
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
        self.reward.enable()

    def reset(self) -> None:
        '''Reset the `subject`, so that it can resume accepting actions.'''
        self.reward.disable()


SubjectType = TypeVar('SubjectType', bound=Subject)

# def reward(self,
#            name: str = 'default', _id: int = 0) -> reildata.ReilData:
#     '''
#     Compute the reward that `agent` receives, based on the reward
#     definition `name`.

#     Arguments
#     ---------
#     name:
#         The name of the reward definition. If omitted, output of the
#         `default_reward` method will be returned.

#     _id:
#         The ID of the calling `agent`.

#     Returns
#     -------
#     :
#         The reward for the given `agent`.
#     '''
#     return self._reward(name, _id)


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
