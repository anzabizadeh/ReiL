# -*- coding: utf-8 -*-
'''
iterableSubject class
=============

This `iterable subject` class takes any `subject` object and returns an iterator. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import pathlib
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import reil.subjects as rlsubjects
from reil import stateful
from reil.datatypes import reildata


class IterableSubject(stateful.Stateful):
    '''
    Makes any subject an iterable.

    Attributes
    ----------
        subject:
        save_instances: whether to save instances of the `subject` class or not (Default=False).
        use_existing_instances: whether try to load instances before attempting to create them (Default=True).
        save_path: the path where instances should be saved/ loaded from (Default='./').
        save_prefix: a prefix in instance filenames (Default='').
        instance_counter_start: the index of the first instance of the subject (Default=0).
        instance_counter_end: an int or a list of the last index of the instances (Default=1).
        auto_rewind: whether to rewind after it hits the last instance (Default=False).

    Methods
    -------
        register: register a new agent and return its ID or return ID of an existing agent.
        take_effect: get an action and change the state accordingly.
        reset: reset the state and is_terminated.
    '''

    def __init__(self,
                 subject: rlsubjects.Subject,
                 save_instances: bool = False,
                 use_existing_instances: bool = True,
                 save_path: Union[pathlib.Path, str] = '',
                 save_prefix: str = '',
                 instance_counter_start: int = 0,
                 instance_counter: int = -1,
                 # -1: infinite
                 instance_counter_end: Union[int, List[int]] = -1,
                 end_index: int = 0,
                 auto_rewind: bool = False,
                 **kwargs: Any) -> None:

        self._subject = subject
        self._agent_list = {}
        self._save_instances = save_instances
        self._use_existing_instances = use_existing_instances
        self._save_path = save_path
        self._save_prefix = save_prefix
        self._instance_counter_start = instance_counter_start
        self._instance_counter = instance_counter
        self._end_index = end_index
        self._auto_rewind = auto_rewind

        super().__init__(**kwargs)

        if isinstance(instance_counter_end, int):
            self._instance_counter_end = [instance_counter_end]
        else:
            self._instance_counter_end = instance_counter_end

        if self._instance_counter_end[0] == -1:
            self._stop_check: Callable[[int, int], bool] = lambda current, end: False
        else:
            self._stop_check: Callable[[int, int], bool] = lambda current, end: current > end

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[int, rlsubjects.Subject]:
        self._instance_counter += 1

        try:
            end = self._instance_counter_end[self._end_index]
        except IndexError:
            if self._auto_rewind:
                self._end_index = 0
                self._instance_counter = 1
                end = self._instance_counter_end[self._end_index]
            else:
                raise StopIteration

        if self._stop_check(self._instance_counter, end):
            self._end_index += 1
            self._instance_counter -= 1
            raise StopIteration
        else:
            current_instance = ''.join(
                (self._save_prefix, f'{self._instance_counter:06}'))
            new_instance = True
            if self._use_existing_instances:
                try:
                    self._subject.load(path=self._save_path,
                                       filename=current_instance)
                    new_instance = False
                except FileNotFoundError:
                    self._subject.reset()
            else:
                self._subject.reset()

            if self._save_instances and new_instance:
                self._subject.save(path=self._save_path,
                                   filename=current_instance)

        return (self._instance_counter, self._subject)

    @property
    def state(self) -> reildata.ReilData:
        return self._subject.state

    @property
    def is_terminated(self) -> bool:
        return self._subject.is_terminated

    @property
    def possible_actions(self) -> Sequence[reildata.ReilData]:
        return self._subject.possible_actions

    def take_effect(self, action: reildata.ReilData, _id: Optional[int] = None):
        return self._subject.take_effect(action, _id)

    def reset(self) -> None:
        '''
        Reset the subject.
        This function allows non-iterable use of the defined subject. (e.g. in `env.trajectory()`).
        If you want to reset the iterator itself, use `rewind()` method instead.
        '''
        self._subject.reset()

    def rewind(self) -> None:
        '''
        Rewind the iterator object.
        '''
        self._instance_counter = self._instance_counter_start - 1

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

    def __repr__(self) -> str:
        try:
            return f'{self.__class__.__qualname__} -> {self._subject.__repr__()}'
        except AttributeError:
            return self.__class__.__qualname__
