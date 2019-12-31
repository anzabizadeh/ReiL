# -*- coding: utf-8 -*-
'''
iterableSubject class
=============

This `iterable subject` class takes any `subject` object and returns an iterator. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from ..rlbase import RLBase


class IterableSubject(RLBase):
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
    def __init__(self, subject=None, **kwargs):
        self.set_defaults(subject=subject,
                            agent_list={},
                            save_instances=False,
                            use_existing_instances=True,
                            save_path='./',
                            save_prefix='',
                            instance_counter_start=0,
                            instance_counter=-1,
                            instance_counter_end=[1],  # -1: infinite
                            end_index=0,
                            auto_rewind=False
                            )
        self.set_params(subject=subject, **kwargs)
        super().__init__(**kwargs)

        if 'filename' in kwargs:
            if 'path' in kwargs:
                self.load(filename=kwargs['filename'], path=kwargs['path'])
            else:
                self.load(filename=kwargs['filename'])
            return

        if isinstance(self._instance_counter_end, int):
            self._instance_counter_end = [self._instance_counter_end]

        if self._instance_counter_end[0] == -1:
            self._stop_check = lambda current, end: False
        else:
            self._stop_check = lambda current, end: current >= end

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._subject = None
            self._agent_list = {}
            self._save_instances = False
            self._use_existing_instances = True
            self._save_path = './'
            self._save_prefix = ''
            self._instance_counter_start = 0
            self._instance_counter = -1
            self._instance_counter_end = [1]
            self._end_index = 0
            self._auto_rewind = False

    def __iter__(self):
        return self

    def __next__(self):
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
            raise StopIteration
        else:
            current_instance = ''.join((self._save_prefix, f'{self._instance_counter:06}'))
            new_instance = True
            if self._use_existing_instances:
                try:
                    self._subject.load(path=self._save_path, filename=current_instance)
                    new_instance = False
                except FileNotFoundError:
                    self._subject.reset()
            else:
                self._subject.reset()

            if self._save_instances and new_instance:
                self._subject.save(path=self._save_path, filename=current_instance)

        return (self._instance_counter, self._subject)

    @property
    def state(self):
        return self._subject.state

    @property
    def is_terminated(self):
        return self._subject.is_terminated

    @property
    def possible_actions(self):
        return self._subject.possible_actions

    def take_effect(self, action, _id=None):
        return self._subject.take_effect(action, _id)

    def reset(self):
        '''
        Reset the subject.
        This function allows non-iterable use of the defined subject. (e.g. in `env.trajectory()`).
        If you want to reset the iterator itself, use `rewind()` method instead.
        '''
        self._subject.reset()

    def rewind(self):
        '''
        Rewind the iterator object.
        '''
        self._instance_counter = self._instance_counter_start - 1

    def register(self, agent_name):
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

    def deregister(self, agent_name):
        '''
        Deegisters an agent given its name.
        \nArguments:
        \n    agent_name: the name of the agent to be registered.
        '''
        self._agent_list.pop(agent_name)

    def __repr__(self):
        try:
            return self._subject.__repr__()
        except AttributeError:
            return 'iterable_subject'