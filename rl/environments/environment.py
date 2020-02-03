# -*- coding: utf-8 -*-
'''
environment class
=================

This `environment` class provides a learning environment for any reinforcement learning agent on any subject. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from dill import load, dump, HIGHEST_PROTOCOL
from functools import reduce
import os
import sys
import pandas as pd

from ..rlbase import RLBase
from ..rldata import RLData
import rl.agents as agents
import rl.subjects as subjects


class Environment(RLBase):
    '''
    Provide a learning environment for agents and subjects.

    Attributes
    ----------

    Methods
    -------
        add: add a set of objects (agents/ subjects) to the environment.
        remove: remove objects (agents/ subjects) from the environment.
        assign: assign agents to subjects.
        elapse: move forward in time and interact agents and subjects.
        elapse_iterable: iterate over instances of each subject and interact agents in each subject instance.
        trajectory: extract (state, action, reward) trajectory.
        load: load an object (agent/ subject) or an environment.
        save: save an object (agent/ subject) or the current environment.

    Agents act on subjects and receive the reward of their action and the new state of subjects.
    Then agents learn based on this information to act better.
    '''

    def __init__(self, **kwargs):
        '''
        Create a new environment.

        Arguments
        ---------
            filename: if given, Environment attempts to open a saved environment by openning the file. This argument cannot be used along other arguments.
            path: path of the file to be loaded (should be used with filename)
            name: the name of the environment
            episodes: number of episodes of run. (Default = 1)
            termination: when to terminate one episode. (Default = 'any')
                'any': terminate if any of the subjects is terminated.
                'all': terminate if all the subjects are terminated.
            reset: how to reset subjects after each episode. (Default = 'any')
                'any': reset only the subject that is terminated.
                'all': reset all subjects.
            learning_method: how to learn from each episode. (Default = 'every step')
                'every step': learn after every move.
                'history': learn after each episode.
        '''
        self.set_defaults(agent={}, subject={}, assignment_list={},
                            episodes=1, max_steps=10000, termination='any', reset='any',
                            learning_batch_size=-1,
                            learning_method='every step', total_experienced_episodes={})
        self.set_params(**kwargs)
        super().__init__(**kwargs)
        if 'filename' in kwargs:
            self.load(path=kwargs.get('path', '.'),
                      filename=kwargs['filename'])
            return

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._agent, self._subject, self._assignment_list = {}, {}, {}
            self._episodes, self._total_experienced_episodes, self._max_steps = 1, {}, 10000
            self._termination, self._reset, self._learning_batch_size, self._learning_method = 'any', 'all', -1, 'every step'

    def add(self, **kwargs):
        '''
        Add agents or subjects to the environment.

        Arguments
        ---------
            agents: a dictionary consist of agent name and agent object. Names should be unique, otherwise overwritten.
            subjects: a dictionary consist of subject name and subject object. Names should be unique, otherwise overwritten.
        '''
        try:
            for name, agent in kwargs['agents'].items():
                self._agent[name] = agent
        except (IndexError, KeyError):
            pass

        try:
            for name, subject in kwargs['subjects'].items():
                self._subject[name] = subject
        except (IndexError, KeyError):
            pass

    def remove(self, **kwargs):
        '''
        Remove agents or subjects from the environment.

        Arguments
        ---------
            agents: a list of agent names to be deleted.
            subjects: a list of subject names to be deleted.

        Raises KeyError if the agent is not found.
        '''
        for name in kwargs['agents']:
            try:
                del self._agent[name]
            except KeyError:
                raise KeyError(f'Agent {name} not found!')

        for name in kwargs['subjects']:
            try:
                del self._subject[name]
            except KeyError:
                raise KeyError(f'Subject {name} not found!')

    def assign(self, agent_subject_names):
        '''
        Assign agents to subjects.

        Arguments
        ---------
            agent_subject_names: a list of agent subject tuples.

        Raises ValueError if an agent or subject is not found.
        Note: An agent cannot be assigned to act on multiple subjects, but a subject can be affected by multiple agents. 
        '''
        for agent_name, subject_name in agent_subject_names:
            if agent_name not in self._agent:
                raise ValueError(f'Agent {agent_name} not found!')
            if subject_name not in self._subject:
                raise ValueError(f'Subject {subject_name} not found!')
            _id = self._subject[subject_name].register(agent_name)

            self._total_experienced_episodes[(agent_name, subject_name)] = 0
            self._assignment_list[(agent_name, subject_name)] = _id
            # try:
            #     self._assignment_list[agent_name].append((subject_name, _id))
            # except KeyError:
            #     self._assignment_list[agent_name] = [(subject_name, _id)]

    def divest(self, agent_subject_names):
        '''
        Divest agent subject assignment.

        Arguments
        ---------
            agent_subject_names: a list of agent subject tuples.

        Raises ValueError if an agent or subject is not found.
        Note: An agent can be assigned to act on multiple subjects and a subject can be affected by multiple agents. 
        '''
        for agent_name, subject_name in agent_subject_names:
            if agent_name not in self._agent:
                raise ValueError(f'Agent {agent_name} not found!')
            if subject_name not in self._subject:
                raise ValueError(f'Subject {subject_name} not found!')
            self._subject[subject_name].deregister(agent_name)
            self._assignment_list.pop((agent_name, subject_name))
            del self._total_experienced_episodes[(agent_name, subject_name)]

    def elapse(self, **kwargs):
        '''
        Move forward in time for a number of episodes.

        At each episode, agents are called sequentially to act on their respective subject.
        NOTE: This method loops over agents and only assumes one subject per agent.
        An episode ends if one (all) subject(s) terminates.
        Arguments
        ---------
            episodes: number of episodes of run.
            max_steps: maximum number of steps in each episode (Default = 10,000)
            termination: when to terminate one episode. (Default = 'any')
                'any': terminate if any of the subjects is terminated.
                'all': terminate if all the subjects are terminated.
            reset: how to reset subjects after each episode. (Default = 'any')
                'any': reset only the subject that is terminated.
                'all': reset all subjects.
            learning_method: how to learn from each episode. (Default = 'every step')
                'every step': learn after every move.
                'history': learn after each episode.
            reporting: what to report. (Default = 'none')
                'all': prints every move.
                'none' reports nothing.
                'important' reports important parts only.
            tally: count wins of each agent or not. (Default = 'no')
                'yes': counts the number of wins of each agent.
                'no': doesn't tally.
            step_count: count average number of steps in each episode. (Default = 'no')
                'yes': counts the average number of steps in each episode.
                'no': doesn't count.
        '''
        episodes = kwargs.get('episodes', self._episodes)
        max_steps = kwargs.get('max_steps', self._max_steps)
        if kwargs.get('termination', self._termination).lower() == 'all':
            termination_func = lambda x, y: x & y.is_terminated
            list_of_subjects = [True] + list(self._subject.values())
        else:
            termination_func = lambda x, y: x | y.is_terminated
            list_of_subjects = [False] + list(self._subject.values())

        reset = kwargs.get('reset', self._reset).lower()
        learning_method = kwargs.get(
            'learning_method', self._learning_method).lower()
        reporting = kwargs.get('reporting', 'none').lower()

        tally = kwargs.get('tally', 'no').lower() == 'yes'
        win_count = dict((agent, 0) for agent in self._agent)

        step_count = kwargs.get('step_count', 'no').lower() == 'yes'

        if learning_method == 'none':
            for agent in self._agent.values():
                agent.training_mode = False
        else:
            for agent in self._agent.values():
                agent.training_mode = True

        steps = 0
        for episode in range(episodes):
            if reporting != 'none':
                report_string = f'episode: {episode+1}'
            history = dict((agent_name, []) for agent_name in self._agent)
            done = False
            stopping_criterion = max_steps * (episode+1)
            while not done:
                if steps >= stopping_criterion:
                    break
                steps += 1
                for agent_name, agent in self._agent.items():
                    if not done:
                        subject_name, _id = self._assignment_list[agent_name]
                        subject = self._subject[subject_name]
                        if not subject.is_terminated:
                            state = subject.state
                            possible_actions = subject.possible_actions
                            action = agent.act(state, actions=possible_actions,
                                               episode=self._total_experienced_episodes[(agent_name, subject_name)])
                            reward = RLData(subject.take_effect(action, _id))

                            if reporting == 'all':
                                print(f'step: {steps: 4} episode: {episode:2} state: {state} action: {action} by:{agent_name}')

                            history[agent_name].append({'state': state, 'action': action, 'reward': action})

                            if subject.is_terminated:
                                win_count[agent_name] += int(reward > 0)
                                for affected_agent in self._agent.keys():
                                    if (self._assignment_list[affected_agent][0] == subject_name) & \
                                            (affected_agent != agent_name):
                                        history[affected_agent][-1]['reward'] = -reward

                    done = reduce(termination_func, list_of_subjects)

                if learning_method == 'every step':
                    for agent_name, agent in self._agent.items():
                        agent.learn(state=self._subject[self._assignment_list[agent_name][0]].state,
                            reward=history[agent_name][-1]['reward'])
                        history[agent_name] = []

            if learning_method == 'history':
                for agent_name, agent in self._agent.items():
                    agent.learn(history=history[agent_name])

            for agent_name, subject_name in self._total_experienced_episodes.keys():
                if self._subject[subject_name].is_terminated:
                    self._total_experienced_episodes[(agent_name, subject_name)] += 1
            
            if reset == 'all':
                for agent in self._agent.values():
                    agent.reset()
                for subject in self._subject.values():
                    subject.reset()
            elif reset == 'any':
                for agent_name, subject_name in self._assignment_list.items():
                    if self._subject[subject_name[0]].is_terminated:
                        self._agent[agent_name].reset()
                        self._subject[subject_name[0]].reset()

            if tally & (reporting != 'none'):
                report_string += f'\n tally:'
                for agent in self._agent:
                    report_string += f'\n {agent} {win_count[agent]}'

            if reporting != 'none':
                print(report_string)

        if tally:
            return win_count
        if step_count:
            return steps/episodes

    def elapse_iterable(self, **kwargs):
        '''
        Move forward in time until IterableSubject is consumed.

        For each IterableSubject, agents are called sequentially to act based on the assignment list.
        Method ends if all IterableSubjects are consumed.
        Arguments
        ---------
            max_steps: maximum number of steps in each episode (Default = 10,000)
            learning_batch_size: how many observations to collect before calling agent's `learn` method. (Default = -1)
                -1: collect the whole sample path.
            training_mode: whether it is in training or test mode. (Default: True)
            reset: whether to reset `agents`. 'any` resets agents who acted on a finished subject. `all` resets all agents. (Default = 'any') 
            return_output: a dictionary that indicates whether to return the resulting outputs or not (Default: False)
            return_stats: a dictionary that indicates whether to calculate and return the stats or not (Default: False)
            stats_func: a dictionary that contains functions to calculate stats.
        '''
        max_steps = kwargs.get('max_steps', self._max_steps)
        learning_batch_size = kwargs.get(
            'learning_batch_size', self._learning_batch_size)
        training_mode = kwargs.get('training_mode', dict((agent_subject, True) for agent_subject in self._assignment_list))
        reset = kwargs.get('reset', self._reset).lower()

        temp = kwargs.get('return_output', False)
        if isinstance(temp, dict):
            return_output = temp            
        else:
            return_output = dict((agent_subject, temp) for agent_subject in self._assignment_list)

        temp = kwargs.get('return_stats', False)
        if isinstance(temp, dict):
            return_stats = temp            
        else:
            return_stats = dict((agent_subject, temp) for agent_subject in self._assignment_list)

        temp = kwargs.get('stats_func', lambda a, d: d)
        if isinstance(temp, dict):
            stats_func = temp            
        else:
            stats_func = dict((agent_subject, temp) for agent_subject in self._assignment_list)

        stats = {}
        output = {}

        for subject_name, subject in self._subject.items():
            assigned_agents = list((k[0], v) for k, v in self._assignment_list.items() if k[1] == subject_name)

            for agent_name, _ in assigned_agents:
                self._agent[agent_name].training_mode = training_mode[(agent_name, subject_name)]
            history = dict((agent_name, []) for agent_name, _ in assigned_agents)
            for instance_id, subject_instance in subject:
                steps = 0
                # for agent_name, _ in assigned_agents:
                #     self._agent[agent_name].exchange_protocol = subject_instance.requested_exchange_protocol
                while not subject_instance.is_terminated:
                    for agent_name, _id in assigned_agents:
                        agent = self._agent[agent_name]
                        if subject_instance.is_terminated or steps >= max_steps:
                            break
                        steps += 1

                        state = subject_instance.state
                        possible_actions = subject_instance.possible_actions
                        action = agent.act(state, actions=possible_actions,
                                            episode=self._total_experienced_episodes[(agent_name, subject_name)])
                        reward = subject_instance.take_effect(action, _id)

                        history[agent_name].append({'instance_id': instance_id, 'state': state, 'action': action, 'reward': reward})

                        if subject_instance.is_terminated:
                            for affected_agent in self._agent.keys():
                                if (affected_agent, subject_name) in self._assignment_list and \
                                        (affected_agent != agent_name):
                                    history[affected_agent][-1]['reward'] = -reward

                        if training_mode[(agent_name, subject_name)] and learning_batch_size != -1 and len(history[agent_name]) >= learning_batch_size:
                            agent.learn(history=history[agent_name])
                            history[agent_name] = []

                if training_mode[(agent_name, subject_name)]:
                    self._total_experienced_episodes[(agent_name, subject_name)] += 1

                    if learning_batch_size == -1:
                        for agent_name, _ in assigned_agents:
                            self._agent[agent_name].learn(history=history[agent_name])

            if reset == 'all':
                for agent in self._agent.values():
                    agent.reset()
            elif reset == 'any':
                for agent_name, _ in assigned_agents:
                    self._agent[agent_name].reset()

            for agent_name, _ in assigned_agents:
                if return_output[(agent_name, subject_name)]:
                    try:
                        output[(agent_name, subject_name)].append(history[agent_name])
                    except KeyError:
                        output[(agent_name, subject_name)] = [history[agent_name]]

                if return_stats[(agent_name, subject_name)]:
                    result = stats_func[(agent_name, subject_name)](agent_name, history[agent_name])
                    try:
                        stats[(agent_name, subject_name)].append(result)
                    except KeyError:
                        stats[(agent_name, subject_name)] = [result]

        return stats, output

    def trajectory(self, **kwargs):
        '''
        Extract (state, action, reward) trajectory.

        At each episode, agents are called sequentially to act on their respective subject(s).
        An episode ends if one (all) subject(s) terminates.
        Arguments
        ---------
            max_steps: maximum number of steps in the episode (Default = 10,000)
            termination: when to terminate one episode. (Default = 'any')
                'any': terminate if any of the subjects is terminated.
                'all': terminate if all the subjects are terminated.
        '''
        max_steps = kwargs.get('max_steps', self._max_steps)

        if kwargs.get('termination', self._termination).lower() == 'all':
            termination_func = lambda x, y: x & y.is_terminated
            list_of_subjects = [True] + list(self._subject.values())
        else:
            termination_func = lambda x, y: x | y.is_terminated
            list_of_subjects = [False] + list(self._subject.values())

        for agent in self._agent.values():
            agent.training_mode = False

        for subject in self._subject.values():
            subject.reset()

        history = dict((agent_subject, []) for agent_subject in self._assignment_list)

        done = False
        steps = 0
        while not done:
            if steps >= max_steps:
                break
            steps += 1
            for agent_name, subject_name in self._assignment_list:
                if not done:
                    _id = self._assignment_list[(agent_name, subject_name)]
                    agent = self._agent[agent_name]
                    subject = self._subject[subject_name]
                    if not subject.is_terminated:
                        state = subject.state
                        possible_actions = subject.possible_actions
                        action = agent.act(state, actions=possible_actions)
                        q = float(agent._q(state, action))
                        reward = subject.take_effect(action, _id)

                        history[(agent_name, subject_name)].append({'state': state, 'action': action, 'q': q, 'reward': reward})
                        if subject.is_terminated:
                            for affected_agent in self._agent.keys():
                                if ((affected_agent, subject) in self._assignment_list.keys()) & \
                                        (affected_agent != agent_name):
                                    history[(affected_agent, subject)][-1]['reward'] = -reward

                done = reduce(termination_func, list_of_subjects)

        for sub in self._subject.values():
            sub.reset()

        return history

    def load(self, **kwargs):
        '''
        Load an object or an environment from a file.
        Arguments
        ---------
            filename: the name of the file to be loaded.
            object_name: if specified, that object (agent or subject) is being loaded from file. 'all' loads an environment. (Default = 'all')
        Raises ValueError if the filename is not specified.
        '''
        object_name = kwargs.get('object_name', 'all')
        filename = kwargs.get('filename', self._name)
        path = kwargs.get('path', self._path)

        if object_name == 'all':
            RLBase.load(self, filename=filename)
            self._agent = {}
            self._subject = {}
            for name, obj_type in self._env_data['agents']:
                self._agent[name] = obj_type()
                self._agent[name].load(
                    path=os.path.join(path, filename + '.data'), filename=name)
            for name, obj_type in self._env_data['subjects']:
                self._subject[name] = obj_type()
                self._subject[name].load(
                    path=os.path.join(path, filename + '.data'), filename=name)

            del self._env_data

        else:
            for obj in object_name:
                if obj in self._agent:
                    self._agent[obj].load(
                        path=os.path.join(path, filename + '.data'), filename=obj)
                    self._agent[obj].reset()
                elif obj in self._subject:
                    self._subject[obj].load(
                        path=os.path.join(path, filename + '.data'), filename=obj)
                    self._subject[obj].reset()

    def save(self, **kwargs):
        '''
        Save an object or the environment to a file.
        Arguments
        ---------
            filename: the name of the file to be saved.
            path: the path of the file to be saved. (Default='./')
            object_name: if specified, that object (agent or subject) is being saved to file. 'all' saves the environment. (Default = 'all')
        Raises ValueError if the filename is not specified.
        '''
        object_name = kwargs.get('object_name', 'all')
        filename = kwargs.get('filename', self._name)
        path = kwargs.get('path', self._path)

        if object_name == 'all':
            self._env_data = {'agents': [], 'subjects': []}

            for name, agent in self._agent.items():
                _, fn = agent.save(path=os.path.join(path, filename + '.data'), filename=name)
                self._env_data['agents'].append((fn, type(agent)))

            for name, subject in self._subject.items():
                _, fn = subject.save(
                    path=os.path.join(path, filename + '.data'), filename=name)
                self._env_data['subjects'].append((fn, type(subject)))

            RLBase.save(self, filename=filename, path=path,
                        data=['_env_data', '_episodes', '_max_steps', '_termination', '_reset', '_learning_method', '_assignment_list',
                        '_total_experienced_episodes'])

            del self._env_data
        else:
            for obj in object_name:
                if obj in self._agent:
                    self._agent[obj].save(
                        path=os.path.join(path, filename + '.data'), filename=obj)
                elif obj in self._subject:
                    self._subject[obj].save(
                        path=os.path.join(path, filename + '.data'), filename=obj)

    def __repr__(self):
        try:
            return 'Env: \n Agents:\n' + \
                '\n\t'.join((a.__repr__() for a in self._agent.values())) + \
                '\nSubjects:\n' + \
                '\n\t'.join((s.__repr__() for s in self._subject.values()))
        except AttributeError:
            return 'Environment: New'
