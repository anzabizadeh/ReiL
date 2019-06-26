# -*- coding: utf-8 -*-
'''
environment class
=================

This `environment` class provides a learning environment for any reinforcement learning agent on any subject. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from pickle import load, dump, HIGHEST_PROTOCOL
import signal
import sys
import inspect
import pathlib
import pandas as pd

from ..base import RLBase
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
        RLBase.__init__(self, **kwargs)
        if 'filename' in kwargs:
            self.load(path=kwargs.get('path', '.'),
                      filename=kwargs['filename'])
            return

        RLBase.set_defaults(self, agent={}, subject={}, assignment_list={},
                            episodes=1, max_steps=10000, termination='any', reset='all',
                            learning_method='every step',
                            allow_user_to_halt=True, save_on_exit=True)
        RLBase.set_params(self, **kwargs)

        # signal.signal(signal.SIGINT, self.__signal_handler)

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._agent, self._subject, self._assignment_list = {}, {}, {}
            self._episodes, self._max_steps = 1, 10000
            self._termination, self._reset, self._learning_method = 'any', 'all', 'every step'

    def add(self, **kwargs):
        '''
        Add agents to the environment.

        Arguments
        ---------
            agents: a dictionary consist of agent name and agent object. Names should be unique, otherwise overwritten.
            subjects: a dictionary consist of subject name and subject object. Names should be unique, otherwise overwritten.
        '''
        try:
            for name, agent in kwargs['agents'].items():
                self._agent[name] = agent
        except IndexError:
            pass

        try:
            for name, subject in kwargs['subjects'].items():
                self._subject[name] = subject
        except IndexError:
            pass

    def remove(self, **kwargs):
        '''
        Remove agents from the environment.

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
                raise KeyError('Agent '+name+' not found')

        for name in kwargs['subjects']:
            try:
                del self._subject[name]
            except KeyError:
                raise KeyError('Subject '+name+' not found')

    def assign(self, agent_subject_names):
        '''
        Assign agents to subjects.

        Arguments
        ---------
            agent_subject_names: a list of agent subject tuples.

        Raises ValueError if an agent or subject is not found.
        Note: An agent can be assigned to act on multiple subjects and a subject can be affected by multiple agents. 
        '''
        for agent_name, subject_name in agent_subject_names:
            if agent_name not in self._agent:
                raise ValueError('Agent ' + agent_name + ' not found.')
            if subject_name not in self._subject:
                raise ValueError('Subject ' + subject_name + ' not found.')
            _id = self._subject[subject_name].register(agent_name)
            self._assignment_list[agent_name] = (subject_name, _id)

    def elapse(self, **kwargs):
        '''
        Move forward in time for a number of episodes.

        At each episode, agents are called sequentially to act on their respective subject(s).
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
        termination = kwargs.get('termination', self._termination).lower()
        reset = kwargs.get('reset', self._reset)
        learning_method = kwargs.get(
            'learning_method', self._learning_method).lower()
        reporting = kwargs.get('reporting', 'none').lower()

        tally = kwargs.get('tally', 'no').lower() == 'yes'
        if tally:
            win_count = {}
            for agent in self._agent:
                win_count[agent] = 0

        step_count = kwargs.get('step_count', 'no').lower() == 'yes'

        if learning_method == 'none':
            for agent in self._agent.values():
                agent.status = 'testing'
        else:
            for agent in self._agent.values():
                agent.status = 'training'

        steps = 0
        for episode in range(episodes):
            if reporting != 'none':
                report_string = 'episode: {}'.format(episode+1)
            # if learning_method == 'history':
            history = {}
            for agent_name in self._agent:
                history[agent_name] = pd.DataFrame(columns=['state', 'action', 'reward'])
                # history[agent_name] = pd.DataFrame(columns=['state', 'action', 'reward'])
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
                                               printable=subject.printable())
                            if reporting == 'all':
                                print('step: {: 4} episode: {:2} state: {} action: {} by:{}'
                                      .format(steps, episode, state, action, agent_name))
                            reward = subject.take_effect(_id, action)

                            history[agent_name].loc[len(history[agent_name].index)] = [state, action, reward]

                            # history[agent_name].append(state)
                            # history[agent_name].append(action)
                            # history[agent_name].append(reward)

                            if subject.is_terminated:
                                if tally & (reward > 0):
                                    win_count[agent_name] += 1
                                for affected_agent in self._agent.keys():
                                    if (self._assignment_list[affected_agent][0] == subject_name) & \
                                            (affected_agent != agent_name):
                                        history[affected_agent][-1] = -reward

                    if termination == 'all':
                        done = True
                        for sub in self._subject.values():
                            done = done & sub.is_terminated
                    elif termination == 'any':
                        done = False
                        for sub in self._subject.values():
                            done = done | sub.is_terminated

                if learning_method == 'every step':
                    for agent_name, agent in self._agent.items():
                        state = self._subject[self._assignment_list[agent_name][0]].state
                        # agent.learn(state=state, reward=history[agent_name][-1]['reward'])
                        # history[agent_name] = pd.DataFrame(columns=['state', 'action', 'reward'])
                        agent.learn(state=state, reward=history[agent_name][2])
                        history[agent_name] = []

            if learning_method == 'history':
                for agent_name, agent in self._agent.items():
                    agent.learn(history=history[agent_name])

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
                report_string += '\n tally:'
                for agent in self._agent:
                    report_string += '\n {} {}'.format(agent, win_count[agent])

            if reporting != 'none':
                print(report_string)

        if tally:
            return win_count
        if step_count:
            return steps/episodes

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
        termination = kwargs.get('termination', self._termination).lower()

        for agent in self._agent.values():
            agent.status = 'testing'

        history = {}
        for agent_name in self._agent:
            history[agent_name] = pd.DataFrame(columns=['state', 'action', 'q', 'reward'])
        done = False
        steps = 0
        while not done:
            if steps >= max_steps:
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
                                           printable=subject.printable())
                        q = agent._q(state, action)
                        reward = subject.take_effect(_id, action)
                        # history[agent_name].append(state)
                        # history[agent_name].append(action)
                        # history[agent_name].append(q)
                        # history[agent_name].append(reward)

                        history[agent_name].loc[len(history[agent_name].index)] = [state, action, q, reward]
                        if subject.is_terminated:
                            for affected_agent in self._agent.keys():
                                if (self._assignment_list[affected_agent][0] == subject_name) & \
                                        (affected_agent != agent_name):
                                    history[affected_agent][-1] = -reward

                if termination == 'all':
                    done = True
                    for sub in self._subject.values():
                        done = done & sub.is_terminated
                elif termination == 'any':
                    done = False
                    for sub in self._subject.values():
                        done = done | sub.is_terminated

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
                    path=path+'/'+filename+'.data', filename=name)

            for name, obj_type in self._env_data['subjects']:
                self._subject[name] = obj_type()
                self._subject[name].load(
                    path=path+'/'+filename+'.data', filename=name)

            del self._env_data

        else:
            for obj in object_name:
                if obj in self._agent:
                    self._agent[obj].load(
                        path=path+'/'+filename+'.data', filename=obj)
                    self._agent[obj].reset()
                elif obj in self._subject:
                    self._subject[obj].load(
                        path=path+'/'+filename+'.data', filename=obj)
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
                _, fn = agent.save(path=path+'/'+filename +
                                   '.data', filename=name)
                self._env_data['agents'].append((fn, type(agent)))

            for name, subject in self._subject.items():
                _, fn = subject.save(
                    path=path+'/'+filename+'.data', filename=name)
                self._env_data['subjects'].append((fn, type(subject)))

            RLBase.save(self, filename=filename, path=path,
                        data=['_env_data', '_episodes', '_max_steps', '_termination', '_reset', '_learning_method', '_assignment_list'])
            del self._env_data
        else:
            for obj in object_name:
                if obj in self._agent:
                    self._agent[obj].save(
                        path=path+'/'+filename+'.data', filename=obj)
                elif obj in self._subject:
                    self._subject[obj].save(
                        path=path+'/'+filename+'.data', filename=obj)

    def __repr__(self):
        try:
            return 'Env: \n Agents:\n' + \
                '\n\t'.join((a.__repr__() for a in self._agent.values())) + \
                '\nSubjects:\n' + \
                '\n\t'.join((s.__repr__() for s in self._subject.values()))
        except AttributeError:
            return 'Environment: New'
