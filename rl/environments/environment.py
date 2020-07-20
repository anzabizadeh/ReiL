# -*- coding: utf-8 -*-
'''
environment class
=================

This `environment` class provides a learning environment for any reinforcement learning agent on any subject.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import functools
import pathlib
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from rl import agents as rlagents
from rl import rlbase
from rl import subjects as rlsubjects

AgentSubjectTuple = Tuple[str, str]


class Environment(rlbase.RLBase):
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

    def __init__(self,
                 filename: Optional[str] = None,
                 path: Optional[str] = None,
                 agents: Optional[Dict[str, rlagents.Agent]] = None,
                 subjects: Optional[Dict[str, rlsubjects.Subject]] = None,
                 assignment_list: Optional[Dict[AgentSubjectTuple, int]] = None,
                 episodes: int = 1,
                 max_steps: int = 10000,
                 termination: str = 'any',
                 reset: str = 'any',
                 learning_batch_size: int = -1,
                 learning_method: str = 'every step',
                 **kwargs):
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

        if filename is not None:
            self.load(filename=filename, path=path)
            return

        super().__init__(name=kwargs.get('name', __name__),
                         logger_name=kwargs.get('logger_name', __name__),
                         **kwargs)

        self._agents = agents if agents is not None else {}
        self._subjects = subjects if subjects is not None else {}
        self._assignment_list = assignment_list if assignment_list is not None else {}

        self._episodes = episodes
        self._total_experienced_episodes = {}
        self._max_steps = max_steps
        self._termination = termination
        self._reset = reset
        self._learning_batch_size = learning_batch_size
        self._learning_method = learning_method

    def add(self,
            agents: Optional[Dict[str, rlagents.Agent]] = None,
            subjects: Optional[Dict[str, rlsubjects.Subject]] = None) -> None:
        '''
        Add agents or subjects to the environment.

        Arguments
        ---------
            agents: a dictionary consist of agent name and agent object. Names should be unique, otherwise overwritten.
            subjects: a dictionary consist of subject name and subject object. Names should be unique, otherwise overwritten.
        '''
        if agents is not None:
            self._agents.update(agents)

        if subjects is not None:
            self._subjects.update(subjects)
        # try:
        #     for name, agent in kwargs['agents'].items():
        #         self._agent[name] = agent
        # except (IndexError, KeyError):
        #     pass

        # try:
        #     for name, subject in kwargs['subjects'].items():
        #         self._subject[name] = subject
        # except (IndexError, KeyError):
        #     pass

    def remove(self,
               agents: Optional[Dict[str, rlagents.Agent]] = None,
               subjects: Optional[Dict[str, rlsubjects.Subject]] = None) -> None:
        '''
        Remove agents or subjects from the environment.

        Arguments
        ---------
            agents: a list of agent names to be deleted.
            subjects: a list of subject names to be deleted.

        Raises KeyError if the agent is not found.
        '''
        if agents is not None:
            for name in agents:
                try:
                    del self._agent[name]
                except KeyError:
                    raise KeyError(f'Agent {name} not found!')

        if subjects is not None:
            for name in subjects:
                try:
                    del self._subject[name]
                except KeyError:
                    raise KeyError(f'Subject {name} not found!')

    def assign(self, agent_subject_names: List[AgentSubjectTuple]) -> None:
        '''
        Assign agents to subjects.

        Arguments
        ---------
            agent_subject_names: a list of agent subject tuples.

        Raises ValueError if an agent or subject is not found.
        Note: An agent cannot be assigned to act on multiple subjects, but a subject can be affected by multiple agents.
        '''
        for agent_name, subject_name in agent_subject_names:
            if agent_name not in self._agents:
                raise ValueError(f'Agent {agent_name} not found!')
            if subject_name not in self._subjects:
                raise ValueError(f'Subject {subject_name} not found!')
            _id = self._subjects[subject_name].register(agent_name)

            self._total_experienced_episodes[(agent_name, subject_name)] = 0
            self._assignment_list[(agent_name, subject_name)] = _id
            # try:
            #     self._assignment_list[agent_name].append((subject_name, _id))
            # except KeyError:
            #     self._assignment_list[agent_name] = [(subject_name, _id)]

    def divest(self, agent_subject_names: List[AgentSubjectTuple]) -> None:
        '''
        Divest agent subject assignment.

        Arguments
        ---------
            agent_subject_names: a list of agent subject tuples.

        Raises ValueError if an agent or subject is not found.
        Note: An agent can be assigned to act on multiple subjects and a subject can be affected by multiple agents.
        '''
        for agent_name, subject_name in agent_subject_names:
            if agent_name not in self._agents:
                raise ValueError(f'Agent {agent_name} not found!')
            if subject_name not in self._subjects:
                raise ValueError(f'Subject {subject_name} not found!')

            self._subjects[subject_name].deregister(agent_name)
            self._assignment_list.pop((agent_name, subject_name))
            del self._total_experienced_episodes[(agent_name, subject_name)]

    def elapse(self,
               episodes: Optional[int] = None,
               max_steps: Optional[int] = None,
               termination: Optional[str] = None,
               reset: Optional[str] = None,
               learning_method: Optional[str] = None,
               reporting: Optional[str] = None,
               tally: bool = False,
               step_count: bool = False):
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
        def get_argument(x, y): return x if x is not None else y
        _episodes = get_argument(episodes, self._episodes)
        _max_steps = get_argument(max_steps, self._max_steps)
        if get_argument(termination, self._termination).lower() == 'all':
            def termination_func(x, y): return x & y.is_terminated
            list_of_subjects = [True] + list(self._subjects.values())
        else:
            def termination_func(x, y): return x | y.is_terminated
            list_of_subjects = [False] + list(self._subjects.values())

        _reset = get_argument(reset, self._reset).lower()
        _learning_method = get_argument(
            learning_method, self._learning_method).lower()
        _reporting = get_argument(reporting, 'none').lower()

        _tally = get_argument(tally, 'no').lower() == 'yes'
        win_count = dict((agent, 0) for agent in self._agents)

        _step_count = get_argument(step_count, 'no').lower() == 'yes'

        if _learning_method == 'none':
            for agent in self._agents.values():
                agent.training_mode = False
        else:
            for agent in self._agents.values():
                agent.training_mode = True

        report_string = ''
        steps = 0
        for episode in range(_episodes):
            if _reporting != 'none':
                report_string = f'episode: {episode + 1}'
            # history = dict((agent_name, []) for agent_name in self._agents)
            history = defaultdict(list)
            done = False
            stopping_criterion = _max_steps * (episode + 1)
            while not done:
                if steps >= stopping_criterion:
                    break
                steps += 1
                for agent_name, agent in self._agents.items():
                    if not done:
                        subject_name, _id = self._assignment_list[agent_name]
                        subject = self._subjects[subject_name]
                        if not subject.is_terminated:
                            state = subject.state
                            possible_actions = subject.possible_actions
                            action = agent.act(state, actions=possible_actions,
                                               episode=self._total_experienced_episodes[(agent_name, subject_name)])
                            reward = subject.take_effect(action, _id)

                            if _reporting == 'all':
                                print(
                                    f'step: {steps: 4} episode: {episode:2} state: {state} action: {action} by:{agent_name}')

                            history[agent_name].append(
                                {'state': state, 'action': action, 'reward': action})

                            if subject.is_terminated:
                                win_count[agent_name] += int(reward > 0)
                                for affected_agent in self._agents:
                                    if (self._assignment_list[affected_agent][0] == subject_name) & \
                                            (affected_agent != agent_name):
                                        history[affected_agent][-1]['reward'] = -reward

                    done = functools.reduce(termination_func, list_of_subjects)

                if _learning_method == 'every step':
                    for agent_name, agent in self._agents.items():
                        agent.learn(state=self._subjects[self._assignment_list[agent_name][0]].state,
                                    reward=history[agent_name][-1]['reward'])
                        history[agent_name] = []

            if _learning_method == 'history':
                for agent_name, agent in self._agents.items():
                    agent.learn(history=history[agent_name])

            for agent_name, subject_name in self._total_experienced_episodes:
                if self._subjects[subject_name].is_terminated:
                    self._total_experienced_episodes[(
                        agent_name, subject_name)] += 1

            if _reset == 'all':
                for agent in self._agents.values():
                    agent.reset()
                for subject in self._subjects.values():
                    subject.reset()
            elif _reset == 'any':
                for agent_name, subject_name in self._assignment_list.items():
                    if self._subjects[subject_name[0]].is_terminated:
                        self._agents[agent_name].reset()
                        self._subjects[subject_name[0]].reset()

            if _tally & (_reporting != 'none'):
                report_string += f'\n tally:'
                for agent in self._agents:
                    report_string += f'\n {agent} {win_count[agent]}'

            if _reporting != 'none':
                print(report_string)

        if _tally:
            return win_count
        if _step_count:
            return steps/_episodes

    def elapse_iterable(self,
                        max_steps: Optional[int] = None,
                        training_mode: Optional[Dict[AgentSubjectTuple, bool]] = None,
                        reset: Optional[str] = None,
                        return_output: Optional[Dict[AgentSubjectTuple, bool]] = None,
                        stats: Optional[Dict[AgentSubjectTuple, Sequence]] = None,
                        stats_func: Optional[Dict[AgentSubjectTuple,
                                                  Callable]] = None
                        ) -> Tuple[Dict[str, list], Dict[str, list]]:
        '''
        Move forward in time until IterableSubject is consumed.

        For each IterableSubject, agents are called sequentially to act based on the assignment list.
        Method ends if all IterableSubjects are consumed.
        Arguments
        ---------
            max_steps: maximum number of steps in each episode (Default = 10,000)
            training_mode: whether it is in training or test mode. (Default: True)
            reset: whether to reset `agents`. 'any` resets agents who acted on a finished subject. `all` resets all agents. (Default = 'any')
            return_output: a dictionary that indicates whether to return the resulting outputs or not (Default: False)
            stats: a dictionary that contains stats for each agent-subject pair
            stats_func: a dictionary that contains functions to calculate stats.
        '''
        def get_argument(x, y): return x if x is not None else y

        _max_steps = get_argument(max_steps, self._max_steps)
        _training_mode = get_argument(training_mode, defaultdict(lambda: True))
        _reset = get_argument(reset, self._reset).lower()

        _return_output = get_argument(return_output, defaultdict(bool))
        _stats = get_argument(stats, defaultdict(list))

        # TODO: stats_func should be functions, not empty list!
        _stats_func = get_argument(stats_func, defaultdict(list))

        output = defaultdict(list)
        stats_agents = defaultdict(list)
        stats_subjects = defaultdict(list)
        stats_final = defaultdict(list)

        for subject_name, subject in self._subject.items():
            assigned_agents = list(
                (k[0], v) for k, v in self._assignment_list.items() if k[1] == subject_name)
            if assigned_agents == []:
                continue

            for agent_name, _ in assigned_agents:
                self._agent[agent_name].training_mode = training_mode[(
                    agent_name, subject_name)]

            history = defaultdict(list)
            for instance_id, subject_instance in subject:
                steps = 0
                # for agent_name, _ in assigned_agents:
                #     self._agent[agent_name].exchange_protocol = subject_instance.requested_exchange_protocol
                while not subject_instance.is_terminated:
                    for agent_name, _id in assigned_agents:
                        agent = self._agent[agent_name]
                        if subject_instance.is_terminated or steps >= _max_steps:
                            break
                        steps += 1

                        try:
                            complete_state = subject_instance.complete_state
                        except NotImplementedError:
                            complete_state = None

                        state = subject_instance.state
                        possible_actions = subject_instance.possible_actions
                        action = agent.act(state, actions=possible_actions,
                                           episode=self._total_experienced_episodes[(agent_name, subject_name)])
                        reward = subject_instance.take_effect(action, _id)

                        history[agent_name].append({'instance_id': instance_id,
                                                    'state': state,
                                                    'action': action,
                                                    'reward': reward,
                                                    'complete_state': complete_state})

                        if subject_instance.is_terminated:
                            for affected_agent in self._agent:
                                if (affected_agent, subject_name) in self._assignment_list and \
                                        (affected_agent != agent_name):
                                    history[affected_agent][-1]['reward'] = -reward

                        if training_mode[(agent_name, subject_name)] \
                                and self._learning_batch_size != -1 \
                                and len(history[agent_name]) >= self._learning_batch_size:
                            agent.learn(history=history[agent_name])
                            history[agent_name] = []

                for agent_name, _ in assigned_agents:
                    agent_subject_tuple = (agent_name, subject_name)
                    if training_mode[agent_subject_tuple]:
                        self._total_experienced_episodes[agent_subject_tuple] += 1

                        if self._learning_batch_size == -1:
                            self._agent[agent_name].learn(
                                history=history[agent_name])

                    try:
                        stats_list = stats[agent_subject_tuple]
                        result_agent = self._agent[agent_name].stats(
                            stats_list)
                        result_subject = subject_instance.stats(stats_list)

                        if result_agent != {}:
                            stats_agents[agent_subject_tuple].append(
                                result_agent)

                        if result_subject != {}:
                            stats_subjects[agent_subject_tuple].append(
                                result_subject)
                    except KeyError:
                        pass

            if reset == 'all':
                for agent in self._agent.values():
                    agent.reset()
            elif reset == 'any':
                for agent_name, _ in assigned_agents:
                    self._agent[agent_name].reset()

            for agent_name, _ in assigned_agents:
                agent_subject_tuple = (agent_name, subject_name)
                if return_output[agent_subject_tuple]:
                    output[agent_subject_tuple].append(history[agent_name])

                # TODO: check if this condition is necessary!
                if stats.get(agent_subject_tuple, []) != []:
                    result = stats_func[agent_subject_tuple](agent_stats=stats_agents.get(agent_subject_tuple, None),
                                                             subject_stats=stats_subjects.get(agent_subject_tuple, None))
                    stats_final[agent_subject_tuple].append(result)

        return stats_final, output

    # TODO: Make arguments explicit + type annotation
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
            def termination_func(x, y): return x & y.is_terminated
            list_of_subjects = [True] + list(self._subject.values())
        else:
            def termination_func(x, y): return x | y.is_terminated
            list_of_subjects = [False] + list(self._subject.values())

        for agent in self._agent.values():
            agent.training_mode = False

        for subject in self._subject.values():
            subject.reset()

        history = dict((agent_subject, [])
                       for agent_subject in self._assignment_list)

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

                        history[(agent_name, subject_name)].append(
                            {'state': state, 'action': action, 'q': q, 'reward': reward})
                        if subject.is_terminated:
                            for affected_agent in self._agent.keys():
                                if ((affected_agent, subject) in self._assignment_list.keys()) & \
                                        (affected_agent != agent_name):
                                    history[(affected_agent, subject)
                                            ][-1]['reward'] = -reward

                done = functools.reduce(termination_func, list_of_subjects)

        for sub in self._subject.values():
            sub.reset()

        return history

    def load(self,
             object_name: Optional[List[str]] = 'all',
             filename: Optional[str] = None,
             path: Optional[str] = None) -> None:
        '''
        Load an object or an environment from a file.
        Arguments
        ---------
            filename: the name of the file to be loaded.
            object_name: if specified, that object (agent or subject) is being loaded from file. 'all' loads an environment. (Default = 'all')
        Raises ValueError if the filename is not specified.
        '''
        _filename = filename if filename is not None else self._name
        _path = pathlib.Path(path if path is not None else self._path)

        if object_name == 'all':
            super().load(filename=_filename, path=_path)
            self._agent = {}
            self._subject = {}
            for name, obj_type in self._env_data['agents']:
                self._agent[name] = obj_type()
                self._agent[name].load(
                    path=(_path / f'{_filename}.data'), filename=name)
            for name, obj_type in self._env_data['subjects']:
                self._subject[name] = obj_type()
                self._subject[name].load(
                    path=(_path / f'{_filename}.data'), filename=name)

            del self._env_data

        else:
            for obj in object_name:
                if obj in self._agent:
                    self._agent[obj].load(
                        path=(_path / f'{_filename}.data'), filename=obj)
                    self._agent[obj].reset()
                elif obj in self._subject:
                    self._subject[obj].load(
                        path=(_path / f'{_filename}.data'), filename=obj)
                    self._subject[obj].reset()

    def save(self,
             object_name: Optional[List[str]] = 'all',
             filename: Optional[str] = None,
             path: Optional[str] = None) -> Tuple[pathlib.Path, str]:
        '''
        Save an object or the environment to a file.
        Arguments
        ---------
            filename: the name of the file to be saved.
            path: the path of the file to be saved. (Default='./')
            object_name: if specified, that object (agent or subject) is being saved to file. 'all' saves the environment. (Default = 'all')
        Raises ValueError if the filename is not specified.
        '''
        _filename = filename if filename is not None else self._name
        _path = pathlib.Path(path if path is not None else self._path)

        if object_name == 'all':
            self._env_data = defaultdict(list)

            for name, agent in self._agent.items():
                _, fn = agent.save(
                    path=_path / f'{_filename}.data', filename=name)
                self._env_data['agents'].append((fn, type(agent)))

            for name, subject in self._subject.items():
                _, fn = subject.save(
                    path=_path / f'{_filename}.data', filename=name)
                self._env_data['subjects'].append((fn, type(subject)))

            super().save(filename=_filename, path=_path,
                         data_to_save=['_env_data', '_episodes', '_max_steps',
                                       '_termination', '_reset',
                                       '_learning_method', '_assignment_list',
                                       '_total_experienced_episodes'])

            del self._env_data
        else:
            for obj in object_name:
                if obj in self._agent:
                    self._agent[obj].save(
                        path=_path / f'{_filename}.data', filename=obj)
                elif obj in self._subject:
                    self._subject[obj].save(
                        path=_path / f'{_filename}.data', filename=obj)

        return _path, _filename

    def __repr__(self) -> str:
        try:
            return 'Env: \n Agents:\n' + \
                '\n\t'.join((a.__repr__() for a in self._agent.values())) + \
                '\nSubjects:\n' + \
                '\n\t'.join((s.__repr__() for s in self._subject.values()))
        except AttributeError:
            return 'Environment: New'
