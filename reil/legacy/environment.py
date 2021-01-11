# -*- coding: utf-8 -*-
'''
environment class
=================

This `environment` class provides a learning environment for any reinforcement
learning agent on any subject.


'''

import functools
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from reil import agents as rlagents
from reil import stateful
from reil import subjects as rlsubjects
from reil import utils
from reil.datatypes import reildata
from reil.subjects.subject import Subject

AgentSubjectTuple = Tuple[str, str]


class Environment(stateful.Stateful):
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
                 agents: Optional[Dict[str, rlagents.Agent]] = None,
                 subjects: Optional[Dict[str, rlsubjects.Subject]] = None,
                 assignment_list: Optional[Sequence[AgentSubjectTuple]] = None,
                 episodes: int = 1,
                 max_steps: int = 10000,
                 termination: str = 'any',
                 reset: str = 'any',
                 learning_batch_size: int = -1,
                 learning_method: str = 'every step',
                 **kwargs: Any):
        '''
        Create a new environment.

        Arguments
        ---------
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
        super().__init__(name=kwargs.get('name', __name__),
                         logger_name=kwargs.get('logger_name', __name__),
                         **kwargs)

        self._agents = {}
        self._subjects = {}
        self._assignment_list = {}
        self._episodes = episodes
        self._total_experienced_episodes = {}
        self._max_steps = max_steps
        self._termination = termination
        self._reset = reset
        self._learning_batch_size = learning_batch_size
        self._learning_method = learning_method

        if agents is not None:
            self.add(agents=agents)
        if subjects is not None:
            self.add(subjects=subjects)
        if assignment_list is not None:
            self.assign(assignment_list)


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
        #         self._agents[name] = agent
        # except (IndexError, KeyError):
        #     pass

        # try:
        #     for name, subject in kwargs['subjects'].items():
        #         self._subjects[name] = subject
        # except (IndexError, KeyError):
        #     pass

    def remove(self,
               agents: Optional[List[str]] = None,
               subjects: Optional[List[str]] = None) -> None:
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
                    del self._agents[name]
                except KeyError:
                    raise KeyError(f'Agent {name} not found!')

        if subjects is not None:
            for name in subjects:
                try:
                    del self._subjects[name]
                except KeyError:
                    raise KeyError(f'Subject {name} not found!')

    def assign(self, agent_subject_names: Sequence[AgentSubjectTuple]) -> None:
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
        _episodes = episodes or self._episodes
        _max_steps = max_steps or self._max_steps
        if (termination or self._termination).lower() == 'all':
            def termination_func(x: bool, y: rlsubjects.Subject) -> bool:
                return x & y.is_terminated
            list_of_subjects: List[Subject] = list(self._subjects.values())
            reduce_initial_val = True

        else:
            def termination_func(x: bool, y: rlsubjects.Subject) -> bool:
                return x | y.is_terminated
            list_of_subjects: List[Subject] = list(self._subjects.values())
            reduce_initial_val = False

        _reset = (reset or self._reset).lower()
        _learning_method = (
            learning_method or self._learning_method).lower()
        _reporting = (reporting or 'none').lower()

        _tally = tally
        win_count: Dict[str, int] = dict((agent, 0) for agent in self._agents)

        _step_count = step_count

        # if _learning_method == 'none':
        #     for agent in self._agents.values():
        #         agent.training_mode = False
        # else:
        #     for agent in self._agents.values():
        #         agent.training_mode = True

        report_string = ''
        steps = 0
        for episode in range(_episodes):
            if _reporting != 'none':
                report_string = f'episode: {episode + 1}'
            # history = dict((agent_name, []) for agent_name in self._agents)
            history: Dict[str, List[Dict[str, reildata.ReilData]]] = defaultdict(list)
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
                                {'state': state, 'action': action, 'reward': reward})

                            if subject.is_terminated:
                                win_count[agent_name] += int(reward[0].value > 0)
                                for affected_agent in self._agents:
                                    if (self._assignment_list[affected_agent][0] == subject_name) & \
                                            (affected_agent != agent_name):
                                        history[affected_agent][-1]['reward'] = -reward

                    done = functools.reduce(termination_func, list_of_subjects, reduce_initial_val)

                if _learning_method == 'every step':
                    raise NotImplementedError
                    # for agent_name, agent in self._agents.items():
                    #     agent.learn(state=self._subjects[self._assignment_list[agent_name][0]].state,
                    #                 reward=history[agent_name][-1]['reward'])
                    #     history[agent_name] = []

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
                        test_mode: Sequence[AgentSubjectTuple] = (),
                        reset: str = 'any',
                        return_output: Sequence[AgentSubjectTuple] = (),
                        stats: Optional[Dict[AgentSubjectTuple, Sequence[str]]] = None,
                        stats_func: Optional[Dict[AgentSubjectTuple,
                                                  Callable[[Any], Any]]] = None
                        ) -> Tuple[Dict[AgentSubjectTuple, List[Any]], Dict[AgentSubjectTuple, List[Any]]]:
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

        _max_steps = max_steps or self._max_steps
        _training_mode: Dict[AgentSubjectTuple, bool] = defaultdict(lambda: True)
        for a_s_tuple in test_mode:
            _training_mode[a_s_tuple] = False

        if reset.lower() not in ('any', 'all'):
            self._logger.warning(f'reset argument should be either "any" or "all". Received {reset}. "any" is assumed.')
            _reset = 'any'
        else:
            _reset = reset.lower()

        _return_output = defaultdict(bool)
        for a_s_tuple in return_output:
            _return_output[a_s_tuple] = True

        _stats = stats or defaultdict(list)

        if stats_func is None and stats is not None:
            self._logger.exception('stats are provided, but no stats_func is provided.')
            raise ValueError('stats are provided, but no stats_func is provided.')
        temp = stats_func or lambda _: None
        if isinstance(temp, Callable):
            _stats_func = dict((agent_subject_tuple, temp) for agent_subject_tuple in self._assignment_list)
        else:
            _stats_func = temp

        output = defaultdict(list)
        stats_agents = defaultdict(list)
        stats_subjects = defaultdict(list)
        stats_final = defaultdict(list)

        for subject_name, subject in self._subjects.items():
            assigned_agents = list((k[0], v)
                for k, v in self._assignment_list.items()
                if k[1] == subject_name)

            if assigned_agents == []:
                continue

            for agent_name, _ in assigned_agents:
                self._agents[agent_name].training_mode = _training_mode[(
                    agent_name, subject_name)]

            history: Dict[str, List[Dict[str, reildata.ReilData]]] = defaultdict(list)
            for instance_id, subject_instance in subject:
                steps = 0
                # for agent_name, _ in assigned_agents:
                #     self._agents[agent_name].exchange_protocol = subject_instance.requested_exchange_protocol
                while not subject_instance.is_terminated():
                    for agent_name, _id in assigned_agents:
                        agent = self._agents[agent_name]
                        if subject_instance.is_terminated() or steps >= _max_steps:
                            break
                        steps += 1

                        complete_state = None
                        try:
                            complete_state = subject_instance.complete_state
                        except NotImplementedError:
                            pass

                        state = subject_instance.state()
                        possible_actions = subject_instance.possible_actions()
                        action = agent.act(state, actions=possible_actions,
                                           episode=self._total_experienced_episodes[(agent_name, subject_name)])
                        subject_instance.take_effect(action, _id)
                        reward = subject_instance.reward()

                        history[agent_name].append({'instance_id': instance_id,
                                                    'state': state,
                                                    'action': action,
                                                    'reward': reward,
                                                    'complete_state': complete_state})

                        if subject_instance.is_terminated():
                            for affected_agent in self._agents:
                                if (affected_agent, subject_name) in self._assignment_list and \
                                        (affected_agent != agent_name):
                                    history[affected_agent][-1]['reward'] = -reward

                            complete_state = None
                            if subject_instance.exchange_protocol['complete_state']:
                                try:
                                    complete_state = subject_instance.complete_state
                                except NotImplementedError:
                                    pass
                            history[agent_name].append({'instance_id': instance_id,
                                                        'state': subject_instance.state,
                                                        'complete_state': complete_state})

                        if _training_mode[(agent_name, subject_name)] \
                                and self._learning_batch_size != -1 \
                                and len(history[agent_name]) >= self._learning_batch_size:
                            agent.learn(history=history[agent_name])
                            history[agent_name] = []

                for agent_name, _ in assigned_agents:
                    agent_subject_tuple = (agent_name, subject_name)
                    if _training_mode[agent_subject_tuple]:
                        self._total_experienced_episodes[agent_subject_tuple] += 1

                        if self._learning_batch_size == -1:
                            self._agents[agent_name].learn(
                                history=history[agent_name])

                    if _return_output[agent_subject_tuple]:
                        output[agent_subject_tuple].append(history[agent_name])

                    history[agent_name] = []

                    try:
                        stats_list = stats[agent_subject_tuple]
                        result_agent = self._agents[agent_name].stats(
                            stats_list)
                        result_subject = subject_instance.stats(stats_list)

                        if result_agent:
                            stats_agents[agent_subject_tuple].append(
                                result_agent)

                        if result_subject:
                            stats_subjects[agent_subject_tuple].append(
                                result_subject)
                    except KeyError:
                        pass

            if _reset == 'all':
                for agent in self._agents.values():
                    agent.reset()
            elif _reset == 'any':
                for agent_name, _ in assigned_agents:
                    self._agents[agent_name].reset()

            for agent_name, _ in assigned_agents:
                agent_subject_tuple = (agent_name, subject_name)

                if _stats.get(agent_subject_tuple, []):
                    result = _stats_func[agent_subject_tuple](agent_stats=stats_agents.get(agent_subject_tuple, None),
                                                                subject_stats=stats_subjects.get(agent_subject_tuple, None))
                    stats_final[agent_subject_tuple].append(result)

        return stats_final, output

    # TODO: Make arguments explicit + type annotation
    def trajectory(self, **kwargs: Any):
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
            def termination_func(x: bool, y: rlsubjects.Subject) -> bool:
                return x & y.is_terminated
            list_of_subjects = list(self._subjects.values())
            reduce_initial_val = True
        else:
            def termination_func(x: bool, y: rlsubjects.Subject) -> bool:
                return x | y.is_terminated
            list_of_subjects = list(self._subjects.values())
            reduce_initial_val = False

        for agent in self._agents.values():
            agent.training_mode = False

        for subject in self._subjects.values():
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
                    agent = self._agents[agent_name]
                    subject = self._subjects[subject_name]
                    if not subject.is_terminated:
                        state = subject.state
                        possible_actions = subject.possible_actions
                        action = agent.act(state, actions=possible_actions)
                        q = float(agent._q(state, action))
                        reward = subject.take_effect(action, _id)

                        history[(agent_name, subject_name)].append(
                            {'state': state, 'action': action, 'q': q, 'reward': reward})
                        if subject.is_terminated:
                            for affected_agent in self._agents.keys():
                                if ((affected_agent, subject) in self._assignment_list.keys()) & \
                                        (affected_agent != agent_name):
                                    history[(affected_agent, subject)
                                            ][-1]['reward'] = -reward

                done = functools.reduce(termination_func, list_of_subjects, reduce_initial_val)

        for sub in self._subjects.values():
            sub.reset()

        return history

    def load(self,
             object_name: Union[List[str], str] = 'all',
             filename: Optional[str] = None,
             path: Optional[Union[pathlib.Path, str]] = None) -> None:
        '''
        Load an object or an environment from a file.
        Arguments
        ---------
            filename: the name of the file to be loaded.
            object_name: if specified, that object (agent or subject) is being loaded from file. 'all' loads an environment. (Default = 'all')
        Raises ValueError if the filename is not specified.
        '''
        _filename: str = filename or self._name
        _path = pathlib.Path(path if path is not None else self._path)

        if object_name == 'all':
            super().load(filename=_filename, path=_path)
            self._agents: Dict[str, rlagents.Agent] = {}
            self._subjects: Dict[str, rlsubjects.Subject] = {}
            for name, obj_type in self._env_data['agents']:  # type: ignore
                self._agents[name] = obj_type.from_pickle(  # type: ignore
                    path=(_path / f'{_filename}.data'), filename=name)
            for name, obj_type in self._env_data['subjects']:  # type: ignore
                self._subjects[name] = obj_type.from_pickle(  # type: ignore
                    path=(_path / f'{_filename}.data'), filename=name)

            del self._env_data  # type: ignore

        else:
            for obj in object_name:
                if obj in self._agents:
                    self._agents[obj].load(
                        path=(_path / f'{_filename}.data'), filename=obj)
                    self._agents[obj].reset()
                elif obj in self._subjects:
                    self._subjects[obj].load(
                        path=(_path / f'{_filename}.data'), filename=obj)
                    self._subjects[obj].reset()

    def save(self,
             filename: Optional[str] = None,
             path: Optional[Union[pathlib.Path, str]] = None,
             data_to_save: Union[List[str], str] = 'all') -> Tuple[pathlib.Path, str]:
        '''
        Save an object or the environment to a file.
        Arguments
        ---------
            filename: the name of the file to be saved.
            path: the path of the file to be saved. (Default='./')
            object_name: if specified, that object (agent or subject) is being saved to file. 'all' saves the environment. (Default = 'all')
        Raises ValueError if the filename is not specified.
        '''
        _filename = filename or self._name
        _path = pathlib.Path(path if path is not None else self._path)

        if data_to_save == 'all':
            self._env_data: Dict[str, List[Any]] = defaultdict(list)

            for name, agent in self._agents.items():
                _, fn = agent.save(
                    path=_path / f'{_filename}.data', filename=name)
                self._env_data['agents'].append((fn, type(agent)))

            for name, subject in self._subjects.items():
                _, fn = subject.save(
                    path=_path / f'{_filename}.data', filename=name)
                self._env_data['subjects'].append((fn, type(subject)))

            super().save(filename=_filename, path=_path,
                         data_to_save=[v for v in self.__dict__
                            if v not in ('_agents', '_subjects')])

            del self._env_data
        else:
            for obj in data_to_save:
                if obj in self._agents:
                    self._agents[obj].save(
                        path=_path / f'{_filename}.data', filename=obj)
                elif obj in self._subjects:
                    self._subjects[obj].save(
                        path=_path / f'{_filename}.data', filename=obj)

        return _path, _filename

    def __repr__(self) -> str:
        try:
            return 'Env: \n Agents:\n' + \
                '\n\t'.join((a.__repr__() for a in self._agents.values())) + \
                '\nSubjects:\n' + \
                '\n\t'.join((s.__repr__() for s in self._subjects.values()))
        except AttributeError:
            return 'Environment: New'
