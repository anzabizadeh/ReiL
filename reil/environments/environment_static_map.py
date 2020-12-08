# -*- coding: utf-8 -*-
'''
environment class
=================

This `environment` class provides a learning environment for any reinforcement learning agent on any subject.


'''
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from reil import agents as rlagents
from reil import stateful
from reil import subjects as rlsubjects
from reil.datatypes import InteractionProtocol, ReilData
from reil.utils import functions
from reil.utils import instance_generator as rlgenerator

AgentSubjectTuple = Tuple[str, str]
Entity = Union[rlagents.Agent, rlsubjects.Subject]
EntityGenerator = Union[rlgenerator.InstanceGenerator[rlagents.Agent],
                        rlgenerator.InstanceGenerator[rlsubjects.Subject]]


class EnvironmentStaticMap(stateful.Stateful):
    '''
    Provide an interaction and learning environment for `agent`s and `subject`s,
    based on a static interaction map.

    Attributes
-----------
    interaction_sequence: a sequence of `InteractionProtocols` that should be
    followed during each pass of simulation.

    Methods
-----------
    add: add a set of entities (agents/ subjects) to the environment.

    append_observations: appends a new list of observations to history.

    assert_protocol: raises an error if a protocol does not follow the rules.

    interact_n_times: simulates "n" interactions of an `agent` and a `subject`.

    interact_once: simulates one interaction of an `agent` and a `subject`.

    interact_while: simulates interactions of an `agent` and a `subject` until
    the subject terminates.

    load: load an entity (agent/ subject) or an environment.

    register_agents: registers agents in subjects befor each interaction.

    remove: remove entities (agents/ subjects) from the environment.

    save: save an entity (agent/ subject) or the current environment.

    simulate_one_pass: Passes over interaction_sequence once and simulates
    interactions.

    simulate_passes: Passes over interaction_sequence multiple times and simulates
    interactions.

    simulate_to_termination: Passes over interaction_sequence and simulates
    interactions until all subject generators terminate.

    _collect_terminal_rewards: collect the rewards after a subject is terminated.

    _calculate_statistics: calculate statistics after a subject is terminated.

    Note: Agents act on subjects and receive the reward of their action and
    the new state of subjects. Then agents learn based on this information to
    improve their actions.

    `Pass`: visiting all protocols in the interaction sequence, once.

    `Epoch`: For each subject, an epoch is one time reaching its terminal state.
    If the subject is an instance generator, then the generator should reach to
    terminal state, not just its current instance.
    '''
    # TODO: Statsitics aggregators are not supported yet!
    # TODO: simulate_passes and simulate_one_pass do not return anything!

    def __init__(self,
        entity_dict: Optional[Dict[str, Union[Entity, EntityGenerator, str]]] = None,
        interaction_sequence: Optional[Tuple[InteractionProtocol, ...]] = None,
        **kwargs: Any):
        '''
        Create a new environment.

        Arguments
-----------
        entity_dict: a dictionary that contains `agents`, `subjects`, and
        `generators` for the environment.

        interaction_sequence: a tuple of `InteractionProtocol`s that specify
        how entities interact in the simulation.
        '''
        super().__init__(name=kwargs.get('name', __name__),
                         logger_name=kwargs.get('logger_name', __name__),
                         **kwargs)

        self._agents: Dict[str, rlagents.AgentType] = {}
        self._subjects: Dict[str, rlsubjects.SubjectType] = {}
        self._instance_generators: Dict[str, EntityGenerator] = {}
        self._interaction_sequence: Tuple[InteractionProtocol, ...] = ()
        self._assignment_list: Dict[AgentSubjectTuple, Union[int, None]] = defaultdict(lambda:None)
        self._epochs: Dict[str, int] = defaultdict(int)
        self._history: Dict[AgentSubjectTuple, stateful.History] = defaultdict(
            lambda: [stateful.Observation()])
        self._agent_statistics: Dict[AgentSubjectTuple, List[ReilData]] = defaultdict(list)
        self._subject_statistics: Dict[AgentSubjectTuple, List[ReilData]] = defaultdict(list)

        if entity_dict is not None:
            self.add(entity_dict)
        if interaction_sequence is not None:
            self.interaction_sequence = interaction_sequence

    def add(self, entity_dict: Dict[str, Union[Entity, EntityGenerator, str]]) -> None:
        '''
        Adds agents and subjects to the environment.

        Arguments
-----------
        entity_dict: a dictionary consist of agent/ subject name and the
        respective entity. Names should be unique, otherwise overwritten.
        To assign the same entity to different names, one can use the name in the
        first assignment as the value of the dict for other assignments. For
        example:
        >>> env.add({'agent_1': Agent(), 'agent_2': 'agent_1'})

        When using name as value, the name is being looked up first in
        instance generators, then agents, and finally subjects. Whichever
        contains the name first, the entity corresponding to that instance is
        being used. 

        Note: `InstanceGenerator` reused might produce unintended consequences.
        '''
        for name, obj in entity_dict.items():
            if isinstance(obj, str):
                _obj = self._instance_generators.get(
                    obj, self._agents.get(
                        obj, self._subjects.get(obj)))
                if _obj is None:
                    raise ValueError(f'entity {obj} defined for {name} is '
                                     'not in the list of agents, subjects, '
                                     'and generators.')
            else:
                _obj = obj
            if isinstance(_obj, rlgenerator.InstanceGenerator):
                self._instance_generators.update({name: _obj})
            elif isinstance(_obj, rlagents.Agent):
                self._agents.update({name: _obj})
            elif isinstance(_obj, rlsubjects.Subject):
                self._subjects.update({name: _obj})
            else:
                raise TypeError(f'entity {name} is niether an agent nor a subject.')

        for name, generator in self._instance_generators.items():
            _, obj = next(generator)
            if isinstance(obj, rlagents.Agent):
                self._agents.update({name: obj})
            elif isinstance(obj, rlsubjects.Subject):
                self._subjects.update({name: obj})
            else:
                raise TypeError(
                    f'entity {name} is niether an agent nor a subject.')

    def remove(self, entity_names: Tuple[str, ...]) -> None:
        '''
        Removes agents, subjects, or instance generators from the environment.

        Arguments
-----------
        entity_names: a tuple of agent/ subject names to be deleted.

        Note: the method removes the item from both agents and subjects lists.
        Hence, it is not recommended to use the same name for both an agent
        and a subject.
        '''
        names_in_use = [p.agent.name
                        for p in self._interaction_sequence] + \
                       [p.subject.name
                        for p in self._interaction_sequence]
        for name in entity_names:
            if name in names_in_use:
                raise ValueError(f'{name} is currently in use '
                                 'in the interaction sequence.')
            if name in self._agents:
                del self._agents[name]
            if name in self._subjects:
                del self._subjects[name]
            if name in self._instance_generators:
                del self._instance_generators[name]

    @property
    def interaction_sequence(self) -> Tuple[InteractionProtocol, ...]:
        return self._interaction_sequence

    @interaction_sequence.setter
    def interaction_sequence(self, seq: Tuple[InteractionProtocol, ...]) -> None:
        for protocol in seq:
            self.assert_protocol(protocol)

        self._interaction_sequence = seq

    def assert_protocol(self, protocol: InteractionProtocol) -> None:
        '''
        Checks whether the given protocol:

        * contains only entities that are known to the `environment`.

        * unit is one of the possible values. 
        '''
        if protocol.agent.name not in self._agents:
            raise ValueError(f'Unknown agent name: {protocol.agent.name}.')
        if protocol.subject.name not in self._subjects:
            raise ValueError(f'Unknown subject name: {protocol.subject.name}.')
        if protocol.unit not in ('interaction', 'instance', 'epoch'):
            raise ValueError(f'Unknown unit: {protocol.unit}. '
                             'It should be one of interaction, instance, or epoch. '
                             'For subjects of non-instance generator, epoch and '
                             'instance are equivalent.')
        # if (protocol.agent_name in self._instance_generators or
        #     protocol.subject_name in self._instance_generators) and

    @staticmethod
    def interact_once(agent_id: int,
                      agent_instance: rlagents.Agent,
                      subject_instance: rlsubjects.Subject,
                      protocol: InteractionProtocol,
                      epoch: int) -> Tuple[ReilData, ReilData, Union[ReilData, None]]:
        '''
        Returns subject's reward and state before taking an action and agent's action.

        Attributes
-----------
        agent_id: agent's ID by which it is registered at the subject.

        agent_instance: an instance of an agent that takes the action.

        subject_instance: an instance of a subject that computes reward, determines
        possible actions, and takes the action.

        protocol: an `InteractionProtocol` that specifies the state and reward
        function definitions.

        epoch: the epoch of of the current run. This value is used by the agent
        to determine the action.

        Note: If the subject is terminated or no possible actions are available,
        None is returned for action. 
        '''
        reward = subject_instance.reward(
            name=protocol.reward_function_name, _id=agent_id)

        state = subject_instance.state(name=protocol.state_name, _id=agent_id)
        if not subject_instance.is_terminated(agent_id):
            possible_actions = subject_instance.possible_actions(agent_id)
            if possible_actions:
                action = agent_instance.act(state, actions=possible_actions,
                                            epoch=epoch)
                subject_instance.take_effect(action, agent_id)
            else:
                action = None
        else:
            action = None

        return reward, state, action

    @staticmethod
    def interact_n_times(agent_id: int,
                         agent_instance: rlagents.Agent,
                         subject_instance: rlsubjects.Subject,
                         protocol: InteractionProtocol,
                         epoch: int,
                         times: int = 1) -> stateful.History:
        '''
        Allows `agent` and `subject` to interact at most "times" times and returns
        a list of subject's reward and state before taking an action and agent's action.

        Attributes
-----------
        agent_id: agent's ID by which it is registered at the subject.

        agent_instance: an instance of an agent that takes the action.

        subject_instance: an instance of a subject that computes reward, determines
        possible actions, and takes the action.

        protocol: an `InteractionProtocol` that specifies the state and reward
        function definitions.

        epoch: the epoch of of the current run. This value is used by the agent
        to determine the action.

        times: number of times the agent and the subject should interact.

        Note: If subject is terminated before "times" iterations, the result will
        be truncated and returned. In other words, the output will not
        necessarily have a lenght of "times".
        '''
        if subject_instance.is_terminated():
            return []

        trajectory = [stateful.Observation() for _ in range(times + 1)]
        for i in range(times):
            reward, state, action = EnvironmentStaticMap.interact_once(
                agent_id, agent_instance, subject_instance, protocol, epoch)

            trajectory[i].reward = reward
            trajectory[i+1].state = state
            trajectory[i+1].action = action

            if action is None:
                if subject_instance.is_terminated(agent_id):
                    trajectory[i+1].reward = None
                    return trajectory[:i+2]

        return trajectory

    @staticmethod
    def interact_while(agent_id: int,
                       agent_instance: rlagents.Agent,
                       subject_instance: rlsubjects.Subject,
                       protocol: InteractionProtocol,
                       epoch: int) -> stateful.History:
        '''
        Allows `agent` and `subject` to interact until subject is terminated and
        returns a list of subject's reward and state before taking an action and
        agent's action. Note that for `instance generators`, only the current
        instance is run to termination, not the whole generator.

        Attributes
-----------
        agent_id: agent's ID by which it is registered at the subject.

        agent_instance: an instance of an agent that takes the action.

        subject_instance: an instance of a subject that computes reward,
        determines possible actions, and takes the action.

        protocol: an `InteractionProtocol` that specifies the state and reward
        function definitions.

        epoch: the epoch of of the current run. This value is used by the agent
        to determine the action.
        '''
        trajectory = [stateful.Observation()]

        while not subject_instance.is_terminated(agent_id):
            reward, state, action = EnvironmentStaticMap.interact_once(
                agent_id, agent_instance, subject_instance, protocol, epoch)

            trajectory[-1].reward = reward
            trajectory.append(stateful.Observation(state=state, action=action))

        return trajectory

    def _collect_terminal_rewards(self, subject_name: str) -> None:
        '''
        When a `subject` is terminated for all interacting `agents`, this
        function is called to collect final rewards for all agents.

        Attributes
-----------
        subject_name: name of the `subject` that is terminated.
        '''
        agents_state_n_rewards = (a_s_n_r
                              for a_s_n_r in set((p.agent.name, p.state_name, p.reward_function_name)
                                               for p in self._interaction_sequence
                                               if p.subject.name == subject_name))

        for agent_name, r_func_name, state_name in agents_state_n_rewards:
            agent_id = self._assignment_list[(agent_name, subject_name)]
            reward = self._subjects[subject_name].reward(
                name=r_func_name, _id=agent_id)
            self._history[(agent_name, subject_name)][-1].reward = reward
            self._history[(agent_name, subject_name)].append(
                stateful.Observation(state=self._subjects[subject_name].state(
                    name=state_name, _id=agent_id)))

    def _calculate_statistics(self, subject_name: str) -> None:
        '''
        When a `subject` is terminated for all interacting `agents`, this
        function is called to calculate statistics for all agents and the subject.

        Attributes
-----------
        subject_name: name of the `subject` that is terminated.
        '''
        agents_and_stats = (a_n_s
                            for a_n_s in set((p.agent.name, p.agent.statistic_name, p.subject.statistic_name)
                                             for p in self._interaction_sequence
                                             if p.subject.name == subject_name))

        for agent_name, a_stat_name, s_stat_name in agents_and_stats:
            agent_id = self._assignment_list[(agent_name, subject_name)]
            self._agent_statistics[(agent_name, subject_name)].append(
                self._agents[agent_name].statistic(a_stat_name, agent_id))
            self._subject_statistics[(agent_name, subject_name)].append(
                self._subjects[subject_name].statistic(s_stat_name, agent_id))

    def _train_related_agents(self, subject_name: str) -> None:
        '''
        When a `subject` is terminated for all interacting `agents`, this
        function is called to provide history data to any related agent that can learn.

        Attributes
-----------
        subject_name: name of the `subject` that is terminated.
        '''
        affected_agents = (a_name
                            for a_name in set(p.agent.name
                                                for p in self._interaction_sequence
                                              if p.subject.name == subject_name))
        for a_name in affected_agents:
            if self._agents[a_name].training_mode:
                self._agents[a_name].learn(self._history[(a_name, subject_name)][1:])
                del self._history[(a_name, subject_name)]

    def _reset_subject(self, subject_name: str) -> None:
        '''
        When a `subject` is terminated for all interacting `agents`, this
        function is called to reset the subject. If the subject is an
        `InstanceGenerator`, a new instance is created. If reset is successful,
        `epoch` is incremented by one.

        Attributes
-----------
        subject_name: name of the `subject` that is terminated.
        '''
        if subject_name in self._instance_generators:
            # get a new instance if possible,
            # if not instance generator returns StopIteration.
            # So, increment epoch by 1, then if the generator is not terminated,
            # get a new instance.
            try:
                _, self._subjects[subject_name] = cast(
                    Tuple[int, rlsubjects.SubjectType],
                    next(self._instance_generators[subject_name]))
            except StopIteration:
                # TODO: self._aggregated
                self._epochs[subject_name] += 1
                if not self._instance_generators[subject_name].is_terminated():
                    _, self._subjects[subject_name] = cast(
                        Tuple[int, rlsubjects.SubjectType],
                        next(self._instance_generators[subject_name]))
        else:
            self._epochs[subject_name] += 1
            self._subjects[subject_name].reset()

    def manage_terminated_subjects(self) -> None:
        '''
        Goes over all `subjects`. If terminated, collects terminal rewards,
        calculates stats, trains related agents, and resets the subject.
        '''
        for s_name in set(p.subject.name for p in self._interaction_sequence):
            if self._subjects[s_name].is_terminated(None):
                self._collect_terminal_rewards(s_name)
                self._calculate_statistics(s_name)
                self._train_related_agents(s_name)
                self._reset_subject(s_name)

    def register_agents(self) -> None:
        '''
        Registers all `agents` in the interaction sequence in their corresponding
        `subjects`. When registration happens for the first time, the agents
        get any ID that subjects provide. However, in the follow up registrations,
        agents attempt to register with the same ID to have access to the same
        information.
        '''
        for p in self._interaction_sequence:
            self._assignment_list[(p.agent.name, p.subject.name)] = \
                self._subjects[p.subject.name].register(
                    agent_name=p.agent.name,
                    _id=self._assignment_list[(p.agent.name, p.subject.name)])

    def append_observations(self,
        agent_name: str, subject_name: str, observations: stateful.History) -> None:
        '''
        Appends observations to the history of the given `agent` and `subject`.

        Arguments
-----------
        agent_name: name of the agent that the observations belong to.

        subject_name: name of the subject that the observations belong to.

        observations: a `History` of agent/ subject interactions.
        '''
        if observations:
            self._history[(agent_name, subject_name)][-1].reward = observations[0].reward
            self._history[(agent_name, subject_name)].extend(observations[1:])

    def simulate_one_pass(self) -> None:
        '''
        Goes through the interaction map for one pass and simulates interactions
        accordingly.
        '''
        self.manage_terminated_subjects()
        self.register_agents()
        for protocol in self._interaction_sequence:
            agent_name = protocol.agent.name
            subject_name = protocol.subject.name
            agent_id = self._assignment_list[(agent_name, subject_name)]
            if protocol.unit == 'interaction':
                observations = self.interact_n_times(
                    agent_id=agent_id,  # type: ignore
                    agent_instance=self._agents[agent_name],
                    subject_instance=self._subjects[subject_name],
                    protocol=protocol,
                    epoch=self._epochs[subject_name],
                    times=protocol.n)
            elif protocol.unit in ['instance', 'epoch']:
                # For epoch, simulate the current instance, then in the next if
                # statement, simulate the rest of the generated instances.
                observations = self.interact_while(
                    agent_id=agent_id,  # type: ignore
                    agent_instance=self._agents[agent_name],
                    subject_instance=self._subjects[subject_name],
                    protocol=protocol,
                    epoch=self._epochs[subject_name])
            else:
                raise ValueError(f'Unknown protocol unit: {protocol.unit}.')

            self.append_observations(agent_name, subject_name, observations)

            if (protocol.unit == 'epoch'
                and subject_name in self._instance_generators):
                for _, instance in self._instance_generators[subject_name]:
                    self._subjects[subject_name] = cast(rlsubjects.Subject, instance)
                    self._assignment_list[(agent_name, subject_name)] = \
                        self._subjects[subject_name].register(
                            agent_name=agent_name,
                            _id=self._assignment_list[(agent_name, subject_name)])

                    observations = self.interact_while(
                        agent_id=agent_id,  # type: ignore
                        agent_instance=self._agents[agent_name],
                        subject_instance=self._subjects[subject_name],
                        protocol=protocol,
                        epoch=self._epochs[subject_name])

                    self.append_observations(agent_name, subject_name, observations)

    def simulate_passes(self, passes: int) -> None:
        '''
        Goes through the interaction map for a number of passes and simulates
        interactions accordingly.

        Attributes
-----------
        passes: the number of passes that simulation should go.
        '''
        for _ in range(passes):
            self.simulate_one_pass()

    def simulate_to_termination(self) -> None:
        '''
        Goes through the interaction map and simulates interactions accordingly,
        until all subjects are terminated.
        To avoid possible infinite loops caused by normal `subjects`, this method
        is only available if all subjects are generated by `instance generators`.
        Attempt to call this method will normal subjects in the interaction map
        will result in TypeError.
        '''
        subjects_in_use = set(s.subject.name
                              for s in self.interaction_sequence)
        no_generators = subjects_in_use.difference(self._instance_generators)
        if no_generators:
        # if any(s.subject.name not in self._instance_generators
        #        for s in self.interaction_sequence):
            raise TypeError('Found subject(s) in the interaction_sequence that '
                            f'are not instance generators: {no_generators}')
        infinites = [s
                     for s in subjects_in_use
                     if not self._instance_generators[s].is_finite]
        if infinites:
            raise TypeError('Found infinite instance generator(s) in the '
                            f'interaction_sequence: {infinites}')
        while not all(self._instance_generators[s].is_terminated()
                      for s in self._subjects):
            self.simulate_one_pass()

    def load(self,
             entity_name: Union[List[str], str] = 'all',
             filename: Optional[str] = None,
             path: Optional[Union[pathlib.Path, str]] = None) -> None:
        '''
        Loads an entity or an environment from a file.

        Arguments
-----------
        filename: the name of the file to be loaded.

        entity_name: if specified, that entity (agent or subject) is being
        loaded from file. 'all' loads an environment. (Default = 'all')

        Raises ValueError if the filename is not specified.
        '''
        _filename: str = functions.get_argument(filename, self._name)
        _path = pathlib.Path(path if path is not None else self._path)

        if entity_name == 'all':
            super().load(filename=_filename, path=_path)
            self._agents: Dict[str, rlagents.AgentType] = {}
            self._subjects: Dict[str, rlsubjects.SubjectType] = {}
            for name, obj_type in self._env_data['agents']:  # type: ignore
                self._agents[name] = obj_type.from_pickle(  # type: ignore
                    path=(_path / f'{_filename}.data'), filename=name)
            for name, obj_type in self._env_data['subjects']:  # type: ignore
                self._subjects[name] = obj_type.from_pickle(  # type: ignore
                    path=(_path / f'{_filename}.data'), filename=name)

            del self._env_data  # type: ignore

        else:
            for obj in entity_name:
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
        Saves an entity or the environment to a file.

        Arguments
-----------
        filename: the name of the file to be saved.

        path: the path of the file to be saved. (Default='./')

        entity_name: if specified, that entity (agent or subject) is being saved
        to file. 'all' saves the environment. (Default = 'all')

        Raises ValueError if the filename is not specified.
        '''
        _filename = functions.get_argument(filename, self._name)
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
                         data_to_save=tuple(v for v in self.__dict__
                                       if v not in ('_agents', '_subjects')))

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
            return super().__repr__() + '\n Agents:\n' + \
                '\n\t'.join((a.__repr__() for a in self._agents.values())) + \
                '\nSubjects:\n' + \
                '\n\t'.join((s.__repr__() for s in self._subjects.values()))
        except AttributeError:
            return super().__repr__()
