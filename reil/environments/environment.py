# -*- coding: utf-8 -*-
'''
Environment class
==================

The base class of all learning environments in which one or more `agents` act
on one or more `subjects`.
'''
import inspect
import pathlib
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from reil.agents import Agent, NoLearnAgent
from reil import stateful
from reil.subjects import Subject
from reil.datatypes import InteractionProtocol, FeatureArray
from reil.utils import InstanceGenerator

AgentSubjectTuple = Tuple[str, str]
EntityType = Union[Agent, Subject]
EntityGenType = Union[InstanceGenerator[Agent],
                      InstanceGenerator[Subject]]


class Environment(stateful.Stateful):
    '''
    Provide an interaction and learning environment for `agents` and
    `subjects`.

    Notes
    -----
    `Agents` act on `subjects` and receive the reward of their action and
    the new state of those `subjects`. Then `agents` learn based on this
    information to improve their actions.

    `Pass`: visiting all protocols in the interaction sequence, once.

    `Epoch`: For each subject, an epoch is one time reaching its terminal
    state. If the subject is an instance generator, then the generator should
    reach to terminal state, not just its current instance.
    '''

    def __init__(self,
                 entity_dict: Optional[
                     Dict[str, Union[EntityType, EntityGenType, str]]] = None,
                 **kwargs: Any):
        '''
        Arguments
        ---------
        entity_dict:
            a dictionary that contains `agents`, `subjects`, and
            `generators`.
        '''
        super().__init__(**kwargs)

        self._agents: Dict[str, Agent] = {}
        self._subjects: Dict[str, Subject] = {}
        self._instance_generators: Dict[str, EntityGenType] = {}
        self._assignment_list: Dict[
            AgentSubjectTuple,
            Tuple[Union[int, None], Union[int, None]]] = \
            defaultdict(lambda: (None, None))
        self._epochs: Dict[str, int] = defaultdict(int)
        self._agent_observers: Dict[
            Tuple[str, str],
            Generator[Union[FeatureArray, None], Any, None]] = {}

        if entity_dict is not None:
            self.add(entity_dict)

    def add(self,
            entity_dict: Dict[str, Union[EntityType, EntityGenType, str]]
            ) -> None:
        '''
        Add `agents` and `subjects` to the environment.

        Arguments
        ---------
        entity_dict:
            a dictionary consist of `agent`/ `subject` name and the
            respective entity. Names should be unique, otherwise overwritten.
            To assign one entity to multiple names, use the name in the
            first assignment as the value of the dict for other assignments.
            For example:
            >>> env.add({'agent_1': Agent(), 'agent_2': 'agent_1'})

            When using name as value, the name is being looked up first in
            instance generators, then agents, and finally subjects. Whichever
            contains the name first, the entity corresponding to that instance
            is being used.

        Notes
        -----
        Reusing an `InstanceGenerator` produces unintended consequences.

        Raises
        ------
        ValueError
            An entity is being reused without being defined first.

        TypeError
            The entity is niether an `agent` [generator] nor a `subject`
            [generator].
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
            if isinstance(_obj, InstanceGenerator):
                self._instance_generators.update({name: _obj})
            elif isinstance(_obj, NoLearnAgent):
                self._agents.update({name: _obj})
            elif isinstance(_obj, Subject):
                self._subjects.update({name: _obj})
            else:
                raise TypeError(
                    f'entity {name} is niether an agent nor a subject.')

        for name, generator in self._instance_generators.items():
            _, obj = next(generator)  # type: ignore
            if isinstance(obj, Agent):
                self._agents.update({name: obj})
            elif isinstance(obj, Subject):
                self._subjects.update({name: obj})
            else:
                raise TypeError(
                    f'entity {name} is niether an agent nor a subject.')

    def remove(self, entity_names: Tuple[str, ...]) -> None:
        '''
        Remove `agents`, `subjects`, or `instance_generators` from
        the environment.

        Arguments
        ---------
        entity_names:
            A list of `agent`/ `subject` names to be deleted.

        Notes
        -----
        This method removes the item from both `agents` and `subjects`
        lists. Hence, it is not recommended to use the same name for both
        an `agent` and a `subject`.
        '''
        for name in entity_names:
            if name in self._agents:
                del self._agents[name]
            if name in self._subjects:
                del self._subjects[name]
            if name in self._instance_generators:
                del self._instance_generators[name]

    def simulate_one_pass(self) -> None:
        '''
        Go through the interaction sequence for one pass and
        simulate interactions accordingly.
        '''
        raise NotImplementedError

    def simulate_passes(self, passes: int) -> None:
        '''
        Go through the interaction sequence for a number of passes and
        simulate interactions accordingly.

        Arguments
        ---------
        passes:
            The number of passes that simulation should go.
        '''
        for _ in range(passes):
            self.simulate_one_pass()

    def simulate_to_termination(self) -> None:
        '''
        Go through the interaction sequence and simulate interactions
        accordingly, until all `subjects` are terminated.

        Notes
        -----
        To avoid possible infinite loops caused by normal `subjects`,
        this method is only available if all `subjects` are generated
        by `instance generators`.

        Raises
        ------
        TypeError:
            Attempt to call this method will normal subjects in the interaction
            sequence.
        '''
        raise NotImplementedError

    @staticmethod
    def interact_once(
            agent_id: int,
            agent_observer: Generator[Union[FeatureArray, None], Any, None],
            subject_instance: Subject,
            state_name: str,
            action_name: str,
            reward_function_name: str,
            epoch: int) -> None:
        '''
        Allow `agent` and `subject` to interact once.

        Attributes
        ----------
        agent_id:
            Agent's ID by which it is registered at the `subject`.

        subject_id:
            Subject's ID by which it is registered at the `agent`.

        agent_instance:
            An instance of an `agent` that takes the action.

        subject_instance:
            An instance of a `subject` that computes reward, determines
            possible actions, and takes the action.

        state_name:
            A string that specifies the state definition.

        action_name:
            A string that specifies the action definition.

        reward_function_name:
            A string that specifies the reward function definition.

        epoch:
            The epoch of of the current run. This value is used by the `agent`
            to determine the action.

        Returns
        -------
        :
            Reward received by `subject`  and state before taking an action and
            the action that `agent` took.

        Notes
        -----
        This method does not check whether the `subject` is terminated.

        If no possible actions are available, `None` will be returned for
        action.
        '''
        agent_observer.send(subject_instance.reward(
            name=reward_function_name, _id=agent_id))

        state = subject_instance.state(name=state_name, _id=agent_id)
        possible_actions = subject_instance.possible_actions(
            name=action_name, _id=agent_id)
        if possible_actions:
            action = agent_observer.send({'state': state,
                                          'actions': possible_actions,
                                          'epoch': epoch})
            subject_instance.take_effect(action, agent_id)  # type: ignore

    @classmethod
    def interact_n_times(
            cls,
            agent_id: int,
            agent_observer: Generator[Union[FeatureArray, None], Any, None],
            subject_instance: Subject,
            state_name: str,
            action_name: str,
            reward_function_name: str,
            epoch: int,
            times: int = 1) -> None:
        '''
        Allow `agent` and `subject` to interact at most `times` times.

        Attributes
        ----------
        agent_id:
            Agent's ID by which it is registered at the subject.

        subject_id:
            Subject's ID by which it is registered at the `agent`.

        agent_instance:
            An instance of an `agent` that takes the action.

        subject_instance:
            An instance of a `subject` that computes reward, determines
            possible actions, and takes the action.

        state_name:
            A string that specifies the state definition.

        action_name:
            A string that specifies the action definition.

        reward_function_name:
            A string that specifies the reward function definition.

        epoch:
            The epoch of of the current run. This value is used by the `agent`
            to determine the action.

        times:
            The number of times the `agent` and the `subject` should interact.

        Returns
        -------
        :
            A list of subject's reward and state before taking an action
            and agent's action.

        Notes
        -----
        If subject is terminated before "times" iterations, the result will
        be truncated and returned. In other words, the output will not
        necessarily have a lenght of "times".
        '''
        for _ in range(times):
            cls.interact_once(agent_id, agent_observer, subject_instance,
                              state_name, action_name, reward_function_name,
                              epoch)

    @classmethod
    def interact_while(
            cls,
            agent_id: int,
            agent_observer: Generator[Union[FeatureArray, None], Any, None],
            subject_instance: Subject,
            state_name: str,
            action_name: str,
            reward_function_name: str,
            epoch: int) -> None:
        '''
        Allow `agent` and `subject` to interact until `subject` is terminated.

        Attributes
        ----------
        agent_id:
            Agent's ID by which it is registered at the subject.

        agent_instance:
            An instance of an agent that takes the action.

        subject_instance:
            An instance of a subject that computes reward,
            determines possible actions, and takes the action.

        state_name:
            A string that specifies the state definition.

        action_name:
            A string that specifies the action definition.

        reward_function_name:
            A string that specifies the reward function definition.

        epoch:
            The epoch of of the current run. This value is used by the agent
            to determine the action.

        Returns
        -------
        :
            A list of subject's reward and state before taking an action and
            agent's action.

        Notes
        -----
        For `instance generators`, only the current
        instance is run to termination, not the whole generator.
        '''
        while not subject_instance.is_terminated(agent_id):
            cls.interact_once(agent_id, agent_observer, subject_instance,
                              state_name, action_name, reward_function_name,
                              epoch)

    def assert_protocol(self, protocol: InteractionProtocol) -> None:
        '''
        Check whether the given protocol:

        * contains only entities that are known to the `environment`.

        * unit is one of the possible values.

        Arguments
        ---------
        protocol:
            An interaction protocol.

        Raises
        ------
        ValueError
            `agent` or `subject` is not defined.

        ValueError
            `unit` is not one of `interaction`, `instance`, or `epoch`.
        '''
        if protocol.agent.name not in self._agents:
            raise ValueError(f'Unknown agent name: {protocol.agent.name}.')
        if protocol.subject.name not in self._subjects:
            raise ValueError(f'Unknown subject name: {protocol.subject.name}.')
        if protocol.unit not in ('interaction', 'instance', 'epoch'):
            raise ValueError(
                f'Unknown unit: {protocol.unit}. '
                'It should be one of interaction, instance, or epoch. '
                'For subjects of non-instance generator, epoch and '
                'instance are equivalent.')
        # if (protocol.agent_name in self._instance_generators or
        #     protocol.subject_name in self._instance_generators) and

    def register(self,
                 interaction_protocol: InteractionProtocol,
                 get_agent_observer: bool = False) -> None:
        '''
        Register the `agent` and `subject` of an interaction protocol.

        Arguments
        ---------
        interaction_protocol:
            The protocol whose `agent` and `subject` should be registered.

        get_agent_observer:
            If `True`, the method calls the `observe` method of the `agent`
            with `subject_id`, and adds the resulting generator to the list
            of observers.

        Notes
        -----
        When registration happens for the first time, agents and subjects
        get any ID that the counterpart provides. However, in the follow up
        registrations, `entities` attempt to register with the same ID to
        have access to the same information.
        '''
        a_name = interaction_protocol.agent.name
        a_stat = interaction_protocol.agent.statistic_name
        s_name = interaction_protocol.subject.name
        a_s_name = (a_name, s_name)

        a_id, s_id = self._assignment_list[a_s_name]
        a_id = self._subjects[s_name].register(entity_name=a_name, _id=a_id)
        s_id = self._agents[a_name].register(entity_name=s_name, _id=s_id)

        self._assignment_list[a_s_name] = (a_id, s_id)

        if get_agent_observer:
            self._agent_observers[a_s_name] = \
                self._agents[a_name].observe(s_id, a_stat)

    def close_agent_observer(self, protocol: InteractionProtocol) -> None:
        '''
        Close an `agent_observer` corresponding to `protocol`.

        Before closing the observer, the final `reward` and `state` of the
        system are passed on to the observer.

        Attributes
        -----------
        protocol:
            The protocol whose `agent_observer` should be closed.

        Notes
        -----
        This method should only be used if a `subject` is terminated.
        Otherwise, the `agent_observer` might be expecting to receive different
        values, and it will corrupt the training data for the `agent`.
        '''
        agent_name = protocol.agent.name
        subject_name = protocol.subject.name
        r_func_name = protocol.reward_function_name
        state_name = protocol.state_name
        a_s_names = (agent_name, subject_name)

        if inspect.getgeneratorstate(
                self._agent_observers[a_s_names]) != inspect.GEN_SUSPENDED:
            return

        a_id, _ = self._assignment_list[a_s_names]
        reward = self._subjects[subject_name].reward(
            name=r_func_name, _id=a_id)
        state = self._subjects[subject_name].state(
            name=state_name, _id=a_id)

        self._agent_observers[a_s_names].send(reward)
        self._agent_observers[a_s_names].send({'state': state,
                                               'actions': None,
                                               'epoch': None})
        self._agent_observers[a_s_names].close()

    def reset_subject(self, subject_name: str) -> bool:
        '''
        When a `subject` is terminated for all interacting `agents`, this
        function is called to reset the `subject`.

        If the `subject` is an `InstanceGenerator`, a new instance is created.
        If reset is successful, `epoch` is incremented by one.

        Attributes
        ----------
        subject_name:
            Name of the `subject` that is terminated.

        Returns
        -------
        :
            `True` if the `instance_generator` for the `subject` is still
            active, `False` if it hit `StopIteration`.

        Notes
        -----
        `Environment.reset_subject` only resets the `subject`. It does not
        get the statistics for that `subject`.
        '''
        if subject_name in self._instance_generators:
            # get a new instance if possible,
            # if not instance generator returns StopIteration.
            # So, increment epoch by 1, then if the generator is not
            # terminated, get a new instance.
            # If the generator is terminated, check if it is finite. If
            # infinite, call it again to get a subject. If not, disable reward
            # for the current subject, so that agent_observer does not raise
            # exception.
            try:
                _, self._subjects[subject_name] = next(
                    self._instance_generators[subject_name])  # type: ignore

            except StopIteration:
                self._epochs[subject_name] += 1
                if self._instance_generators[subject_name].is_terminated():
                    self._subjects[subject_name].reward.disable()
                else:
                    _, self._subjects[subject_name] = next(
                        self._instance_generators[subject_name]
                        )  # type: ignore
                return False
        else:
            self._epochs[subject_name] += 1
            self._subjects[subject_name].reset()

        return True

    def load(self,  # noqa: C901
             entity_name: Union[List[str], str] = 'all',
             filename: Optional[str] = None,
             path: Optional[Union[str, pathlib.PurePath]] = None) -> None:
        '''
        Load an entity or an `environment` from a file.

        Arguments
        ---------
        filename:
            The name of the file to be loaded.

        entity_name:
            If specified, that entity (`agent` or `subject`) is being
            loaded from file. 'all' loads an `environment`.

        Raises
        ------
        ValueError
            The filename is not specified.
        '''
        _filename: str = filename or self._name
        _path = pathlib.Path(path or self._path)

        if entity_name == 'all':
            super().load(filename=_filename, path=_path)
            self._instance_generators: Dict[str, EntityGenType] = {}
            self._agents: Dict[str, Agent] = {}
            self._subjects: Dict[str, Subject] = {}

            for name, obj_type in self._env_data['instance_generators']:
                self._instance_generators[name] = obj_type.from_pickle(
                    path=(_path / f'{_filename}.instance_generators'),
                    filename=name)

            for name, obj_type in self._env_data['agents']:
                if name in self._instance_generators:
                    self._agents[name] = (  # type: ignore
                        self._instance_generators[name]._object)
                else:
                    self._agents[name] = obj_type.from_pickle(
                        path=(_path / f'{_filename}.agents'), filename=name)

            for name, obj_type in self._env_data['subjects']:
                if name in self._instance_generators:
                    self._subjects[name] = (  # type: ignore
                        self._instance_generators[name]._object)
                else:
                    self._subjects[name] = obj_type.from_pickle(
                        path=(_path / f'{_filename}.subjects'), filename=name)

            del self._env_data

        else:
            for obj in entity_name:
                if obj in self._instance_generators:
                    self._instance_generators[obj].load(
                        path=(_path / f'{_filename}.instance_generators'),
                        filename=obj)
                    self._instance_generators[obj].reset()

                if obj in self._agents:
                    self._agents[obj].load(
                        path=(_path / f'{_filename}.agents'), filename=obj)
                    self._agents[obj].reset()
                elif obj in self._subjects:
                    self._subjects[obj].load(
                        path=(_path / f'{_filename}.subjects'), filename=obj)
                    self._subjects[obj].reset()

    def save(self,  # noqa: C901
             filename: Optional[str] = None,
             path: Optional[Union[str, pathlib.PurePath]] = None,
             data_to_save: Union[List[str], str] = 'all'
             ) -> Tuple[pathlib.PurePath, str]:
        '''
        Save an entity or the `environment` to a file.

        Arguments
        ---------
        filename:
            The name of the file to be saved.

        path:
            The path of the file to be saved.

        entity_name:
            If specified, that entity (`agent` or `subject`) is being saved
            to file. 'all' saves the `environment`.

        Raises
        ------
        ValueError
            The filename is not specified.
        '''
        _filename = filename or self._name
        _path = pathlib.Path(path or self._path)

        if data_to_save == 'all':
            open_observers = set(a_s_names
                                 for a_s_names in self._agent_observers
                                 if inspect.getgeneratorstate(
                                     self._agent_observers[a_s_names]
                                 ) not in [inspect.GEN_CREATED or
                                           inspect.GEN_CLOSED])
            if open_observers:
                raise RuntimeError('Cannot save an environment in '
                                   'the middle of a simulation. '
                                   'These agent/subject interactions '
                                   'are still underway:\n'
                                   f'{open_observers}')

            temp, self._agent_observers = (  # type: ignore
                self._agent_observers, None)

            self._env_data: Dict[str, List[Any]] = defaultdict(list)

            try:
                for name, entity in self._instance_generators.items():
                    _, filename = entity.save(
                        path=_path / f'{_filename}.instance_generators',
                        filename=name)
                    self._env_data['instance_generators'].append(
                        (filename, type(entity)))

                for name, agent in self._agents.items():
                    if name in self._instance_generators:
                        self._env_data['agents'].append((name, None))
                    else:
                        _, filename = agent.save(
                            path=_path / f'{_filename}.agents', filename=name)
                        self._env_data['agents'].append(
                            (filename, type(agent)))

                for name, subject in self._subjects.items():
                    if name in self._instance_generators:
                        self._env_data['subjects'].append((name, None))
                    else:
                        _, filename = subject.save(
                            path=_path / f'{_filename}.subjects',
                            filename=name)
                        self._env_data['subjects'].append(
                            (filename, type(subject)))

                super().save(
                    filename=_filename, path=_path,
                    data_to_save=tuple(v for v in self.__dict__
                                       if v not in ('_agents',
                                                    '_subjects',
                                                    '_instance_generators')))

                del self._env_data

            finally:
                self._agent_observers = temp

        else:
            for obj in data_to_save:
                if obj in self._instance_generators:
                    self._instance_generators[obj].save(
                        path=_path / f'{_filename}.instance_generators',
                        filename=obj)
                elif obj in self._agents:
                    self._agents[obj].save(
                        path=_path / f'{_filename}.agents', filename=obj)
                elif obj in self._subjects:
                    self._subjects[obj].save(
                        path=_path / f'{_filename}.subjects', filename=obj)
                else:
                    self._logger.warning(f'Cannot save {obj} individually. '
                                         'Try saving the whole environment.')

        return _path, _filename

    def report_statistics(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        try:
            return super().__repr__() + '\n Agents:\n' + \
                '\n\t'.join((a.__repr__() for a in self._agents.values())) + \
                '\nSubjects:\n' + \
                '\n\t'.join((s.__repr__() for s in self._subjects.values()))
        except AttributeError:
            return super().__repr__()
