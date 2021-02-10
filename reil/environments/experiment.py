# -*- coding: utf-8 -*-
'''
Experiment class
================

A class that accepts an `Environment` and executes *training*, *validation*,
and *test* stages.
'''
import inspect
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from reil import Stateful
from reil import agents as rlagents
from reil import subjects as rlsubjects
from reil.datatypes import InteractionProtocol
from reil.environments import EntityType, EntityGenType, EnvironmentStaticMap
from reil.utils import InstanceGenerator, OutputWriter


class Experiment(Stateful):
    def __init__(self,
                 env: EnvironmentStaticMap,
                 validation_subjects: Dict[
                     str, Union[EntityType, EntityGenType, str]],
                 test_subjects: Dict[
                     str, Union[EntityType, EntityGenType, str]],
                 writer: OutputWriter,
                 max_epoch: int = -1,
                 **kwargs: Any):
        '''
        Arguments
        ---------
        ... TODO: Documentation
        '''
        super().__init__(**kwargs)
        self._env = env
        self._max_epoch = max_epoch

        subject_set = set(self._env._subjects)
        no_val_subjects = subject_set.difference(validation_subjects)

        if no_val_subjects:
            self._logger.warning('No validation subjects for '
                                 f'{no_val_subjects}')

        no_test_subjects = subject_set.difference(test_subjects)

        if no_test_subjects:
            self._logger.warning('No test subjects for '
                                 f'{no_test_subjects}')

        # TODO: test generators...
        # generator_set = subject_set.intersection(
        #     self._env._instance_generators)

        # self._validation_generators = dict(
        #     (name, obj)
        #     for name, obj in validation_subjects.items()
        #     if isinstance(obj, InstanceGenerator))

        # self._test_generators = dict(
        #     (name, obj)
        #     for name, obj in test_subjects.items()
        #     if isinstance(obj, InstanceGenerator))

        self._validation_subjects = {f'{name}_validation': obj
            for name, obj in validation_subjects.items()}
        # if name not in self._validation_generators)

        self._test_subjects = {f'{name}_test': obj
            for name, obj in test_subjects.items()}
        # if name not in self._test_generators)

        self._env.add(self._validation_subjects)
        self._env.add(self._test_subjects)

        self._training_interaction_sequence = tuple(
            self._env.interaction_sequence)
        self._validation_interaction_sequence = (
            InteractionProtocol(
                agent=p.agent, subject=Entity(p.subject.name, ) f'{p.subject}_validation',
                state_name=p.state_name,
                reward_function_name=p.reward_function_name,
                n=1, unit='epoch')
            for p in self._training_interaction_sequence)

        env.simulate_one_pass()
        rep = env.report_statistics(unstack=True, reset_history=True)
        writer.write_stats_output(rep)

        temp, d_agent_test._training_trigger = (
            d_agent_test._training_trigger, 'none')

        env.interaction_sequence = interaction_protocols[1:]

        env.simulate_one_pass()
        rep = env.report_statistics(unstack=True, reset_history=True)
        writer.write_stats_output(rep)

        d_agent_test._training_trigger = temp


    def run(self) -> None:
        while max(self._env._epochs.values()) < self._max_epoch:
            self.train()
            self.validate()

        self.test()

    def train(self) -> None:
        while max(self._env._epochs.values()) < self._max_epoch:
            self._env.simulate_one_pass()

    def validate(self) -> None:
        # replace subjects

        self._env.simulate_to_termination()

        # replace subjects

    def test(self) -> None:
        self._env.simulate_to_termination()

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

        a_id, _ = cast(Tuple[int, int],
                       self._assignment_list[a_s_names])
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
                _, self._subjects[subject_name] = cast(
                    Tuple[int, rlsubjects.SubjectType],
                    next(self._instance_generators[subject_name]))

            except StopIteration:
                self._epochs[subject_name] += 1
                if self._instance_generators[subject_name].is_terminated():
                    self._subjects[subject_name].reward.disable()
                    # if self._instance_generators[subject_name].is_finite:
                    #     self._subjects[subject_name].reward.disable()
                    # else:
                    #     _, self._subjects[subject_name] = cast(
                    #         Tuple[int, rlsubjects.SubjectType],
                    #         next(self._instance_generators[subject_name]))
                else:
                    _, self._subjects[subject_name] = cast(
                        Tuple[int, rlsubjects.SubjectType],
                        next(self._instance_generators[subject_name]))
                return False
        else:
            self._epochs[subject_name] += 1
            self._subjects[subject_name].reset()

        return True

    def load(self,  # noqa: C901
             entity_name: Union[List[str], str] = 'all',
             filename: Optional[str] = None,
             path: Optional[Union[pathlib.Path, str]] = None) -> None:
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
        _path = pathlib.Path(path if path is not None else self._path)

        if entity_name == 'all':
            super().load(filename=_filename, path=_path)
            self._instance_generators: Dict[str, EntityGenType] = {}
            self._agents: Dict[str, rlagents.AgentType] = {}
            self._subjects: Dict[str, rlsubjects.SubjectType] = {}

            for name, obj_type in self._env_data['instance_generators']:
                self._instance_generators[name] = obj_type.from_pickle(
                    path=(_path / f'{_filename}.instance_generators'),
                    filename=name)

            for name, obj_type in self._env_data['agents']:
                if name in self._instance_generators:
                    self._agents[name] = \
                        self._instance_generators[name]._object
                else:
                    self._agents[name] = obj_type.from_pickle(
                        path=(_path / f'{_filename}.agents'), filename=name)

            for name, obj_type in self._env_data['subjects']:
                if name in self._instance_generators:
                    self._subjects[name] = \
                        self._instance_generators[name]._object
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
             path: Optional[Union[pathlib.Path, str]] = None,
             data_to_save: Union[List[str], str] = 'all'
             ) -> Tuple[pathlib.Path, str]:
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
