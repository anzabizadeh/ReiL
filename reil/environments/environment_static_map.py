# -*- coding: utf-8 -*-
'''
EnvironmentStaticMap class
==========================

This class provides a learning environment for any reinforcement learning
`agent` on any `subject`. The interactions between `agents` and `subjects`
are determined by a fixed `interaction_sequence`.
'''
from collections import namedtuple
import pathlib
from reil.subjects import SubjectDemon
from reil.agents import AgentDemon
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from reil.datatypes import InteractionProtocol
from reil.environments import EntityType, EntityGenType, Environment


class EnvironmentStaticMap(Environment):
    '''
    Provide an interaction and learning environment for `agents` and
    `subjects`, based on a static interaction sequence.
    '''

    def __init__(self,
                 entity_dict: Optional[
                     Dict[str, Union[EntityType, EntityGenType, str]]] = None,
                 demon_dict: Optional[
                     Dict[str, Union[AgentDemon, SubjectDemon, str]]] = None,
                 interaction_sequence: Optional[
                     Tuple[InteractionProtocol, ...]] = None,
                 **kwargs: Any):
        '''
        Arguments
        ---------
        entity_dict:
            a dictionary that contains `agents`, `subjects`, and
            `generators`.

        interaction_sequence:
            a tuple of `InteractionProtocols` that specify
            how entities interact in the simulation.
        '''
        super().__init__(
            entity_dict=entity_dict, demon_dict=demon_dict, **kwargs)

        self._interaction_sequence: Tuple[InteractionProtocol, ...] = ()

        if interaction_sequence is not None:
            self.interaction_sequence = interaction_sequence

    def remove_entity(self, entity_names: Tuple[str, ...]) -> None:
        '''
        Extends `Environment.remove_entity`.

        Remove `agents`, `subjects`, or `instance_generators` from
        the environment.

        Arguments
        ---------
        entity_names:
            A list of `agent`/ `subject` names to be deleted.

        Raises
        ------
        RuntimeError
            The entity listed for deletion is used in the
            `interaction_sequence`.

        Notes
        -----
        This method removes the item from both `agents` and `subjects`
        lists. Hence, it is not recommended to use the same name for both
        an `agent` and a `subject`.
        '''
        names_in_use = [p.agent.name
                        for p in self._interaction_sequence] + \
                       [p.subject.name
                        for p in self._interaction_sequence]
        temp = set(entity_names).difference(names_in_use)
        if temp:
            raise RuntimeError(f'Some entities are in use: {temp}')

        super().remove_entity(entity_names)

    def remove_demon(self, demon_names: Tuple[str, ...]) -> None:
        '''
        Extends `Environment.remove_demon`.

        Remove `agent demons`, or `subject demons` from
        the environment.

        Arguments
        ---------
        demon_names:
            A list of `agent demon`/ `subject demon` names to be deleted.

        Raises
        ------
        RuntimeError
            The entity listed for deletion is used in the
            `interaction_sequence`.

        Notes
        -----
        This method removes the item from both `agent_demons` and
        `subject_demons` lists.
        Hence, it is not recommended to use the same name for both
        an `agent demon` and a `subject demon`.
        '''
        names_in_use = [p.agent.demon_name
                        for p in self._interaction_sequence] + \
                       [p.subject.demon_name
                        for p in self._interaction_sequence]
        temp = set(demon_names).difference(names_in_use)
        if temp:
            raise RuntimeError(f'Some demons are in use: {temp}')

        super().remove_demon(demon_names)

    @property
    def interaction_sequence(self) -> Tuple[InteractionProtocol, ...]:
        return self._interaction_sequence

    @interaction_sequence.setter
    def interaction_sequence(self,
                             seq: Tuple[InteractionProtocol, ...]) -> None:
        self._agent_observers = {}
        for protocol in seq:
            self.assert_protocol(protocol)
            self.register(protocol, get_agent_observer=True)

        self._interaction_sequence = seq

    def simulate_pass(self, n: int = 1) -> None:
        '''
        Go through the interaction sequence for a number of passes and
        simulate interactions accordingly.

        Arguments
        ---------
        n:
            The number of passes that simulation should go.
        '''
        for _ in range(n):
            for protocol in self._interaction_sequence:
                subject_name = protocol.subject.name

                if self._subjects[subject_name].is_terminated(None):
                    continue

                agent_name = protocol.agent.name
                a_s_name = (agent_name, subject_name)
                unit = protocol.unit
                state_name = protocol.state_name
                action_name = protocol.action_name
                reward_name = protocol.reward_name
                agent_id, _ = self._assignment_list[a_s_name]

                if protocol.subject.demon_name is None:
                    subject_instance = self._subjects[subject_name]
                else:
                    subject_instance = \
                        self._subject_demons[protocol.subject.demon_name](
                            self._subjects[subject_name])

                if unit == 'interaction':
                    self.interact(
                        agent_id=agent_id,  # type: ignore
                        agent_observer=self._agent_observers[a_s_name],
                        subject_instance=subject_instance,
                        state_name=state_name,
                        action_name=action_name,
                        reward_name=reward_name,
                        iteration=self._iterations[subject_name],
                        times=protocol.n)

                    if self._subjects[subject_name].is_terminated(None):
                        self.check_subject(subject_name)

                elif unit in ('instance', 'iteration'):
                    # For iteration, simulate the current instance, then in
                    # the next if statement, simulate the rest of the
                    # generated instances.
                    self.interact_while(
                        agent_id=agent_id,  # type: ignore
                        agent_observer=self._agent_observers[a_s_name],
                        subject_instance=subject_instance,
                        state_name=state_name,
                        action_name=action_name,
                        reward_name=reward_name,
                        iteration=self._iterations[subject_name])

                    if (unit == 'iteration'
                            and subject_name in self._instance_generators):
                        while self.check_subject(subject_name):
                            self.interact_while(
                                agent_id=agent_id,  # type: ignore
                                agent_observer=self._agent_observers[a_s_name],
                                subject_instance=subject_instance,
                                state_name=state_name,
                                action_name=action_name,
                                reward_name=reward_name,
                                iteration=self._iterations[subject_name])

                    else:
                        self.check_subject(subject_name)

                else:
                    raise ValueError(f'Unknown protocol unit: {unit}.')

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
        subjects_in_use = set(s.subject.name
                              for s in self.interaction_sequence)
        no_generators = subjects_in_use.difference(self._instance_generators)
        if no_generators:
            raise TypeError(
                'Found subject(s) in the interaction_sequence that '
                f'are not instance generators: {no_generators}')

        infinites = [s
                     for s in subjects_in_use
                     if not self._instance_generators[s].is_finite]
        if infinites:
            raise TypeError('Found infinite instance generator(s) in the '
                            f'interaction_sequence: {infinites}')

        while not all(self._instance_generators[s].is_terminated()
                      for s in subjects_in_use):
            self.simulate_pass()

        self.report_statistics(True)

    def check_subject(self, subject_name: str) -> bool:
        '''
        Go over all `subjects`. If terminated, close related `agent_observers`,
        reset the `subject`, and create new `agent_observers`.
        '''
        # print(self._subjects[subject_name])
        affected_protocols = list(p for p in self._interaction_sequence
                                  if p.subject.name == subject_name)

        success: bool = True
        if affected_protocols:
            for p in affected_protocols:
                self.close_agent_observer(p)

            success = self.reset_subject(subject_name)

            for p in affected_protocols:
                self.register(p, get_agent_observer=True)

        return success

    def reset_subject(self, subject_name: str) -> bool:
        '''
        Extends `Environment.reset_subject()`.
        '''
        entities = set(
            (p.subject.statistic_name,
             self._assignment_list[(p.agent.name, p.subject.name)][1])
            for p in self.interaction_sequence
            if p.subject.name == subject_name and
            p.subject.statistic_name is not None)

        for e in entities:
            self._instance_generators.get(
                subject_name,
                self._subjects[subject_name]).statistic.append(*e)

        return super().reset_subject(subject_name)

    def report_statistics(self,
                          unstack: bool = True,
                          reset_history: bool = True
                          ) -> Dict[Tuple[str, str], pd.DataFrame]:
        '''Generate statistics for agents and subjects.

        Parameters
        ----------
        unstack:
            Whether to unstack the resulting pivottable or not.

        reset_history:
            Whether to clear up the history after computing stats.

        Returns
        -------
        :
            A dictionary with state-subject pairs as keys and dataframes as
            values.
        '''
        StatInfo = namedtuple(
            'StatInfo', ('obj', 'entity_name', 'assigned_to', 'a_s_name',
                         'aggregators', 'groupby'))

        entities = set(
            StatInfo('_agents', p.agent.name, p.subject.name,
                     (p.agent.name, p.subject.name),
                     p.agent.aggregators, p.agent.groupby)
            for p in self.interaction_sequence
            if p.agent.statistic_name is not None)

        entities.update(set(
            StatInfo('_subjects', p.subject.name, p.agent.name,
                     (p.agent.name, p.subject.name),
                     p.subject.aggregators, p.subject.groupby)
            for p in self.interaction_sequence
            if p.subject.statistic_name is not None))

        transform = (
            lambda x: x.unstack().reset_index().rename(
                columns={'level_0': 'aggregator', 0: 'value'})
            if unstack else lambda x: x)

        result = {e.a_s_name:
                  transform(
                      self._instance_generators.get(
                          e.entity_name,
                          self.__dict__[e.obj][e.entity_name]
                      ).statistic.aggregate(
                          e.aggregators, e.groupby,
                          self._assignment_list[e.a_s_name][e.a_s_name.index(
                              e.entity_name)],
                          reset_history=reset_history)
                  ).assign(
                      entity=e.entity_name,
                      assigned_to=e.assigned_to,
                      iteration=self._iterations[e.a_s_name[1]])
                  for e in entities}

        return result

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
        super().load(entity_name=entity_name, filename=filename, path=path)
        # To generate observers!
        self.interaction_sequence = self.interaction_sequence
