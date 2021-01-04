# -*- coding: utf-8 -*-
'''
EnvironmentStaticMap class
==========================

This class provides a learning environment for any reinforcement learning
`agent` on any `subject`. The interactions between `agents` and `subjects`
are determined by a fixed `interaction_sequence`.
'''
from typing import Any, Dict, Generator, Optional, Tuple, Union, cast

from reil import agents as rlagents
from reil import environments
from reil import subjects as rlsubjects
from reil.datatypes import InteractionProtocol, ReilData
from reil.utils import instance_generator as rlgenerator

AgentSubjectTuple = Tuple[str, str]
Entity = Union[rlagents.Agent, rlsubjects.Subject]
EntityGenerator = Union[rlgenerator.InstanceGenerator[rlagents.Agent],
                        rlgenerator.InstanceGenerator[rlsubjects.Subject]]


class EnvironmentStaticMap(environments.Environment):
    '''
    Provide an interaction and learning environment for `agents` and
    `subjects`, based on a static interaction sequence.
    '''

    def __init__(self,
                 entity_dict: Optional[
                     Dict[str, Union[Entity, EntityGenerator, str]]] = None,
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
        super().__init__(entity_dict=entity_dict, **kwargs)

        self._interaction_sequence: Tuple[InteractionProtocol, ...] = ()

        if interaction_sequence is not None:
            self.interaction_sequence = interaction_sequence

    def remove(self, entity_names: Tuple[str, ...]) -> None:
        '''
        Extends `Environment.remove`.

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

        super().remove(entity_names)

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

    def simulate_one_pass(self) -> None:
        '''
        Go through the interaction sequence for one pass and
        simulate interactions accordingly.
        '''
        for protocol in self._interaction_sequence:
            self.manage_terminated_subjects()
            agent_name = protocol.agent.name
            subject_name = protocol.subject.name
            a_s_name = (agent_name, subject_name)
            unit = protocol.unit
            agent_id, _ = cast(
                Tuple[int, int],
                self._assignment_list[a_s_name])
            agent_observer = self._agent_observers[a_s_name]

            if unit == 'interaction':
                self.interact_n_times(
                    agent_id=agent_id,
                    agent_observer=agent_observer,
                    subject_instance=self._subjects[subject_name],
                    protocol=protocol,
                    epoch=self._epochs[subject_name],
                    times=protocol.n)

            elif unit in ['instance', 'epoch']:
                # For epoch, simulate the current instance, then in the next if
                # statement, simulate the rest of the generated instances.
                self.interact_while(
                    agent_id=agent_id,
                    agent_observer=agent_observer,
                    subject_instance=self._subjects[subject_name],
                    protocol=protocol,
                    epoch=self._epochs[subject_name])
            else:
                raise ValueError(f'Unknown protocol unit: {unit}.')

            if (unit == 'epoch'
                    and subject_name in self._instance_generators):
                while not self._instance_generators[subject_name].is_terminated():
                # for _, instance in self._instance_generators[subject_name]:
                #     self._subjects[subject_name] = \
                #         cast(rlsubjects.Subject, instance)

                #     self.register(protocol, True)
                    self.manage_terminated_subjects()
                    agent_observer = self._agent_observers[a_s_name]

                    self.interact_while(
                        agent_id=agent_id,
                        agent_observer=agent_observer,
                        subject_instance=self._subjects[subject_name],
                        protocol=protocol,
                        epoch=self._epochs[subject_name])

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
                      for s in self._subjects):
            self.simulate_one_pass()

        self.report_statistics(True)

    def manage_terminated_subjects(self) -> None:
        '''
        Go over all `subjects`. If terminated, close related `agent_observers`,
        reset the `subject`, and create new `agent_observers`.
        '''
        terminated_protocols = (p for p in self._interaction_sequence
                                if self._subjects[
                                    p.subject.name].is_terminated(None))

        for protocol in terminated_protocols:
            self.close_agent_observer(protocol)
            self.reset_subject(protocol.subject.name)
            self.register(protocol, get_agent_observer=True)

    def reset_subject(self, subject_name: str) -> None:
        '''
        Extends `Environment.reset_subject()`.
        '''
        entities = set(
            (p.subject.statistic_name,
             self._assignment_list[(p.agent.name, p.subject.name)][0])
            for p in self.interaction_sequence
            if p.subject.name == subject_name and
            p.subject.statistic_name is not None)

        for e in entities:
            self._subjects[subject_name].statistic.append(*e)

        super().reset_subject(subject_name)

    def report_statistics(self, reset_history: bool = False):
        entities = set(
            (p.subject.name, p.subject.aggregators, p.subject.groupby,
             self._assignment_list[(p.agent.name, p.subject.name)][0])
            for p in self.interaction_sequence
            if p.subject.statistic_name is not None and
            p.subject.aggregators is not None)

        for e in entities:
            print(e)
            print(self._subjects[e[0]].statistic.aggregate(
                e[1], e[2], e[3], reset_history=reset_history))

        entities = set(
            (p.agent.name, p.agent.aggregators, p.agent.groupby,
             self._assignment_list[(p.agent.name, p.subject.name)][1])
            for p in self.interaction_sequence
            if p.agent.statistic_name is not None and
            p.agent.aggregators is not None)

        for e in entities:
            print(e)
            print(self._agents[e[0]].statistic.aggregate(
                e[1], e[2], e[3], reset_history=reset_history))
