# -*- coding: utf-8 -*-
'''
EnvironmentStaticMap class
==========================

This class provides a learning environment for any reinforcement learning
`agent` on any `subject`. The interactions between `agents` and `subjects`
are determined by a fixed `interaction_sequence`.
'''
from typing import Any, Dict, Optional, Tuple, Union, cast

from reil import agents as rlagents
from reil import environments, stateful
from reil import subjects as rlsubjects
from reil.datatypes import InteractionProtocol
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
    # TODO: Statistics aggregators are not supported yet!
    # TODO: simulate_passes and simulate_one_pass do not return anything!

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
        for protocol in seq:
            self.assert_protocol(protocol)

        self._interaction_sequence = seq

    def simulate_one_pass(self) -> None:
        '''
        Go through the interaction sequence for one pass and
        simulate interactions accordingly.
        '''
        self.manage_terminated_subjects()
        self.register_agents()
        for protocol in self._interaction_sequence:
            agent_name = protocol.agent.name
            subject_name = protocol.subject.name
            unit = protocol.unit
            agent_id = cast(
                int, self._assignment_list[(agent_name, subject_name)])
            if unit == 'interaction':
                observations = self.interact_n_times(
                    agent_id=agent_id,
                    agent_instance=self._agents[agent_name],
                    subject_instance=self._subjects[subject_name],
                    protocol=protocol,
                    epoch=self._epochs[subject_name],
                    times=protocol.n)
            elif unit in ['instance', 'epoch']:
                # For epoch, simulate the current instance, then in the next if
                # statement, simulate the rest of the generated instances.
                observations = self.interact_while(
                    agent_id=agent_id,
                    agent_instance=self._agents[agent_name],
                    subject_instance=self._subjects[subject_name],
                    protocol=protocol,
                    epoch=self._epochs[subject_name])
            else:
                raise ValueError(f'Unknown protocol unit: {unit}.')

            self.append_observations(agent_name, subject_name, observations)

            if (unit == 'epoch'
                    and subject_name in self._instance_generators):
                for _, instance in self._instance_generators[subject_name]:
                    self._subjects[subject_name] = \
                        cast(rlsubjects.Subject, instance)
                    self._assignment_list[(agent_name, subject_name)] = \
                        self._subjects[subject_name].register(
                            agent_name=agent_name,
                            _id=self._assignment_list[
                                (agent_name, subject_name)])

                    observations = self.interact_while(
                        agent_id=agent_id,
                        agent_instance=self._agents[agent_name],
                        subject_instance=self._subjects[subject_name],
                        protocol=protocol,
                        epoch=self._epochs[subject_name])

                    self.append_observations(
                        agent_name, subject_name, observations)

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

    def register_agents(self) -> None:
        '''
        Register all `agents` in the interaction sequence in their
        corresponding `subjects`.

        Notes
        -----
        When registration happens for the first time, the agents
        get any ID that subjects provide. However, in the follow up
        registrations, `agents` attempt to register with the same ID to
        have access to the same information.
        '''
        for p in self._interaction_sequence:
            self._assignment_list[(p.agent.name, p.subject.name)] = \
                self._subjects[p.subject.name].register(
                    agent_name=p.agent.name,
                    _id=self._assignment_list[(p.agent.name, p.subject.name)])

    def _reset_subject(self, subject_name: str) -> None:
        '''
        When a `subject` is terminated for all interacting `agents`, this
        function is called to reset the subject.

        If the subject is an `InstanceGenerator`, a new instance is created.
        If reset is successful, `epoch` is incremented by one.

        Attributes
        ----------
        subject_name:
            Name of the `subject` that is terminated.


        :meta private:
        '''
        if subject_name in self._instance_generators:
            # get a new instance if possible,
            # if not instance generator returns StopIteration.
            # So, increment epoch by 1, then if the generator is not
            # terminated, get a new instance.
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
        Go over all `subjects`. If terminated, collect terminal rewards,
        calculate stats, trains related `agents`, and reset the `subject`.
        '''
        for s_name in set(p.subject.name for p in self._interaction_sequence):
            if self._subjects[s_name].is_terminated(None):
                self._collect_terminal_rewards(s_name)
                self._calculate_statistics(s_name)
                self._train_related_agents(s_name)
                self._reset_subject(s_name)

    def _collect_terminal_rewards(self, subject_name: str) -> None:
        '''
        When a `subject` is terminated for all interacting `agents`, this
        function is called to collect final rewards for all agents.

        Attributes
        -----------
        subject_name:
            Name of the `subject` that is terminated.


        :meta private:
        '''
        agents_state_n_rewards = (
            a_s_n_r
            for a_s_n_r in set(
                (p.agent.name, p.state_name, p.reward_function_name)
                for p in self._interaction_sequence
                if p.subject.name == subject_name)
        )

        for agent_name, r_func_name, state_name in agents_state_n_rewards:
            agent_id = self._assignment_list[(agent_name, subject_name)]
            reward = self._subjects[subject_name].reward(
                name=r_func_name, _id=agent_id)  # type: ignore
            self._history[(agent_name, subject_name)][-1].reward = reward
            self._history[(agent_name, subject_name)].append(
                stateful.Observation(state=self._subjects[subject_name].state(
                    name=state_name, _id=agent_id)))

    def _calculate_statistics(self, subject_name: str) -> None:
        '''
        When a `subject` is terminated for all interacting `agents`, this
        function is called to calculate statistics for all `agents` and
        the terminated `subject`.

        Attributes
        ----------
        subject_name:
            Name of the `subject` that is terminated.


        :meta private:
        '''
        agents_and_stats = (
            a_n_s
            for a_n_s in set(
                (p.agent.name, p.agent.statistic_name,
                 p.subject.statistic_name)
                for p in self._interaction_sequence
                if p.subject.name == subject_name)
        )

        for agent_name, a_stat_name, s_stat_name in agents_and_stats:
            agent_id = self._assignment_list[(agent_name, subject_name)]
            self._agent_statistics[(agent_name, subject_name)].append(
                self._agents[agent_name].statistic(a_stat_name, agent_id))
            self._subject_statistics[(agent_name, subject_name)].append(
                self._subjects[subject_name].statistic(s_stat_name, agent_id))

    def _train_related_agents(self, subject_name: str) -> None:
        '''
        When a `subject` is terminated for all interacting `agents`, this
        function is called to provide history data to any related agent
        that can learn.

        Attributes
        ----------
        subject_name:
            Name of the `subject` that is terminated.


        :meta private:
        '''
        affected_agents = (
            a_name
            for a_name in set(p.agent.name
                              for p in self._interaction_sequence
                              if p.subject.name == subject_name)
        )

        for a_name in affected_agents:
            if self._agents[a_name].training_mode:
                self._agents[a_name].learn(
                    self._history[(a_name, subject_name)][1:])
                # TODO: what about history for agents without learning?!
                # should we delete them as well, or let them grow?!!!
                # This one deletes only the ones used, but does not make sense!
                # del self._history[(a_name, subject_name)]
            # This one makes more sense, because it deletes them no matter
            # what!
            # However, there should be an easier way to do it!
            del self._history[(a_name, subject_name)]
