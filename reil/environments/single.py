# -*- coding: utf-8 -*-
'''
Single class
============

This class provides a learning environment for any reinforcement learning
`agent` on any `subject`. The interactions between `agents` and `subjects`
are determined by a fixed `interaction_sequence`.
'''
from collections.abc import Generator
from typing import Any, NamedTuple, TypedDict

import pandas as pd

from reil.agents.agent_demon import AgentDemon
from reil.datatypes import InteractionProtocol, Observation
from reil.datatypes.feature import FeatureSet
from reil.environments.environment import (EntityGenType, EntityType,
                                           Environment, Plan)
from reil.subjects.subject import Subject
from reil.subjects.subject_demon import SubjectDemon


class StatInfo(NamedTuple):
    obj: str
    entity_name: str
    assigned_to: str
    a_s_name: tuple[str, str]
    aggregators: tuple[str, ...] | None
    groupby: tuple[str, ...] | None


class InteractArgs(TypedDict):
    agent_id: int
    subject_id: int
    agent_observer: Generator[FeatureSet | None, dict[str, Any], None]
    subject_instance: Subject | SubjectDemon
    protocol: InteractionProtocol
    iteration: int


class Single(Environment):
    '''
    Provide an interaction and learning environment for `agents` and
    `subjects`, based on one interaction plan.
    '''
    _object_version: str = '0.0.01'

    def __init__(
            self,
            entity_dict: dict[str, EntityType | EntityGenType | str] | None = None,
            demon_dict: dict[str, AgentDemon | SubjectDemon | str] | None = None,
            interaction_plans: dict[str, InteractionProtocol] | None = None,
            **kwargs: Any):
        '''
        Arguments
        ---------
        entity_dict:
            a dictionary that contains `agents`, `subjects`, and
            `generators`.

        interaction_plans:
            a dictionary with plan names as keys and
            tuple of `InteractionProtocols` that specify
            how entities interact in the simulation.
        '''
        super().__init__(
            entity_dict=entity_dict, demon_dict=demon_dict,
            interaction_plans=interaction_plans, **kwargs)
        self._active_plan: Plan[InteractionProtocol]

    def remove_entity(self, entity_names: tuple[str, ...]) -> None:
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
        if (plan := self._active_plan.plan):
            names_in_use = {plan.agent.name, plan.subject.name}

            temp = set(entity_names).difference(names_in_use)
            if temp:
                raise RuntimeError(f'Some entities are in use: {temp}')

        super().remove_entity(entity_names)

    def remove_demon(self, demon_names: tuple[str, ...]) -> None:
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
        if (plan := self._active_plan.plan):
            names_in_use = {plan.agent.demon_name, plan.subject.demon_name}

            temp = set(demon_names).difference(names_in_use)
            if temp:
                raise RuntimeError(f'Some demons are in use: {temp}')

        super().remove_demon(demon_names)

    def add_plans(
            self, interaction_plans: dict[str, InteractionProtocol]) -> None:
        for protocol in interaction_plans.values():
            self.assert_protocol(protocol)

        super().add_plans(interaction_plans)

    def activate_plan(self, plan_name: str) -> None:
        super().activate_plan(plan_name)
        if (plan := self._active_plan.plan):
            self.register_protocol(plan, get_agent_observer=True)

    def interact(  # noqa: C901
            self,
            agent_id: int,
            subject_id: int,
            agent_observer: Generator[FeatureSet | None, Any, None],
            subject_instance: Subject | SubjectDemon,
            protocol: InteractionProtocol,
            iteration: int,
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

        protocol:
            The `InteractionProtocol` that should be used.

        iteration:
            The iteration of of the current run. This value is used by the
            `agent` to determine the action.

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
        state_name = protocol.state_name
        action_name = protocol.action_name
        reward_name = protocol.reward_name
        lookahead = protocol.lookahead
        if lookahead is not None:
            steps = lookahead.steps
            subject_count = lookahead.subject_count
            lookahead_action_type = lookahead.action_type
            lookahead_reward_name = lookahead.reward_name
            agent = self._agents[subject_instance._entity_list[agent_id]]
        else:
            steps = 0
            subject_count = 0
            lookahead_action_type = ''
            lookahead_reward_name = ''
            agent = self._agents[subject_instance._entity_list[agent_id]]

        for _ in range(times):
            reward = subject_instance.reward(
                name=reward_name, _id=agent_id)
            # When dealing with multiple agents, the first agent enables the
            # reward. Hence, other agent observers cannot get `None` to start.
            # In such cases, we have to manually feed the generator with a
            # `None`.
            try:
                agent_observer.send(
                    None if reward is None else {'reward': reward})
            except TypeError:
                agent_observer.send(None)

            state = subject_instance.state(name=state_name, _id=agent_id)
            possible_actions = subject_instance.possible_actions(
                name=action_name, _id=agent_id)
            if possible_actions:
                try:
                    next(possible_actions)
                except TypeError:
                    pass
                action = agent_observer.send(
                    {'state': state,
                     'possible_actions': possible_actions,
                     'iteration': iteration})
                action_taken = subject_instance.take_effect(
                    action, agent_id)  # type: ignore

                if lookahead is None:
                    lookahead_data = None
                else:
                    subject_pool: list[Subject] | list[SubjectDemon] = \
                        subject_instance.copy(  # type: ignore
                            perturb=lookahead.perturb_subject, n=subject_count)
                    lookahead_data = [
                        []
                        for _ in range(subject_count)
                    ]

                    if lookahead_action_type == 'optimal':
                        current_trigger, agent._training_trigger = (
                            agent._training_trigger, 'none')
                    else:
                        current_trigger = agent._training_trigger

                    if lookahead_action_type == 'fixed':
                        def act(state, _id, actions, iteration):
                            return actions.send(
                                f'lookup {tuple(action_taken.index.values())}')
                    else:
                        act = agent.act

                    for i, subject in enumerate(subject_pool):
                        for step in range(steps):
                            if subject.is_terminated(agent_id):
                                break
                            lookahead_state = subject.state(
                                name=state_name, _id=agent_id)
                            lookahead_possible_actions = \
                                subject.possible_actions(
                                    name=action_name, _id=agent_id)
                            if lookahead_possible_actions:
                                try:
                                    next(lookahead_possible_actions)
                                except TypeError:
                                    pass
                                lookahead_action = act(
                                    lookahead_state, subject_id,
                                    lookahead_possible_actions, iteration)
                                lookahead_action_taken = subject.take_effect(
                                    action, agent_id)  # type: ignore
                                lookahead_reward = subject.reward(
                                    name=lookahead_reward_name, _id=agent_id)
                            else:
                                lookahead_action = None
                                lookahead_action_taken = None
                                lookahead_reward = None
                            lookahead_data[i].append(Observation(
                                state=lookahead_state,
                                possible_actions=lookahead_possible_actions,
                                action=lookahead_action,
                                action_taken=lookahead_action_taken,
                                reward=lookahead_reward
                            ))

                        if not subject.is_terminated(agent_id):
                            possible_actions = subject.possible_actions(
                                name=action_name, _id=agent_id)
                            if possible_actions:
                                next(possible_actions)
                            lookahead_data[i].append(Observation(
                                state=subject.state(
                                    name=state_name, _id=agent_id),
                                possible_actions=possible_actions
                            ))

                    if lookahead_action_type == 'optimal':
                        agent._training_trigger = current_trigger

                agent_observer.send({
                    'action_taken': action_taken,
                    'lookahead': lookahead_data})
                # try:
                #     possible_actions.close()
                # except AttributeError:
                #     pass

    def simulate_pass(self, n: int = 1) -> None:  # noqa: C901
        '''
        Go through the interaction sequence for a number of passes and
        simulate interactions accordingly.

        Arguments
        ---------
        n:
            The number of passes that simulation should go.
        '''
        protocol = self._active_plan.plan
        if protocol is None:
            raise ValueError('No active plan!')

        subject_name = protocol.subject.name
        agent_name = protocol.agent.name
        a_s_name = (agent_name, subject_name)
        agent_id, subject_id = self._assignment_list[a_s_name]
        if agent_id is None or subject_id is None:
            raise ValueError(f'{a_s_name} are not assigned!')

        subject_demon = protocol.subject.demon_name
        unit = protocol.unit

        for _ in range(n):
            subject_instance = self._subjects[subject_name]

            if subject_instance.is_terminated(None):
                continue

            if subject_demon:
                subject_instance = \
                    self._subject_demons[subject_demon](subject_instance)

            args: InteractArgs = {
                'agent_id': agent_id,
                'subject_id': subject_id,
                'agent_observer': self._agent_observers[a_s_name],
                'subject_instance': subject_instance,
                'protocol': protocol,
                'iteration': self._iterations[subject_name]}

            if unit == 'interaction':
                self.interact(**args, times=protocol.n)

                if self._subjects[subject_name].is_terminated(None):
                    self.check_subject(subject_name)

            elif unit == 'instance':
                self.interact_while(**args)
                self.check_subject(subject_name)

            elif unit == 'iteration':
                # For iteration, simulate the current instance, then in
                # the next if statement, simulate the rest of the
                # generated instances.
                self.interact_while(**args)

                if subject_name in self._instance_generators:
                    while self.check_subject(subject_name):
                        subject_instance = self._subjects[subject_name]
                        if subject_demon:
                            subject_instance = \
                                self._subject_demons[subject_demon](
                                    subject_instance)
                        args['subject_instance'] = subject_instance
                        args['agent_observer'] = \
                            self._agent_observers[a_s_name]

                        self.interact_while(**args)

                else:
                    self.check_subject(subject_name)

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
        plan: Any = self._active_plan.plan
        # if plan is None:
        #     raise ValueError('No active plan!')

        try:
            subject_name = plan.subject.name
        except AttributeError:
            raise ValueError('No active plan!')

        if subject_name not in self._instance_generators:
            raise TypeError(
                'Found subject(s) in the interaction_sequence that '
                f'are not instance generators: {subject_name}')

        if not self._instance_generators[subject_name].is_finite:
            raise TypeError(
                'Found infinite instance generator(s) in the '
                f'interaction_sequence: {subject_name}')

        while not self._instance_generators[subject_name].is_terminated():
            self.simulate_pass()

        self.report_statistics(True)

    def check_subject(self, subject_name: str) -> bool:
        '''
        Go over all `subjects`. If terminated, close related `agent_observers`,
        reset the `subject`, and create new `agent_observers`.
        '''
        plan: Any = self._active_plan.plan
        # if plan is None:
        #     raise ValueError('No active plan!')

        try:
            if subject_name != plan.subject.name:
                return True
        except AttributeError:
            raise ValueError('No active plan!')

        metrics = self.close_agent_observer(plan)
        success = self.reset_subject(subject_name)
        self.register_protocol(plan, get_agent_observer=True)

        if (metrics and self._stopping_criteria and
                self._agents[plan.agent.name]._training_trigger != 'none'):
            if self._stopping_criteria(
                    metrics, self._agents[plan.agent.name].get_parameters):
                _, best_parameters = self._stopping_criteria.get_best()
                if best_parameters is not None:
                    self._agents[plan.agent.name].set_parameters(
                        best_parameters)

                return False

        return success

    def reset_subject(self, subject_name: str) -> bool:
        '''
        Extends `Environment.reset_subject()`.
        '''
        plan: Any = self._active_plan.plan
        # if plan is None:
        #     raise ValueError('No active plan!')

        try:
            agent_name = plan.agent.name
        except AttributeError:
            raise ValueError('No active plan!')

        subject_name = plan.subject.name
        _id = self._assignment_list[(agent_name, subject_name)][1]
        stat_name = plan.subject.statistic_name
        trajectory_name = plan.subject.trajectory_name

        if stat_name:
            self._instance_generators.get(
                subject_name,
                self._subjects[subject_name]).statistic.append(stat_name, _id)

        if trajectory_name:
            self._subjects[subject_name].state.dump(
                trajectory_name, _id, {
                    'agent_name': agent_name,
                    'agent_demon': plan.agent.demon_name or 'none',
                    'subject_name': subject_name,
                    'subject_demon': plan.subject.demon_name or 'none',
                    'state_name': plan.state_name,
                    'action_name': plan.action_name,
                    'subject_instance_name':
                        self._subjects[subject_name]._name,
                    'environment': self._name,
                    'iteration': self._iterations[subject_name]})

        return super().reset_subject(subject_name)

    def report_statistics(
            self,
            unstack: bool = True,
            reset_history: bool = True,
    ) -> dict[tuple[str, str], pd.DataFrame]:
        '''Generate statistics for agents and subjects.

        Arguments
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
        plan = self._active_plan.plan
        if plan is None:
            raise ValueError('No active plan!')

        entities = []
        if plan.agent.statistic_name:
            entities = [StatInfo(
                '_agents', plan.agent.name, plan.subject.name,
                (plan.agent.name, plan.subject.name),
                plan.agent.aggregators, plan.agent.groupby)]

        if plan.subject.statistic_name:
            entities.append(StatInfo(
                '_subjects', plan.subject.name, plan.agent.name,
                (plan.agent.name, plan.subject.name),
                plan.subject.aggregators, plan.subject.groupby))

        def do_transform(x: pd.DataFrame) -> pd.DataFrame:
            return x.unstack().reset_index().rename(  # type: ignore
                columns={'level_0': 'aggregator', 0: 'value'})

        def no_transform(x: pd.DataFrame) -> pd.DataFrame:
            return x

        if unstack:
            transform = do_transform
        else:
            transform = no_transform

        result = {
            e.a_s_name: transform(  # type: ignore
                self._instance_generators.get(
                    e.entity_name,
                    self.__dict__[e.obj][e.entity_name]
                ).statistic.aggregate(  # type: ignore
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
