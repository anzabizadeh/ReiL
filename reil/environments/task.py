import logging
from math import ceil, log10
from typing import Dict, Literal, Optional, Union

from reil.agents import Agent, AgentDemon
from reil.datatypes import InteractionProtocol
from reil.environments import Environment, EnvironmentStaticMap
from reil.subjects import Subject, SubjectDemon
from reil.utils import InstanceGenerator, OutputWriter


class Task:
    def __init__(
            self, name: str, path: str, task_type: Literal['training', 'test'],
            interaction_protocol: InteractionProtocol,
            agent: Agent, subject: Union[Subject, InstanceGenerator],
            auto_rewind: bool,  # ??
            agent_demon: Optional[AgentDemon] = None,
            subject_demon: Optional[SubjectDemon] = None,
            start_iteration: int = 0, iterations: int = 1,
            writer: Optional[OutputWriter] = None,
            save_instances: bool = True, save_iterations: bool = True):
        # TODO: auto_rewind, save_instances, save_iterations are
        # not fully implemented!
        self._name = name
        self._path = path
        self.iterations = iterations
        self._start_iteration = start_iteration
        self._type = task_type
        self._save_instances = save_instances
        self._save_iterations = save_iterations
        self._auto_rewind = auto_rewind
        self._writer = writer

        filename_format = f'{{}}_{{:0{ceil(log10(iterations))}}}'
        self._filename = (
            lambda i: filename_format.format(name, i)
        ) if save_iterations else (lambda _: name)

        if task_type != 'training':
            agent._training_trigger = 'none'

        self._interaction_protocol = interaction_protocol
        if interaction_protocol.unit != 'iteration':
            logging.warning('Interaction protocol unit should be "iteration". '
                            'Current unit might have unintended consequences.')

        a_demon_name = interaction_protocol.agent.demon_name
        s_demon_name = interaction_protocol.subject.demon_name
        demon_dict: Dict[str, Union[AgentDemon, SubjectDemon]] = {}
        if a_demon_name:
            if agent_demon is None:
                raise ValueError(
                    'AgentDemon not provided. Expected one for '
                    f'{a_demon_name}.')
            demon_dict = {a_demon_name: agent_demon}
        if s_demon_name:
            if subject_demon is None:
                raise ValueError(
                    'SubjectDemon not provided. Expected one for '
                    f'{s_demon_name}.')
            demon_dict.update({s_demon_name: subject_demon})

        env = EnvironmentStaticMap(
            name=name,
            path=path,
            entity_dict={interaction_protocol.agent.name: agent,
                         interaction_protocol.subject.name: subject},
            demon_dict=demon_dict or None,  # type: ignore
            interaction_sequence=(interaction_protocol,))

        env._iterations[interaction_protocol.subject.name] = start_iteration
        env.save(filename=self._filename(start_iteration))

    @staticmethod
    def simulate(env: Environment, writer: Optional[OutputWriter]):
        env.simulate_pass()
        if writer:
            rep = env.report_statistics(unstack=True, reset_history=True)
            writer.write_stats_output(rep)

    def run(self, iteration: int):
        env = EnvironmentStaticMap.from_pickle(
            self._filename(iteration), self._path)

        self.simulate(env, self._writer)
        env.save(filename=self._filename(iteration + 1), path=self._path)
