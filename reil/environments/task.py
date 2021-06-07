import logging
from math import ceil, log10
import pathlib
from typing import Dict, Literal, Optional, Tuple, Union

from reil.datatypes import InteractionProtocol
from reil.environments import EnvironmentStaticMap
from reil.utils import OutputWriter

# TODO: Documentation
# TODO: what should we do with `save` and `iterations`?!


class Task:
    def __init__(
            self, name: str, path: Union[pathlib.PurePath, str],
            agent_training_triggers: Dict[str, Literal[
                'none', 'termination',
                'state', 'action', 'reward']],
            interaction_sequence: Tuple[InteractionProtocol, ...],
            start_iteration: int = 0, max_iterations: int = 1,
            writer: Optional[OutputWriter] = None,
            save_iterations: bool = True):
        self._name = name
        self._path = path
        self.max_iterations = max_iterations
        self._start_iteration = start_iteration
        self._agent_training_triggers = agent_training_triggers
        self._writer = writer

        # self._subjects = subjects

        self._save_iterations = save_iterations
        if save_iterations:
            self._filename_format = f'{{}}_{{:0{ceil(log10(max_iterations))}}}'
        else:
            self._filename_format = f'{{}}'

        for protocol in interaction_sequence:
            if protocol.unit != 'iteration':
                logging.warning(
                    'Interaction protocol unit should be "iteration". '
                    'Current unit might have unintended consequences.')

        self._interaction_sequence = interaction_sequence

    def run(self, environment_filename: str,
            path: pathlib.PurePath, iteration: int):
        env = EnvironmentStaticMap.from_pickle(environment_filename, path)

        env.interaction_sequence = self._interaction_sequence

        for agent, trigger in self._agent_training_triggers.items():
            env._agents[agent]._training_trigger = trigger

        for protocol in self._interaction_sequence:
            env._iterations[protocol.subject.name] = iteration

        env.simulate_pass()
        if self._writer:
            rep = env.report_statistics(unstack=True, reset_history=True)
            self._writer.write_stats_output(rep)

        # self.simulate(env, self._writer)
        # env.save(
        #     filename=self._filename_format.format(self._name, iteration + 1),
        #     path=self._path)
