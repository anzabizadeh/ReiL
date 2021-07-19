import logging
import pathlib
from typing import List, Optional, Union

from reil.datatypes import Entity, InteractionProtocol
from reil.datatypes.feature_array_dumper import FeatureArrayDumper
from reil.environments.environment_static_map import EnvironmentStaticMap
from reil.environments.task import Task
from reil.healthcare.trajectory_dumper import TrajectoryDumper
from reil.subjects.subject import Subject
from reil.utils import InstanceGenerator
from reil.utils.argument_parser import ConfigParser


class Trajectory:
    def __init__(
            self,
            env_filename: str,
            env_path: Optional[Union[pathlib.Path, str]],
            parser: Optional[ConfigParser] = None,
            state_dumper: Optional[FeatureArrayDumper] = None,
            agent_name: Optional[str] = None,
            subject_name: Optional[str] = None,
            state_name: Optional[str] = None,
            action_name: Optional[str] = None,
            trajectory_name: Optional[str] = None):
        self._parser = parser
        self._env_filename = env_filename
        self._env_path = env_path

        self._state_dumper = state_dumper
        self._agent_name = agent_name
        self._subject_name = subject_name
        self._state_name = state_name
        self._action_name = action_name
        self._trajectory_name = trajectory_name

    def run(
            self,
            trajectory_subjects: Union[InstanceGenerator[Subject], str],
            iteration: int,
            subject_list: Optional[List[str]] = None,
            subject_save_path: str = '.',
            state_dumper: Optional[FeatureArrayDumper] = None,
            agent_name: Optional[str] = None,
            subject_name: Optional[str] = None,
            state_name: Optional[str] = None,
            action_name: Optional[str] = None,
            trajectory_name: Optional[str] = None) -> None:

        subjects: InstanceGenerator[Subject]
        dumper = state_dumper or self._state_dumper
        if isinstance(trajectory_subjects, str):
            if self._parser:
                s = self._parser.extract(
                    'subjects', trajectory_subjects, True)
            else:
                raise ValueError('parser not found.')
        else:
            s = trajectory_subjects

        if isinstance(s, InstanceGenerator):
            subjects = s
            if dumper:
                subjects._state_dumper = dumper
                subjects._object.state._dumper = dumper

            if subject_list is not None:
                logging.warning(
                    'The parsed subject is an `InstanceGenerator`. '
                    'subject_list will not be used.'
                )
        else:
            if subject_list is None:
                raise ValueError(
                    'subject_list should be provided when '
                    'trajectory_subjects is of type str.')

            subjects = InstanceGenerator[Subject].from_instance_list(
                s, (subject_list,), save_instances=True,
                save_path=subject_save_path,
                state_dumper=dumper)

        _sub_name = subject_name or self._subject_name
        if _sub_name is None:
            raise TypeError('subject_name is not specified.')

        _agent_name = agent_name or self._agent_name
        if _agent_name is None:
            raise TypeError('agent_name is not specified.')

        _state_name = state_name or self._state_name
        if _state_name is None:
            raise TypeError('state_name is not specified.')

        _action_name = action_name or self._action_name
        if _action_name is None:
            raise TypeError('action_name is not specified.')

        _tr_name = trajectory_name or self._trajectory_name
        if _tr_name is None:
            raise TypeError('trajectory_name is not specified.')

        env = EnvironmentStaticMap.from_pickle(
            filename=self._env_filename, path=self._env_path)
        env.add_entities({_sub_name: subjects})

        protocol = InteractionProtocol(
            agent=Entity(name=_agent_name),
            subject=Entity(
                name=_sub_name,
                trajectory_name=_tr_name),
            state_name=_state_name,
            action_name=_action_name,
            reward_name='no_reward',
            n=1, unit='iteration')

        t = Task(
            name='trajectory',
            path='../experiments/trajectories',
            agent_training_triggers={_agent_name: 'none'},
            interaction_sequence=(protocol,),
        )

        t.run_env(env, iteration)


if __name__ == '__main__':
    configs = {
        c: f'{c}.yaml'
        for c in (
            'sessions',
            'tasks',
            'agents',
            'subjects',
            'demons',
            'interaction_protocols',
        )
    }
    config_path = './Warfarin-Dosing/configs'

    vars_dict = {
        # 'session': None,
        # 'project_path': None,
        # 'env_filename': None,
        'start_iteration': '0',
        'h': '01',
        'action': '237_15',
        # 'sdemon': None,
        'arm': 'aaa',
        'size': '10000',
        'size_tr': '10',
        'iterations': '100',
    }

    parser = ConfigParser(
        config_filenames=configs, config_path=config_path,
        vars_dict=vars_dict)

    t = Trajectory(
        parser=parser, env_filename='h01_237_15_10k_validation_71',
        env_path='Z:/experiments/h01_237_15_10k',
        agent_name='protocol',
        subject_name='patient_trajectory',
        state_name='patient_w_dosing_01',
        action_name='daily_15',
        trajectory_name='patient_w_full_dosing',
    )

    t.run(
        trajectory_subjects='object',
        iteration=71,
        subject_list=['10', '20', '30'],
        subject_save_path='../experiments/data',
        state_dumper=TrajectoryDumper('test', '../experiments/tr')
    )
