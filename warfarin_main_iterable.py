# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:47:46 2018

@author: Sadjad
"""
# Disable GPU before loading tensorflow
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import argparse
import random

from statistics import stdev
import itertools

import numpy as np
import tensorflow as tf
from rl.agents import DQNAgent  # , WarfarinQAgent
from rl.environments import Environment
from rl.subjects import WarfarinModel_v5, IterableSubject
from rl.stats import WarfarinStats


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--training_episodes', type=int, default=250)
    parser.add_argument('--test_episodes', type=int, default=50)
    parser.add_argument('--max_day', type=int, default=90)
    parser.add_argument('--dose_history', type=int, default=10)
    parser.add_argument('--INR_history', type=int, default=10)
    parser.add_argument('--patient_selection', type=str, default='ravvaz')
    parser.add_argument('--dose_change_penalty_coef', type=float, default=1.0)
    parser.add_argument('--dose_change_penalty_func', type=str, default='change_count')
    parser.add_argument('--dose_change_penalty_days', type=int, default=10)

    parser.add_argument('--randomized', type=bool, default=True)

    parser.add_argument('--agent_type', type=str, default='DQN')
    parser.add_argument('--method', type=str, default='forward')
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--buffer_size', type=int, default=90*10)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--validation_split', type=float, default=0.3)
    parser.add_argument('--hidden_layer_sizes', nargs='+', type=int, default=(32, 32, 32))
    parser.add_argument('--clear_buffer', type=bool, default=False)

    parser.add_argument('--initial_phase_duration', type=int, default=90)
    parser.add_argument('--max_initial_dose_change', type=int, default=15)
    parser.add_argument('--max_day_1_dose', type=int, default=15)
    parser.add_argument('--maintenance_day_interval', type=int, default=1)
    parser.add_argument('--max_maintenance_dose_change', type=int, default=15)

    parser.add_argument('--extended_state', type=bool, default=False)
    parser.add_argument('--save_instances', type=bool, default=False)

    parser.add_argument('--save_runs', type=bool, default=False)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    set_seeds(1234)

    args = parse_args()
    print(args)

    epsilon = lambda n: 1/(1+n/200)
    if args.dose_change_penalty_func.lower() == 'none':
        dose_change_penalty_func = lambda x: 0
    elif args.dose_change_penalty_func == 'change':
        dose_change_penalty_func = lambda x: int(max(x[-i]!=x[-i-1] for i in range(1, args.dose_change_penalty_days)))
    elif args.dose_change_penalty_func == 'stdev':
        if args.dose_change_penalty_days == args.dose_history:
            dose_change_penalty_func = lambda x: stdev(x)
        else:
            dose_change_penalty_func = lambda x: stdev(list(itertools.islice(x, args.dose_history - args.dose_change_penalty_days, args.dose_history)))
    elif args.dose_change_penalty_func == 'change_count':
        if args.dose_change_penalty_days == args.dose_history:
            dose_change_penalty_func = lambda x: (np.diff(x) != 0).sum()
        else:
            dose_change_penalty_func = lambda x: (np.diff(itertools.islice(x,
                                                          args.dose_history - args.dose_change_penalty_days,
                                                          args.dose_history)) != 0).sum()

    patient_model = 'W'

    text = ''.join((str(args.hidden_layer_sizes),
                        'g', str(args.gamma),
                        'e', 'fn' if callable(epsilon) else str(epsilon),
                        'bff', str(args.buffer_size),
                        'clr', 'T' if args.clear_buffer else 'F',
                        'btch', str(args.batch_size),
                        'vld', str(args.validation_split)))

    filename = ''.join((patient_model,
                            f'{args.max_day:2}' + 'day',
                            f'd_{args.dose_history:2}',
                            f'INR_{args.INR_history:2}',
                            'T' if args.randomized else 'F',
                            f'd_chg_coef_{args.dose_change_penalty_coef:2.2f}',
                            f'func{args.dose_change_penalty_func}',
                            f'ph1_{args.initial_phase_duration:2}',
                            f'ph1chg_{args.max_initial_dose_change:2}',
                            f'mxd1_{args.max_day_1_dose:2}',
                            f'ph2d_{args.maintenance_day_interval:2}',
                            f'ph2chg_{args.max_maintenance_dose_change:2}',
                            args.agent_type,
                            'fwd' if args.method.lower() == 'forward' else 'bwd', text)).replace(' ', '')

    try:
        env = Environment(filename=filename)
        agents = env._agent
        subjects = env._subject
    except FileNotFoundError:
        env = Environment()
        # initialize dictionaries
        agents = {}
        subjects = {}

        # define subjects
        training_patient = WarfarinModel_v5(max_day=args.max_day,
                                            patient_selection=args.patient_selection,
                                            dose_history=args.dose_history,
                                            INR_history=args.INR_history,
                                            dose_change_penalty_coef=args.dose_change_penalty_coef,
                                            dose_change_penalty_func=dose_change_penalty_func,
                                            extended_state=args.extended_state,
                                            randomized=args.randomized,
                                            initial_phase_duration=args.initial_phase_duration,
                                            max_initial_dose_change=args.max_initial_dose_change,
                                            max_day_1_dose=args.max_day_1_dose,
                                            maintenance_day_interval=args.maintenance_day_interval,
                                            max_maintenance_dose_change=args.max_maintenance_dose_change)

        test_patient = WarfarinModel_v5(max_day=args.max_day,
                                        patient_selection=args.patient_selection,
                                        dose_history=args.dose_history,
                                        INR_history=args.INR_history,
                                        dose_change_penalty_coef=args.dose_change_penalty_coef,
                                        dose_change_penalty_func=dose_change_penalty_func,
                                        extended_state=args.extended_state,
                                        randomized=args.randomized,
                                        initial_phase_duration=args.initial_phase_duration,
                                        max_initial_dose_change=args.max_initial_dose_change,
                                        max_day_1_dose=args.max_day_1_dose,
                                        maintenance_day_interval=args.maintenance_day_interval,
                                        max_maintenance_dose_change=args.max_maintenance_dose_change)

        subjects['training'] = IterableSubject(subject=training_patient,
                                                save_instances=args.save_instances,
                                                use_existing_instances=True,
                                                save_path='./iterable_training',
                                                save_prefix='',
                                                instance_counter_start=0,
                                                instance_counter=0,
                                                instance_counter_end=list(range(args.training_episodes,
                                                                                args.training_episodes*args.runs + 1,
                                                                                args.training_episodes))
                                                )

        subjects['test'] = IterableSubject(subject=test_patient,
                                            save_instances=True,
                                            use_existing_instances=True,
                                            save_path='./iterable_test',
                                            save_prefix='test',
                                            instance_counter_start=0,
                                            instance_counter=0,
                                            instance_counter_end=args.test_episodes,
                                            auto_rewind=True
                                            )

        input_length = len(subjects['training'].state.normalize().as_list()) + len(subjects['training'].possible_actions[0].normalize().as_list())

        agents['protocol'] = DQNAgent(learning_rate=0.1,
                                        gamma=args.gamma,
                                        epsilon=epsilon,
                                        buffer_size=args.buffer_size,
                                        clear_buffer=args.clear_buffer,
                                        batch_size=args.batch_size,
                                        input_length=input_length,
                                        validation_split=args.validation_split,
                                        hidden_layer_sizes=tuple(args.hidden_layer_sizes),
                                        default_actions=subjects['training'].possible_actions,
                                        tensorboard_path='temp_log',
                                        save_instances=args.save_instances,
                                        method=args.method)

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign([('protocol', 'training'), ('protocol', 'test')])

        warf_stats = WarfarinStats(agent_stat_dict={'protocol': {'stats': ['TTR'], 'groupby': ['sensitivity']}})

    if args.save_runs:
        env_filename = lambda i: filename+f'{i:04}'
    else:
        env_filename = lambda i: filename

    for i in range(args.runs):
        print(f'run {i: }')
        output = env.elapse_iterable(training_status={('protocol', 'training'): True, ('protocol', 'test'): False},
                            stats_func=warf_stats.stats_func, return_output=True)

        print(output)
        # env.save(filename=env_filename(i))

        # trajectories = env.trajectory()
        # for row in trajectories[('protocol', 'test')]:
        #     print(f'{row["state"]["Doses"][-1]} \t {row["state"]["INRs"][-1]} \t {row["reward"]} \t {row["q"]}\n')

        # with open(filename+'.txt', 'a+') as f:
        #     f.write(f'{i:-^20}\n')
        #     for row in trajectories[('protocol', 'test')]:
        #         print(f'{row["state"]["Doses"][-1]} \t {row["state"]["INRs"][-1]} \t {row["reward"]} \t {row["q"]}\n')
        #         f.write(f'{row["state"]["Doses"][-1]} \t {row["state"]["INRs"][-1]} \t {row["reward"]} \t {row["q"]}\n')
