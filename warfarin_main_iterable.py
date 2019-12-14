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
import itertools
import random
from statistics import stdev

import numpy as np
import pandas as pd
import tensorflow as tf
from rl.agents import DQNAgent  # , WarfarinQAgent
from rl.environments import Environment
from rl.stats import WarfarinStats
from rl.subjects import IterableSubject, WarfarinLookAhead


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

    parser.add_argument('--INR_penalty_coef', type=float, default=1.0)
    parser.add_argument('--dose_change_penalty_coef', type=float, default=0.0)
    parser.add_argument('--dose_change_penalty_func', type=str, default='none')
    parser.add_argument('--dose_change_penalty_days', type=int, default=0)
    parser.add_argument('--lookahead_duration', type=int, default=7)
    parser.add_argument('--lookahead_penalty_coef', type=float, default=0.0)

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
                            f'ahead_coef_{args.lookahead_penalty_coef}',
                            f'aheadd{args.lookahead_duration}',
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
        training_patient = WarfarinLookAhead(max_day=args.max_day,
                                            patient_selection=args.patient_selection,
                                            dose_history=args.dose_history,
                                            INR_history=args.INR_history,
                                            INR_penalty_coef=args.INR_penalty_coef,
                                            dose_change_penalty_coef=args.dose_change_penalty_coef,
                                            dose_change_penalty_func=dose_change_penalty_func,
                                            lookahead_penalty_coef=args.lookahead_penalty_coef,
                                            lookahead_duration=args.lookahead_duration,
                                            extended_state=args.extended_state,
                                            randomized=args.randomized,
                                            initial_phase_duration=args.initial_phase_duration,
                                            max_initial_dose_change=args.max_initial_dose_change,
                                            max_day_1_dose=args.max_day_1_dose,
                                            maintenance_day_interval=args.maintenance_day_interval,
                                            max_maintenance_dose_change=args.max_maintenance_dose_change)

        test_patient = WarfarinLookAhead(max_day=args.max_day,
                                        patient_selection=args.patient_selection,
                                        dose_history=args.dose_history,
                                        INR_history=args.INR_history,
                                        INR_penalty_coef=args.INR_penalty_coef,
                                        dose_change_penalty_coef=args.dose_change_penalty_coef,
                                        dose_change_penalty_func=dose_change_penalty_func,
                                        lookahead_penalty_coef=args.lookahead_penalty_coef,
                                        lookahead_duration=args.lookahead_duration,
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

        agents['protocol'] = DQNAgent(learning_rate=0.001,
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

        warf_stats = WarfarinStats(agent_stat_dict={'protocol': {'stats': ['TTR', 'dose_change', 'delta_dose'], 'groupby': ['sensitivity']}})

    if args.save_runs:
        env_filename = lambda i: filename+f'{i:04}'
    else:
        env_filename = lambda i: filename

    for i in range(args.runs):
        print(f'run {i: }')
        stats, output = env.elapse_iterable(training_status={('protocol', 'training'): True, ('protocol', 'test'): False},
                                stats_func=warf_stats.stats_func, return_stats=True, return_output=True)

        with open(filename+'.txt', 'a+') as f:
            for k1, v1 in stats.items():
                for l in v1:
                    for k2, v2 in l.items():
                        for j in range(v2.shape[0]):
                            print(f'{i}\t{k1}\t{k2}\t{v2.index[j]}\t{v2[j]}')
                            f.write(f'{i}\t{k1}\t{k2}\t{v2.index[j]}\t{v2[j]}\n')

        trajectories = []
        for label in output.keys():
            for hist in output[label]:
                trajectories += [(i, label, h['instance_id'], h['state']['age'][0], h['state']['CYP2C9'][0], h['state']['VKORC1'][0], h['state']['INRs'][-1], h['action'][0], h['reward'][0]) for h in hist]
        trajectories_df = pd.DataFrame(trajectories, columns=['run', 'agent/subject', 'instance_id', 'age', 'CYP2C9', 'VKORC1', 'INR_prev', 'action', 'reward'])
        trajectories_df['shifted_id'] = trajectories_df['instance_id'].shift(periods=-1)
        trajectories_df['INR'] = trajectories_df['INR_prev'].shift(periods=-1)
        trajectories_df['INR'] = trajectories_df.apply(lambda x: x['INR'] if x['instance_id'] == x['shifted_id'] else None, axis=1)
        trajectories_df.drop('shifted_id', axis=1, inplace=True)
        trajectories_df.to_csv(f'{filename}{i:04}.csv')

        # env.save(filename=env_filename(i))
