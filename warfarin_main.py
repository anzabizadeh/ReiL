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
from rl.subjects import WarfarinModel_v5


if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    tf.set_random_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=500)
    parser.add_argument('--training_episodes', type=int, default=200)
    parser.add_argument('--max_day', type=int, default=90)
    parser.add_argument('--dose_history', type=int, default=10)
    parser.add_argument('--INR_history', type=int, default=10)
    parser.add_argument('--patient_selection', type=str, default='ravvaz')
    parser.add_argument('--dose_change_penalty_coef', type=float, default=1.0)
    parser.add_argument('--dose_change_penalty_func', type=str, default='stdev')
    parser.add_argument('--dose_change_penalty_days', type=int, default=10)

    parser.add_argument('--randomized', type=bool, default=True)

    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--agent_type', type=str, default='DQN')
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
    parser.add_argument('--save_patients', type=bool, default=False)

    parser.add_argument('--save_runs', type=bool, default=False)

    args = parser.parse_args()
    print(args)

    epsilon = lambda n: 1/(1+n/200)
    if args.dose_change_penalty_func.lower() == 'none':
        dose_change_penalty_func = lambda x: 0
    elif args.dose_change_penalty_func == 'change':
        dose_change_penalty_func = lambda x: int(max(x[-i]!=x[-i-1] for i in range(1, args.dose_change_penalty_days)))
    elif args.dose_change_penalty_func == 'stdev':
        if args.dose_change_penalty_days == args.dose_history:
            dose_change_penalty_func = lambda x: stdev(x)
        else
            dose_change_penalty_func = lambda x: stdev(list(itertools.islice(x, args.dose_history - args.dose_change_penalty_days, args.dose_history)))

    patient_model = 'W'

    text = ''.join((str(args.hidden_layer_sizes),
                        'g', str(args.gamma),
                        'e', 'fn' if callable(epsilon) else str(epsilon),
                        'bff', str(args.buffer_size),
                        'clr', 'T' if args.clear_buffer else 'F',
                        'btch', str(args.batch_size),
                        'vld', str(args.validation_split)))

    filename = ''.join((patient_model,
                            '{:2}'.format(args.max_day) + 'day',
                            'd_{:2}'.format(args.dose_history),
                            'INR_{:2}'.format(args.INR_history),
                            'T' if args.randomized else 'F',
                            'd_chg_coef_{:2.2f}'.format(args.dose_change_penalty_coef),
                            'func{}'.format(args.dose_change_penalty_func),
                            'ph1_{:2}'.format(args.initial_phase_duration),
                            'ph1chg_{:2}'.format(args.max_initial_dose_change),
                            'mxd1_{:2}'.format(args.max_day_1_dose),
                            'ph2d_{:2}'.format(args.maintenance_day_interval),
                            'ph2chg_{:2}'.format(args.max_maintenance_dose_change),
                            args.agent_type, text)).replace(' ', '')

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
        subjects['W'] = WarfarinModel_v5(max_day=args.max_day,
                                        patient_selection=args.patient_selection,
                                        dose_history=args.dose_history,
                                        INR_history=args.INR_history,
                                        dose_change_penalty_coef=args.dose_change_penalty_coef,
                                        dose_change_penalty_func=dose_change_penalty_func,
                                        extended_state=args.extended_state,
                                        randomized=args.randomized,
                                        save_patients=args.save_patients,
                                        patients_save_prefix='',
                                        initial_phase_duration=args.initial_phase_duration,
                                        max_initial_dose_change=args.max_initial_dose_change,
                                        max_day_1_dose=args.max_day_1_dose,
                                        maintenance_day_interval=args.maintenance_day_interval,
                                        max_maintenance_dose_change=args.max_maintenance_dose_change)

        input_length = len(subjects['W'].state.normalize().as_list()) + len(subjects['W'].possible_actions[0].normalize().as_list())

        agents['protocol'] = DQNAgent(gamma=args.gamma,
                                        epsilon=epsilon,
                                        buffer_size=args.buffer_size,
                                        clear_buffer=args.clear_buffer,
                                        batch_size=args.batch_size,
                                        input_length=input_length,
                                        validation_split=args.validation_split,
                                        hidden_layer_sizes=tuple(args.hidden_layer_sizes),
                                        default_actions=subjects['W'].possible_actions,
                                        tensorboard_path=filename,
                                        save_patients=args.save_patients,
                                        method='backward')

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign([('protocol', 'W')])

    if args.save_runs:
        env_filename = lambda i: filename+'{:04}'.format(i)
    else:
        env_filename = lambda i: filename

    for i in range(args.runs):
        print('run {: }'.format(i))
        env.elapse(episodes=args.training_episodes, reset='all',
                   termination='all', learning_method='history',
                   reporting='none', tally='no')

        env.save(filename=env_filename(i))

        with open(filename+'.txt', 'a+') as f:
            f.write('{:-^20}\n'.format(i))
            for row in env.trajectory()['protocol'].iterrows():
                # print('{}, {} \n {}'.format(row[0], row[1].state.value.loc['Doses'], row[1].reward))
                f.write('{}, {} \t {} \t {}\n'.format(row[0], row[1].state.value.loc['Doses'][-1], row[1].state.value.loc['INRs'][-1], row[1].reward))
