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
import configparser
import itertools
import random
from statistics import stdev
from pathlib import Path
from ast import literal_eval as make_tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from rl.agents import DQNAgent  # , WarfarinQAgent
from rl.environments import Environment
from rl.stats import WarfarinStats
from rl.subjects import IterableSubject, WarfarinModelFixedInterval, WarfarinModel_v5, WarfarinLookAhead

all_args = {
    'project_name': {'type': str, 'default': None},
    'start_epoch': {'type': int, 'default': 0},
    'epochs': {'type': int, 'default': 200},
    'training_size': {'type': int, 'default': 250},
    'training_save_path': {'type': str, 'default': './training'},
    'test_size': {'type': int, 'default': 10000},
    'test_save_path': {'type': str, 'default': './comparison_test_set'},
    'max_day': {'type': int, 'default': 90},
    'dose_history_length': {'type': int, 'default': 10},
    'INR_history_length': {'type': int, 'default': 10},
    'interval': {'type': int, 'default': (1,)},
    'interval_max_dose': {'type': int, 'default': (15,)},
    'patient_selection_training': {'type': str, 'default': 'ravvaz'},
    'patient_selection_test': {'type': str, 'default': 'random'},
    'action_type': {'type': str, 'default': 'dose_only'},
    'INR_penalty_coef': {'type': float, 'default': 1.0},
    'dose_change_penalty_coef': {'type': float, 'default': 0.0},
    'dose_change_penalty_func': {'type': str, 'default': 'none'},
    'dose_change_penalty_days': {'type': int, 'default': 0},
    'lookahead_duration': {'type': int, 'default': 0},
    'lookahead_penalty_coef': {'type': float, 'default': 0.0},
    'randomized': {'type': bool, 'default': True},
    'agent_type': {'type': str, 'default': 'DQN'},
    'method': {'type': str, 'default': 'backward'},
    'gamma': {'type': float, 'default': 0.95},
    'learning_rate': {'type': float, 'default': 0.001},
    'lr_scheduler': {'type': bool, 'default': False},
    'buffer_size': {'type': int, 'default': 90*10},
    'batch_size': {'type': int, 'default': 50},
    'validation_split': {'type': float, 'default': 0.3},
    'hidden_layer_sizes': {'type': int, 'default': (32, 32, 32)},
    'clear_buffer': {'type': bool, 'default': False},
    'initial_phase_duration': {'type': int, 'default': 1},
    'max_initial_dose_change': {'type': float, 'default': 15.0},
    'max_day_1_dose': {'type': float, 'default': 15.0},
    'maintenance_day_interval': {'type': int, 'default': 1},
    'max_maintenance_dose_change': {'type': float, 'default': 15.0},
    'state_representation': {'type': str, 'default': 'extended'},
    'save_instances': {'type': bool, 'default': False},
    'save_epochs': {'type': bool, 'default': False},
    'tf_log': {'type': bool, 'default': False}
    }

warfarin_subjects_list = {
    'warfarinmodel': WarfarinModel_v5,
    'warfarinmodel_v5': WarfarinModel_v5,
    'fixedinterval': WarfarinModelFixedInterval,
    'warfarinmodelfixedinterval': WarfarinModelFixedInterval,
    'lookahead': WarfarinLookAhead,
    'warfarinlookahead': WarfarinLookAhead
}

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    for arg, info in all_args.items():
        if isinstance(info['default'], (list, tuple)):
            parser.add_argument(f'--{arg}', nargs='+', type=info['type'], default=info['default'])
        else:
            parser.add_argument(f'--{arg}', type=info['type'], default=info['default'])

    args = parser.parse_args()

    return args

def parse_config(arguments, config_filename='config.ini', overwrite=False):
    if arguments.project_name is None:
        project_name = f'{random.randint(0, 1000000):06}'
        print(f'New project name: {project_name} created.')
    else:
        project_name = arguments.project_name
        print(f'Project {project_name} exists.')

    if not Path(config_filename).is_file():
        config = configparser.ConfigParser()
        config['DEFAULT'] = {}
        for arg, info in all_args.items():
            if arg != 'project_name':
                config['DEFAULT'][arg] = str(info['default'])
        with open(config_filename, 'w') as configfile:
            config.write(configfile)

    config = configparser.ConfigParser()
    config.sections()
    config.read(config_filename)

    vars_args = vars(arguments)

    if overwrite or project_name not in config:
        config[project_name] = {}
        for v in vars_args:
            if v != 'project_name':
                if str(vars_args[v]) != config['DEFAULT'][v]:
                    config[project_name][v] = str(vars_args[v])
        with open(config_filename, 'w') as configfile:
            config.write(configfile)

    temp = {"project_name": project_name}
    for v in vars_args:
        if v != 'project_name':
            try:
                if all_args[v]['type'] is bool:
                    temp[v] = config[project_name].getboolean(v)
                elif isinstance(all_args[v]['default'], (list, tuple)):
                    temp[v] = make_tuple(config[project_name][v])
                else:
                    temp[v] = all_args[v]['type'](config[project_name][v])
            except KeyError:
                pass

    return temp

if __name__ == "__main__":
    set_seeds(1234)
    cmd_args = parse_args()
    args = parse_config(cmd_args)

    epsilon = lambda n: 1/(1+n/200)

    filename = args["project_name"]
    try:
        env = Environment(filename=f'{filename}', path=f'{filename}')
        agents = env._agent
        subjects = env._subject
        assignment_list = list((a, s) for a in agents.keys() for s in subjects.keys())
    except FileNotFoundError:
        env = Environment()
        agents = {}
        subjects = {}

        if args["dose_change_penalty_func"].lower() == 'none':
            dose_change_penalty_func = lambda x: 0
        elif args["dose_change_penalty_func"] == 'change':
            dose_change_penalty_func = \
                lambda x: int(max(x[-i]!=x[-i-1]
                    for i in range(1, args["dose_change_penalty_days"])))
        elif args["dose_change_penalty_func"] == 'stdev':
            if args["dose_change_penalty_days"] == args["dose_history_length"]:
                dose_change_penalty_func = lambda x: stdev(x)
            else:
                dose_change_penalty_func = \
                    lambda x: stdev(list(itertools.islice(x,
                        args["dose_history_length"] - args["dose_change_penalty_days"], args["dose_history_length"])))
        elif args["dose_change_penalty_func"] == 'change_count':
            if args["dose_change_penalty_days"] == args["dose_history_length"]:
                dose_change_penalty_func = lambda x: (np.diff(x) != 0).sum()
            else:
                dose_change_penalty_func = \
                    lambda x: (np.diff(itertools.islice(x,
                        args["dose_history_length"] - args["dose_change_penalty_days"],
                        args["dose_history_length"])) != 0).sum()

        warfarin_subject = warfarin_subjects_list['warfarinmodel']

        # define subjects
        if args["training_size"]>0:
            training_patient = \
                warfarin_subject(max_day=args["max_day"],
                    patient_selection=args["patient_selection_training"],
                    dose_history_length=args["dose_history_length"],
                    INR_history_length=args["INR_history_length"],
                    action_type=args["action_type"],
                    INR_penalty_coef=args["INR_penalty_coef"],
                    dose_change_penalty_coef=args["dose_change_penalty_coef"],
                    dose_change_penalty_func=dose_change_penalty_func,
                    lookahead_penalty_coef=args["lookahead_penalty_coef"],
                    lookahead_duration=args["lookahead_duration"],
                    randomized=args["randomized"],
                    interval=args["interval"],
                    interval_max_dose=args["interval_max_dose"],
                    ex_protocol_current={'state': args["state_representation"]})

            subjects['training'] = \
                IterableSubject(subject=training_patient,
                    save_instances=args["save_instances"],
                    use_existing_instances=True,
                    save_path= args["training_save_path"],  # f'./training_{args["patient_selection_training"]}',
                    save_prefix='',
                    instance_counter_start=0,
                    instance_counter=0,
                    instance_counter_end=list(range(args["training_size"],
                                                    args["training_size"]*args["epochs"] + 1,
                                                    args["training_size"]))
                    )

            training_patient_for_stats = \
                warfarin_subject(max_day=args["max_day"],
                    patient_selection=args["patient_selection_training"],
                    dose_history_length=args["dose_history_length"],
                    INR_history_length=args["INR_history_length"],
                    action_type=args["action_type"],
                    INR_penalty_coef=args["INR_penalty_coef"],
                    dose_change_penalty_coef=args["dose_change_penalty_coef"],
                    dose_change_penalty_func=dose_change_penalty_func,
                    lookahead_penalty_coef=args["lookahead_penalty_coef"],
                    lookahead_duration=args["lookahead_duration"],
                    randomized=args["randomized"],
                    interval=args["interval"],
                    interval_max_dose=args["interval_max_dose"],
                    ex_protocol_current={'state': args["state_representation"], 'take_effect': 'no_reward'})

            subjects['training_patient_for_stats'] = \
                IterableSubject(subject=training_patient_for_stats,
                    save_instances=args["save_instances"],
                    use_existing_instances=True,
                    save_path= args["training_save_path"],  # f'./training_{args["patient_selection_training"]}',
                    save_prefix='',
                    instance_counter_start=0,
                    instance_counter=0,
                    instance_counter_end=list(range(args["training_size"],
                                                    args["training_size"]*args["epochs"] + 1,
                                                    args["training_size"]))
                    )

            input_length = len(subjects['training'].state.normalize().as_list()) \
                + len(subjects['training'].possible_actions[0].normalize().as_list())


        if args["test_size"]>0:
            test_patient = \
                warfarin_subject(max_day=args["max_day"],
                    patient_selection=args["patient_selection_test"],
                    dose_history_length=args["dose_history_length"],
                    INR_history_length=args["INR_history_length"],
                    action_type=args["action_type"],
                    INR_penalty_coef=args["INR_penalty_coef"],
                    dose_change_penalty_coef=args["dose_change_penalty_coef"],
                    dose_change_penalty_func=dose_change_penalty_func,
                    lookahead_penalty_coef=args["lookahead_penalty_coef"],
                    lookahead_duration=args["lookahead_duration"],
                    randomized=args["randomized"],
                    interval=args["interval"],
                    interval_max_dose=args["interval_max_dose"],
                    ex_protocol_current={'state': args["state_representation"], 'take_effect': 'no_reward'})

            subjects['test'] = \
                IterableSubject(subject=test_patient,
                    save_instances=True,
                    use_existing_instances=True,
                    save_path= args["test_save_path"],  # save_path=f'./test_{args["patient_selection_test"]}',
                    save_prefix='test',
                    instance_counter_start=0,
                    instance_counter=0,
                    instance_counter_end=args["test_size"],
                    auto_rewind=True
                    )

            input_length = len(subjects['test'].state.normalize().as_list()) \
                + len(subjects['test'].possible_actions[0].normalize().as_list())


        class lr_scheduler:
            def __init__(self, min_lr= 1e-5, factor=2, step=1):
                self._min_lr = min_lr
                self._factor = factor
                self._step = step

            def schedule(self, epoch, lr):
                if epoch % self._step != 0 or epoch == 0:
                    return lr
                else:
                    return max(self._min_lr, lr / self._factor)

        agents['protocol'] = \
            DQNAgent(lr_initial=args["learning_rate"],
                     lr_scheduler=lr_scheduler().schedule if args["lr_scheduler"] else None,
                     gamma=args["gamma"],
                     epsilon=epsilon,
                     buffer_size=args["buffer_size"],
                     clear_buffer=args["clear_buffer"],
                     batch_size=args["batch_size"],
                     input_length=input_length,
                     validation_split=args["validation_split"],
                     hidden_layer_sizes=tuple(args["hidden_layer_sizes"]),
                     default_actions=subjects['training'].possible_actions,
                     tensorboard_path=filename if args["tf_log"] else None,
                     save_instances=args["save_instances"],
                     method=args["method"])

        # update environment
        env.add(agents=agents, subjects=subjects)
        assignment_list = list((a, s) for a in agents.keys() for s in subjects.keys())
        env.assign(assignment_list)

    stats_to_collect = ('TTR', 'dose_change', 'delta_dose')
    aggregators = ('min', 'max', 'mean','std', 'median')
    warf_stats = WarfarinStats(active_stats=stats_to_collect,
                               aggregators=aggregators,
                               groupby=['sensitivity'])

    if args["save_epochs"]:
        env_filename = lambda i: filename+f'{i:04}'
    else:
        env_filename = lambda i: filename

    training_mode = dict((k, v) for k, v in {('protocol', 'training'): True,
                    ('protocol', 'training_patient_for_stats'): False,
                    ('protocol', 'test'): False}.items() if k in assignment_list)
    stats = dict((k, v) for k, v in {('protocol', 'training_patient_for_stats'): stats_to_collect,
             ('protocol', 'test'): stats_to_collect}.items() if k in assignment_list)
    return_output = dict((k, v) for k, v in {('protocol', 'training'): False,
                     ('protocol', 'training_patient_for_stats'): True,
                     ('protocol', 'test'): True}.items() if k in assignment_list)

    for i in range(args["epochs"]):
        print(f'run {i: }')
        stats_output, trajectory_output = \
            env.elapse_iterable(
                training_mode=training_mode,
                stats_func=warf_stats.aggregate,
                stats=stats,
                return_output=return_output)

        env.save(filename=f'./{filename}/{env_filename(i)}')

        if not Path(f'./{filename}/{filename}.txt').exists():
            with open(f'./{filename}/{filename}.txt', 'a+') as f:
                f.write(f'run\tagent\tsubject\tstat\taggregator\tsensitivity\tage>=65\tvalue\n')
        with open(f'./{filename}/{filename}.txt', 'a') as f:
            for k1, v1 in stats_output.items():
                for l in v1:
                    for k2, v2 in l.items():
                        for row in range(v2.shape[0]):
                            for col in range(v2.shape[1]):
                                print(f'{i}\t{k1}\t{k2}\t{v2.columns[col]}\t{v2.index[row]}\t{v2.iat[row, col]}')
                                f.write(f'{i}\t{k1[0]}\t{k1[1]}\t{k2}\t{v2.columns[col]}\t{v2.index[row][0]}\t{v2.index[row][1]}\t{v2.iat[row, col]}\n')


        trajectories = []
        for label in trajectory_output.keys():
            for hist in trajectory_output[label]:
                trajectories += [(i, label, h['instance_id'],
                    h['state']['age'][0], h['state']['CYP2C9'][0], h['state']['VKORC1'][0],
                    h['state']['INRs'][-1], h['action'][0], h['reward'][0]) for h in hist]
        trajectories_df = pd.DataFrame(trajectories, columns=['run', 'agent/subject',
                          'instance_id', 'age', 'CYP2C9', 'VKORC1', 'INR_prev', 'action',
                          'reward'])
        trajectories_df['shifted_id'] = trajectories_df['instance_id'].shift(periods=-1)
        trajectories_df['INR'] = trajectories_df['INR_prev'].shift(periods=-1)
        trajectories_df['INR'] = trajectories_df.apply(
            lambda x: x['INR'] if x['instance_id'] == x['shifted_id'] else None, axis=1)
        trajectories_df.drop('shifted_id', axis=1, inplace=True)
        trajectories_df.to_csv(f'./{filename}/{filename}{i:04}.csv')
