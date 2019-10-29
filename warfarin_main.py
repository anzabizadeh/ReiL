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

import numpy as np
import tensorflow as tf
from rl.agents import DQNAgent  # , WarfarinQAgent
from rl.environments import Environment
from rl.subjects import WarfarinModel_v5


if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    tf.set_random_seed(1234)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--runs', type=int, default=500)
    # parser.add_argument('--training_episodes', type=int, default=200)
    # parser.add_argument('--max_day', type=int, default=90)
    # parser.add_argument('--dose_history', type=int, default=10)
    # parser.add_argument('--INR_history', type=int, default=10)
    # parser.add_argument('--patient_selection', type=str, default='ravvaz')

    runs = 500
    training_episodes = 200

    max_day = 90
    dose_history = 10
    INR_history = 10
    patient_selection = 'ravvaz'

    randomized = True

    gamma = 0.95
    epsilon = lambda x: 1/(1+x/200)
    agent_type = 'DQN'
    input_length = 32  # 46
    buffer_size = 90*10
    batch_size = 50
    validation_split = 0.3
    hidden_layer_sizes = (20, 20)
    clear_buffer = False
    dose_change_penalty_coef = 0.0
    dose_change_penalty_func = lambda x: 0  # int(x[-7]!=x[-6]!=x[-5]!=x[-4]!=x[-3]!=x[-2]!=x[-1])
    patient_model = 'WARFV5'
    extended_state = False  # True
    save_patients = True

    text = ''.join((str(hidden_layer_sizes),
                        'g', str(gamma),
                        'e', 'fn' if callable(epsilon) else str(epsilon),
                        'bff', str(buffer_size),
                        'clr', 'T' if clear_buffer else 'F',
                        'btch', str(batch_size),
                        'vld', str(validation_split)))

    filename = ''.join((patient_model,
                            'd_{:2}'.format(max_day),
                            'dose_{:2}'.format(dose_history),
                            'INR_{:2}'.format(INR_history),
                            'T' if randomized else 'F',
                            'dose_change_coef_{:2.2f}'.format(dose_change_penalty_coef),
                            agent_type, text)).replace(' ', '')

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
        subjects['W'] = WarfarinModel_v5(max_day=max_day,
                                        patient_selection=patient_selection,
                                        dose_history=dose_history,
                                        INR_history=INR_history,
                                        dose_change_penalty_coef=dose_change_penalty_coef,
                                        dose_change_penalty_func=dose_change_penalty_func,
                                        extended_state=extended_state,
                                        randomized=randomized,
                                        save_patients=save_patients,
                                        patients_save_prefix='')

        agents['protocol'] = DQNAgent(gamma=gamma,
                                        epsilon=epsilon,
                                        buffer_size=buffer_size,
                                        clear_buffer=clear_buffer,
                                        batch_size=batch_size,
                                        input_length=input_length,
                                        validation_split=validation_split,
                                        hidden_layer_sizes=hidden_layer_sizes,
                                        default_actions=subjects['W'].possible_actions,
                                        tensorboard_path=filename,
                                        save_patients=save_patients)

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign([('protocol', 'W')])

    for i in range(runs):
        print('run {: }'.format(i))
        env.elapse(episodes=training_episodes, reset='all',
                   termination='all', learning_method='history',
                   reporting='none', tally='no')

        env.save(filename=filename+'{:04}'.format(i))

        for row in env.trajectory()['protocol'].iterrows():
            # print('{}, {} \n {}'.format(row[0], row[1].state.value.loc['Doses'], row[1].reward))
            print('{}, {} \t {} \t {}'.format(row[0], row[1].state.value.loc['Doses'][-1], row[1].state.value.loc['INRs'][-1], row[1].reward))
