# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:47:46 2018

@author: Sadjad
"""

from time import time
import matplotlib.pyplot as plt

from rl.agents import QAgent, ANNAgent
from rl.environments import Environment
from rl.subjects import CancerModel


def cancer():
    # load the environment or create a new one
    filename = 'cancer_case_1'
    try:
        env = Environment(filename=filename)
    except FileNotFoundError:
        env = Environment()
        # initialize dictionaries
        agents = {}
        subjects = {}

        # define subjects
        subjects['Patient'] = CancerModel(
            drug={'initial_value': 0, 'decay_rate': 1,
                  'normal_cell_kill_rate': 0.1, 'tumor_cell_kill_rate': 0.3, 'immune_cell_kill_rate': 0.2},
            normal_cells={'initial_value': 0.40, 'growth_rate': 1, 'carrying_capacity': 1},
            tumor_cells={'initial_value': 1.00, 'growth_rate': 1.5, 'carrying_capacity': 1},
            immune_cells={'initial_value': 0, 'influx_rate': 0.33, 'threshold_rate': 0.3, 'response_rate': 0.01, 'death_rate': 0.2},
            competition_term={'normal_from_tumor': 1, 'tumor_from_normal': 1, 'tumor_from_immune': 0.5, 'immune_from_tumor': 1},
            e=lambda x: x['tumor_cells'], termination_check= lambda x: x['tumor_cells']<=1e-5, u_max=10, u_steps=20,
            state_range=[0, 0.0063, 0.0125, 0.025, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9])

        # define agents
        # agents['Doctor'] = QAgent(gamma=0.7, alpha=0.2, epsilon=0.4)
        agents['Doctor'] = ANNAgent(gamma=0.7, alpha=0.2, epsilon=0.5, learning_rate=1e-3, batch_size=10,
            default_actions=subjects['Patient'].possible_actions, input_length=40, hidden_layer_sizes=(10, 5))
        # agents['Doctor'].report(items=['states action'])
        # assign agents to subjects
        assignment = [('Doctor', 'Patient')]

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    # set experiment variables
    runs = 100
    training_episodes = 100

    # env._agent['Doctor'].data_collector.start()
    # env._agent['Doctor'].data_collector.collect(statistic=['diff-agent'])

    for i in range(runs):
        # run and collect statistics
        steps = env.elapse(episodes=training_episodes, max_steps=250, learning_method='history', step_count='yes')
        print(i, steps)
        # print(agents['Doctor'].data_collector.report(statistic=['diff-agent'], update_data=True))

        # save occasionally in case you don't lose data if you get bored of running the code!
        # env.save(filename=filename)

    print(sum(len(a) for a in agents['Doctor']._state_action_list.values()))
    history = env.trajectory()
    states = list(s.value[0] for i, s in enumerate(history['Doctor']) if (i % 3)==0)
    actions = list(a.value[0] for i, a in enumerate(history['Doctor']) if (i % 3)==1)
    rewards = list(r for i, r in enumerate(history['Doctor']) if (i % 3)==2)
    x = list(range(len(states)))
    plt.subplot(1, 3, 1)
    plt.plot(x, states, 'b')
    plt.subplot(1, 3, 2)
    plt.plot(x, actions, 'g')
    plt.subplot(1, 3, 3)
    plt.plot(x, rewards, 'r')
    plt.show()

if __name__ == '__main__':
    cancer()
