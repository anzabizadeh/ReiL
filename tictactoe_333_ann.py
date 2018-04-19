# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:47:46 2018

@author: Sadjad
"""

from time import time
from itertools import product
import matplotlib.pyplot as plt

from rl.agents import ANNAgent, RandomAgent
from rl.environments import Environment
from rl.subjects import MNKGame


def mnk():
    # load the environment or create a new one
    # filename = 'mnk333_ann'
    # try:
    #     env = Environment(filename=filename)
    # except FileNotFoundError:

    runs = 1000
    training_episodes = 1000
    test_episodes = 100

    m, n, k = 3, 3, 3

    gamma = [1.0]
    alpha = [0.0, 0.2, 0.6]
    epsilon = [0.0]
    learning_rate = [1e-3]
    batch_size = [10, 20, 50]
    hidden_layer_sizes = [(26,), (26, 5), (26, 10, 5)]
    input_length = [26]
    scenario = []
    for data in product(gamma, alpha, epsilon, learning_rate, batch_size, hidden_layer_sizes, input_length):
        scenario.append({'gamma': data[0], 'alpha': data[1], 'epsilon': data[2], 'learning_rate': data[3],
                        'batch_size': data[4], 'hidden_layer_sizes': data[5], 'input_length': data[6]})

    # gma_1.0_alf_0.1_eps_0.0_slvr_sgd_lrn_0.001_hddn_(26, 5)_itr_1_btch_10
    # gma_1.0_alf_0.3_eps_0.0_slvr_sgd_lrn_0.001_hddn_(26, 5)_itr_1_btch_20
    # gma_1.0_alf_0.3_eps_0.0_slvr_sgd_lrn_0.001_hddn_(26, 10, 5)_itr_1_btch_10

    # scenario = [{'gamma': 1.0, 'alpha': 0.0, 'epsilon': 0.0, 'learning_rate': 1e-3,
    #              'batch_size': 25, 'hidden_layer_sizes': (26,), 'input_length': 26}]
    number_of_scenarios = len(scenario)

    # initialize dictionaries
    environments = []
    results = []
    for i in range(number_of_scenarios):
        environments.append(Environment(name='scenario_'+str(i)))
        subjects = {'Board': MNKGame(m=m, n=n, k=k)}
        agents = {'ANN': ANNAgent(**scenario[i], default_actions=subjects['Board'].possible_actions), 'Opponent': RandomAgent()}
        assignment = [('ANN', 'Board'), ('Opponent', 'Board')]
        environments[i].add(agents=agents, subjects=subjects)
        environments[i].assign(assignment)
        results.append({'ANN win': [], 'ANN draw': [], 'ANN lose': []})

    # env._agent['ANN'].data_collector.start()
    # env._agent['ANN'].data_collector.collect(statistic=['report'])

    print('run \tscenario \twin \tdraw \tlose')
    for run in range(runs):
        # run and collect statistics
        tally = [{}]*number_of_scenarios
        for i, env in enumerate(environments):
            env.elapse(episodes=training_episodes, reset='all',
                       termination='all', learning_method='history',
                       reporting='none', tally='no')

            tally[i] = env.elapse(episodes=test_episodes, reset='all',
                                  termination='all', learning_method='none',
                                  reporting='none', tally='yes')
            # print(agents['ANN'].data_collector.report(statistic=['report']))

            results[i]['ANN win'] = tally[i]['ANN']
            results[i]['ANN lose'] = tally[i]['Opponent']
            results[i]['ANN draw'] = test_episodes-tally[i]['ANN']-tally[i]['Opponent']

            # print result of each run
            print('{} \t{} \t{: } \t{: } \t{: }'
                .format(run, i, results[i]['ANN win'], results[i]['ANN draw'], results[i]['ANN lose']))
        env.save(path='./test')

    # x = list(range(len(results[0]['ANN win'])))
    # plt.plot(x, results[0]['ANN win'], 'b', x, results[0]['ANN draw'], 'g', x, results[0]['ANN lose'], 'r')
    # plt.axis([0, len(x), 0, training_episodes])
    # plt.show()
    # env.save(filename=filename)


if __name__ == '__main__':
    mnk()
    print('done!')
