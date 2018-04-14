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
    training_episodes = 10

    m, n, k = 3, 3, 3

    # gamma = [1.0]
    # alpha = [0.1]
    # epsilon = [0.0]
    # learning_rate = [1e-5, 1e-3]
    # batch_size = [10, 20]
    # hidden_layer_sizes = [(26, 5), (26, 10, 5)]
    # state_size = [26]
    # scenario = []
    # for data in product(gamma, alpha, epsilon, learning_rate, batch_size, hidden_layer_sizes, state_size):
    #     scenario.append({'gamma': data[0], 'alpha': data[1], 'epsilon': data[2], 'learning_rate': data[3],
    #                     'batch_size': data[4], 'hidden_layer_sizes': data[5], 'state_size': data[6]})

    # gma_1.0_alf_0.1_eps_0.0_slvr_sgd_lrn_0.001_hddn_(26, 5)_itr_1_btch_10
    # gma_1.0_alf_0.3_eps_0.0_slvr_sgd_lrn_0.001_hddn_(26, 5)_itr_1_btch_20
    # gma_1.0_alf_0.3_eps_0.0_slvr_sgd_lrn_0.001_hddn_(26, 10, 5)_itr_1_btch_10

    scenario = [{'gamma': 1.0, 'alpha': 0.1, 'epsilon': 0.0, 'learning_rate': 1e-3,
                 'batch_size': 10, 'hidden_layer_sizes': (26,), 'state_size': 26}]
    number_of_scenarios = len(scenario)

    # initialize dictionaries
    environments = []
    results = []
    for i in range(number_of_scenarios):
        environments.append(Environment())
        subjects = {'Board': MNKGame(m=m, n=n, k=k)}
        agents = {'ANN': ANNAgent(**scenario[i], default_actions=subjects['Board'].possible_actions), 'Opponent': RandomAgent()}
        assignment = [('ANN', 'Board'), ('Opponent', 'Board')]
        environments[i].add(agents=agents, subjects=subjects)
        environments[i].assign(assignment)
        results.append({'ANN win': [], 'ANN draw': [], 'ANN lose': []})

    # env._agent['ANN'].data_collector.start()
    # env._agent['ANN'].data_collector.collect(statistic=['report'])

    environments[0].save()
    environments[0].load(filename=environments[0]._name)
    for run in range(runs):
        # run and collect statistics
        tally = [{}]*number_of_scenarios
        for i, env in enumerate(environments):
            tally[i] = env.elapse(episodes=training_episodes, reset='all',
                                  termination='all', learning_method='history',
                                  reporting='none', tally='yes')
            # print(agents['ANN'].data_collector.report(statistic=['report']))

            results[i]['ANN win'] = tally[i]['ANN']
            results[i]['ANN lose'] = tally[i]['Opponent']
            results[i]['ANN draw'] = training_episodes-tally[i]['ANN']-tally[i]['Opponent']

            # print result of each run
            print('run {}: scenario {}: win: {: } draw:{: } lose:{: } '
                .format(run, i, results[i]['ANN win'], results[i]['ANN draw'], results[i]['ANN lose']), end=' ')
        env.save()
        print()

    x = list(range(len(results[0]['ANN win'])))
    plt.plot(x, results[0]['ANN win'], 'b', x, results[0]['ANN draw'], 'g', x, results[0]['ANN lose'], 'r')
    plt.axis([0, len(x), 0, training_episodes])
    plt.show()
    # env.save(filename=filename)


if __name__ == '__main__':
    mnk()
