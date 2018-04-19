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

    test_episodes = 100
    runs = 10

    m, n, k = 3, 3, 3

    number_of_scenarios = 26

    # initialize dictionaries
    environments = []
    results = []
    for i in range(number_of_scenarios):
        environments.append(Environment(path='./ann on GPU', filename='scenario_'+str(i)))
        environments[i]._assignment_list = {}
        environments[i].assign([('ANN', 'Board'), ('Opponent', 'Board')])
        # subjects = {'Board': MNKGame(m=m, n=n, k=k)}
        # agents = {'ANN': ANNAgent(), 'Opponent': RandomAgent()}
        # agents['ANN'].load(filename='ANN')
        # assignment = [('ANN', 'Board'), ('Opponent', 'Board')]
        # environments[i].add(agents=agents, subjects=subjects)
        # environments[i].assign(assignment)
        results.append({'ANN win': [], 'ANN draw': [], 'ANN lose': []})

    print('\nrun \tscen. \twin \tdraw \tlose')
    for run in range(runs):
        # run and collect statistics
        tally = [{}]*number_of_scenarios
        for i, env in enumerate(environments):
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
            env.save(path='./ann on GPU', filename='scenario_'+str(i))

if __name__ == '__main__':
    mnk()
