# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:47:46 2018

@author: Sadjad
"""

import matplotlib.pyplot as plt

from rl.agents import QAgent, TD0Agent, ANNAgent
from rl.environments import Environment
from rl.subjects import MNKGame, WindyGridworld


def mnk():
    # load the environment or create a new one
    filename = 'mnk333_ANN___'
    try:
        env = Environment(filename=filename)
    except FileNotFoundError:
        env = Environment()
        # initialize dictionaries
        agents = {}
        subjects = {}

        # define subjects
        subjects['Board A'] = MNKGame(m=3, n=3, k=3)

        # define agents
        # agents['TD'] = TD0Agent(gamma=1, alpha=0.2, epsilon=0.1)
        # agents['Q'] = QAgent(gamma=1, alpha=0.2, epsilon=0.1)
        agents['ANN'] = ANNAgent(gamma=1, alpha=0.2, epsilon=0.1, hidden_layer_sizes=(20,))
        agents['Opponent'] = QAgent()
        agents['Opponent'].load(filename='mnk333_opponent')
        agents['Opponent'].report(items=['states action'])
        # assign agents to subjects
        assignment = [('ANN', 'Board A'), ('Opponent', 'Board A')]

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    # set experiment variables
    runs = 50
    training_episodes = 100
    test_episodes = 10
    results = {'ANN win': [], 'ANN lose': []}
    for i in range(runs):
        # run and collect statistics
        env.elapse(episodes=training_episodes, reset='all',
                   termination='all', learning_method='history',
                   reporting='none', tally='no')

        tally = env.elapse(episodes=test_episodes, reset='all',
                            termination='all', learning_method='none',
                            reporting='none', tally='yes')

        results['ANN win'].append(tally['ANN'])
        results['ANN lose'].append(tally['Opponent'])

        # print result of each run
        print('run {: }: win: {: } draw:{: } lose:{: }'
              .format(i, results['ANN win'][-1], test_episodes-results['ANN win'][-1]-results['ANN lose'][-1], results['ANN lose'][-1]))


        # save occasionally in case you don't lose data if you get bored of running the code!
        env.save(filename=filename)

    x = list(range(len(results['ANN win'])))
    plt.plot(x, results['ANN win'], 'b', x, results['ANN lose'], 'r')
    plt.axis([0, len(x), 0, 1])
    plt.show()

def windy():
    # load the environment or create a new one
    filename = 'windy'
    try:
        env = Environment(filename=filename)
    except FileNotFoundError:
        env = Environment()
        # initialize dictionaries
        agents = {}
        subjects = {}

        # define subjects
        # subjects['Board Q'] = WindyGridworld(dim=(7, 10), start=(3, 0), goal=(3, 7), move_pattern='R',
        #                                     h_wind=[0]*7, v_wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        subjects['Board TD'] = WindyGridworld(dim=(7, 10), start=(3, 0), goal=(3, 7), move_pattern='R',
                                            h_wind=[0]*7, v_wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

        # define agents
        agents['TD'] = TD0Agent(gamma=0.9, alpha=0.2, epsilon=0.1)
        # agents['Q'] = QAgent(gamma=0.9, alpha=0.2, epsilon=0.1)
        # agents['ANN'] = ANNAgent(gamma=1, alpha=0.2, epsilon=0.1, hidden_layer_sizes=(20,))
        # agents['Opponent'] = QAgent()
        # agents['Opponent'].load(filename='mnk333_opponent')

        # assign agents to subjects
        assignment = [('TD', 'Board TD')] # ('Q', 'Board Q'), 

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    # set experiment variables
    runs = 100
    training_episodes = 10
    test_episodes = 1
    results = {'Q': []}
    steps1, steps2 = 0, 0
    for i in range(runs):
        # run and collect statistics
        steps1 = env.elapse(episodes=training_episodes, reset='all', step_count='yes',
                            termination='all', learning_method='every step')
        # if steps1 < 200:
        #     print('testing')
        # steps2 = env.elapse(episodes=test_episodes, reset='all',
        #                     termination='all', learning_method='none', step_count='yes')
        results['Q'].append(steps1)

        # print result of each run
        print('run {: }: training steps: {: 2.2f} testing steps: {: 2.2f}'.format(i, steps1, steps2))

        # save occasionally in case you don't lose data if you get bored of running the code!
    env.save(filename=filename)

    x = list(range(len(results['Q'])))
    plt.plot(x, results['Q'], 'b')
    plt.axis([0, len(x), 0, max(results['Q'])])
    plt.show()

if __name__ == '__main__':
    mnk()
