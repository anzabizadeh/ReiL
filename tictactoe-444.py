# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:47:46 2018

@author: Sadjad
"""

from time import time
import matplotlib.pyplot as plt

from rl.agents import QAgent, TD0Agent, ANNAgent, RandomAgent
from rl.environments import Environment
from rl.subjects import MNKGame, WindyGridworld


def mnk():
    # load the environment or create a new one
    filename = 'mnk444'
    try:
        env = Environment(filename=filename)
    except FileNotFoundError:
        env = Environment()
        # initialize dictionaries
        agents = {}
        subjects = {}

        # define subjects
        subjects['Board A'] = MNKGame(m=4, n=4, k=4)

        # define agents
        # agents['TD'] = TD0Agent(gamma=1, alpha=0.2, epsilon=0.1)
        agents['Q'] = QAgent(gamma=1, alpha=0.2, epsilon=0.1)
        # agents['ANN'] = ANNAgent(gamma=1, alpha=0.2, epsilon=0.1, hidden_layer_sizes=(26,4))
        agents['Opponent'] = QAgent(gamma=1, alpha=0.2, epsilon=0.1)
        # agents['Opponent'] = QAgent()
        # agents['Opponent'].load(filename='mnk333_opponent')
        # agents['Q'].report(items=['states action'])
        # assign agents to subjects
        assignment = [('Q', 'Board A'), ('Opponent', 'Board A')]

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    # set experiment variables
    runs = 1
    training_episodes = 100
    test_episodes = 100
    test_agent = RandomAgent()
    results = {'Q training win': [], 'Q training draw': [], 'Q training lose': [],
               'Q testing win': [], 'Q testing draw': [], 'Q testing lose': []}
    # env._agent['Q'].data_collector.start()
    # env._agent['Q'].data_collector.collect(statistic=['diff-q'])

    for i in range(runs):
        # run and collect statistics
        tally1 = env.elapse(episodes=training_episodes, reset='all',
                            termination='all', learning_method='history',
                            reporting='none', tally='yes')
        # print(agents['Q'].data_collector.report(statistic=['diff-q'], update_data=True))

        # switch agents for test
        temp = env._agent['Opponent']
        env._agent['Opponent'] = test_agent
        tally2 = env.elapse(episodes=test_episodes, reset='all',
                            termination='all', learning_method='none',
                            reporting='none', tally='yes')
        env._agent['Opponent'] = temp

        results['Q training win'].append(tally1['Q'])
        results['Q training lose'].append(tally1['Opponent'])
        results['Q training draw'].append(training_episodes-tally1['Q']-tally1['Opponent'])
        results['Q testing win'].append(tally2['Q'])
        results['Q testing lose'].append(tally2['Opponent'])
        results['Q testing draw'].append(test_episodes-tally2['Q']-tally2['Opponent'])

        # # print result of each run
        print('{:} run {: }: TRAINING: win: {: } draw:{: } lose:{: } TESTING: win: {: } draw:{: } lose:{: }'
              .format(time(), i, results['Q training win'][-1], results['Q training draw'][-1], results['Q training lose'][-1],
                    results['Q testing win'][-1], results['Q testing draw'][-1], results['Q testing lose'][-1]))


        # # save occasionally in case you don't lose data if you get bored of running the code!
        env.save(filename=filename)

    x = list(range(len(results['Q training win'])))
    plt.plot(x, results['Q training win'], 'b', x, results['Q training draw'], 'g', x, results['Q training lose'], 'r')
    plt.axis([0, len(x), 0, training_episodes])
    plt.show()


if __name__ == '__main__':
    mnk()
