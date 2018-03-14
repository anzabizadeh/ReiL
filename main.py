# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:47:46 2018

@author: Sadjad
"""

import matplotlib.pyplot as plt

from rl.agents import QAgent, RandomAgent, TD0Agent, ANNAgent
from rl.environments import Environment
from rl.subjects import MNKGame


def main():
    # load the environment or create a new one
    filename = 'mnk333_ANN_new'
    try:
        env = Environment(filename=filename)
    except FileNotFoundError:
        env = Environment()
        # initialize dictionaries
        agents = {}
        subjects = {}

        # define subjects
        subjects['Board A'] = MNKGame(m=3, n=3, k=3)
        # subjects['Board B'] = MNKGame(m=3, n=3, k=3)

        # define agents
        # agents['TD'] = TD0Agent(gamma=1, alpha=0.2, epsilon=0.1)
        # agents['Q'] = QAgent(gamma=1, alpha=0.2, epsilon=0.1)
        agents['ANN'] = ANNAgent(gamma=1, alpha=0.2, epsilon=0.1,
                                 hidden_layer_sizes=(9, 9, 3), default_actions=subjects['Board A'].possible_actions)
        agents['Random 1'] = RandomAgent()
        # agents['Random 2'] = agents['Random 1']

        # assign agents to subjects
        assignment = [('ANN', 'Board A'), ('Random 1', 'Board A')]

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    # set experiment variables
    runs = 50
    training_episodes = 1000
    results = {'ANN win': [], 'ANN draw': []}
    for i in range(runs):
        # run and collect statistics
        tally = env.elapse(episodes=training_episodes, reset='all',
                            termination='all', learning_method='history',
                            reporting='none', tally='yes')
        results['ANN win'].append(tally['ANN']/training_episodes)
        results['ANN draw'].append((training_episodes-tally['Random 1']-tally['ANN'])/training_episodes)

        # print result of each run
        print('run {: }: ANN win: {: 2.2f} draw: {: 2.2f} no lose: {: 2.2f}'
                .format(i, results['ANN win'][-1], results['ANN draw'][-1],
                        results['ANN win'][-1] + results['ANN draw'][-1]))

        # save occasionally in case you don't lose data if you get bored of running the code!
        env.save(filename=filename)

    x = list(range(len(results['ANN win'])))
    plt.plot(x, results['ANN win'], 'r', x, results['ANN draw'], 'b')
    plt.axis([0, len(x), 0, 1])
    plt.show()


if __name__ == '__main__':
    main()
