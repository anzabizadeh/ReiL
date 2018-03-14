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
    filename = 'mnk333_opponent_training'
    try:
        env = Environment(filename=filename)
        print('Load successful!')
    except FileNotFoundError:
        env = Environment()
        # initialize dictionaries
        agents = {}
        subjects = {}

        # define subjects
        subjects['Board A'] = MNKGame(m=3, n=3, k=3)

        # define agents
        agents['Random 1'] = RandomAgent()
        agents['Q'] = QAgent(gamma=1, alpha=0.2, epsilon=0.1)

        # assign agents to subjects
        assignment = [('Random 1', 'Board A'), ('Q', 'Board A')]

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    # set experiment variables
    runs = 500
    training_episodes = 200
    test_episodes = 100
    results = {'Q': []}
    for i in range(runs):
        # run and collect statistics
        env.elapse(episodes=training_episodes, reset='all',
                   termination='all', learning_method='history',
                   reporting='none', tally='no')

        tally = env.elapse(episodes=test_episodes, reset='all',
                            termination='all', learning_method='none',
                            reporting='none', tally='yes')

        results['Q'].append((training_episodes-tally['Random 1'])/training_episodes)

        # print result of each run
        print('run {: }: Win: {: 2.2f}, No Lose: {: 2.2f} state: {: }'
              .format(i, tally['Q']/training_episodes, results['Q'][-1], len(env._agent['Q']._state_action_list)))

        # save occasionally in case you don't lose data if you get bored of running the code!
        env.save(filename=filename)
        env.save(object_name='Q', filename='mnk333_opponent')

    x = list(range(len(results['Q'])))
    plt.plot(x, results['Q'], 'r')
    plt.axis([0, len(x), 0, 1])
    plt.show()


if __name__ == '__main__':
    main()
