# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:47:46 2018

@author: Sadjad
"""

import matplotlib.pyplot as plt

from rl.agents import QAgent, TD0Agent, ANNAgent
from rl.environments import Environment
from rl.subjects import MNKGame


def main():
    # load the environment or create a new one
    filename = 'mnk333_Q_TD_ANN_separate_testing'
    try:
        env = Environment(filename=filename)
    except FileNotFoundError:
        env = Environment()
        # initialize dictionaries
        agents = {}
        subjects = {}

        # define subjects
        subjects['Board A'] = MNKGame(m=3, n=3, k=3)
        subjects['Board B'] = MNKGame(m=3, n=3, k=3)
        subjects['Board C'] = MNKGame(m=3, n=3, k=3)

        # define agents
        agents['TD'] = TD0Agent(gamma=1, alpha=0.2, epsilon=0.1)
        agents['Q'] = QAgent(gamma=1, alpha=0.2, epsilon=0.1)
        agents['ANN'] = ANNAgent(gamma=1, alpha=0.2, epsilon=0.1,
                                 hidden_layer_sizes=(9, 9, 3), default_actions=subjects['Board A'].possible_actions)
        agents['Opponent 1'] = QAgent()
        agents['Opponent 1'].load(filename='mnk333_opponent')
        agents['Opponent 2'] = agents['Opponent 1']
        agents['Opponent 3'] = agents['Opponent 1']

        # assign agents to subjects
        assignment = [('TD', 'Board A'), ('Opponent 1', 'Board A'),
                      ('Q', 'Board B'), ('Opponent 2', 'Board B'),
                      ('ANN', 'Board C'), ('Opponent 3', 'Board C')]

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    # set experiment variables
    runs = 1
    training_episodes = 5
    test_episodes = 50
    results = {'TD': [], 'Q': [], 'ANN': []}
    for i in range(runs):
        # run and collect statistics
        env.elapse(episodes=training_episodes, reset='all',
                   termination='all', learning_method='history',
                   reporting='none', tally='no')

        tally = env.elapse(episodes=test_episodes, reset='all',
                            termination='all', learning_method='none',
                            reporting='none', tally='yes')

        results['TD'].append((training_episodes-tally['Opponent 1'])/training_episodes)
        results['Q'].append((training_episodes-tally['Opponent 2'])/training_episodes)
        results['ANN'].append((training_episodes-tally['Opponent 3'])/training_episodes)

        # print result of each run
        print('run {: }: TD: {: 2.2f} Q: {: 2.2f} ANN: {: 2.2f} TD size: {: } Q size: {: }'
                .format(i, results['TD'][-1], results['Q'][-1], results['ANN'][-1],
                        len(env._agent['TD']._state_action_list), len(env._agent['Q']._state_action_list)))

        # save occasionally in case you don't lose data if you get bored of running the code!
        env.save(filename=filename)

    x = list(range(len(results['ANN'])))
    plt.plot(x, results['TD'], 'r', x, results['Q'], 'b', x, results['ANN'], 'g')
    plt.axis([0, len(x), 0, 1])
    plt.show()


if __name__ == '__main__':
    main()
