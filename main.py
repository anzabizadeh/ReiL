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
    filename = 'mnk333_with_report'
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
        # agents['Q'].report(items=['states action'])
        # assign agents to subjects
        assignment = [('ANN', 'Board A'), ('Opponent', 'Board A')]

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    # set experiment variables
    runs = 50
    training_episodes = 10
    test_episodes = 10
    results = {'ANN win': [], 'ANN lose': []}
    for i in range(runs):
        # run and collect statistics

        if agents['ANN'].data_collector.is_active:
            agents['ANN'].data_collector.collect()

        env.elapse(episodes=training_episodes, reset='all',
                   termination='all', learning_method='history',
                   reporting='none', tally='no')
        if agents['ANN'].data_collector.is_active:
            print(agents['ANN'].data_collector.report())
        else:
            agents['ANN'].data_collector.start()

        # tally = env.elapse(episodes=test_episodes, reset='all',
        #                     termination='all', learning_method='none',
        #                     reporting='none', tally='yes')

        # results['ANN win'].append(tally['ANN'])
        # results['ANN lose'].append(tally['Opponent'])

        # # print result of each run
        # print('run {: }: win: {: } draw:{: } lose:{: }'
        #       .format(i, results['ANN win'][-1], test_episodes-results['ANN win'][-1]-results['ANN lose'][-1], results['ANN lose'][-1]))


        # # save occasionally in case you don't lose data if you get bored of running the code!
        # env.save(filename=filename)

    x = list(range(len(results['ANN win'])))
    plt.plot(x, results['ANN win'], 'b', x, results['ANN lose'], 'r')
    plt.axis([0, len(x), 0, 1])
    plt.show()

def windy():
    # load the environment or create a new one
    filename = 'windy_2'
    try:
        env = Environment(filename=filename)
    except FileNotFoundError:
        env = Environment()
        # initialize dictionaries
        agents = {}
        subjects = {}

        active_agent_name = 'ANN'
        # define subjects
        # subjects['Board Q'] = WindyGridworld(dim=(7, 10), start=(3, 0), goal=(3, 7), move_pattern='R',
        #                                     h_wind=[0]*7, v_wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        # subjects['Board TD'] = WindyGridworld(dim=(7, 10), start=(3, 0), goal=(3, 7), move_pattern='R',
        #                                     h_wind=[0]*7, v_wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        subjects['Board ANN'] = WindyGridworld(dim=(7, 10), start=(3, 0), goal=(3, 7), move_pattern='R',
                                               h_wind=[0]*7, v_wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

        # define agents
        # agents['TD'] = TD0Agent(gamma=0.9, alpha=0.2, epsilon=0.1)
        # agents['Q'] = QAgent(gamma=0.9, alpha=0.2, epsilon=0.1)
        agents['ANN'] = ANNAgent(gamma=1, alpha=0.2, epsilon=0.1, hidden_layer_sizes=(70, 10, 4),
                                 default_actions=subjects['Board ANN'].possible_actions)
        # agents['Opponent'] = QAgent()
        # agents['Opponent'].load(filename='mnk333_opponent')

        # assign agents to subjects
        assignment = [('ANN', 'Board ANN')] # ('Q', 'Board Q'), ('TD', 'Board TD')

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    # set experiment variables
    runs = 1
    training_episodes = 5
    max_steps = 10000
    test_episodes = 1
    results = {active_agent_name: []}
    steps1, steps2 = 0, 0
    for i in range(runs):
        # run and collect statistics
        if agents[active_agent_name].data_collector.is_active:
            agents[active_agent_name].data_collector.collect(statistic=['diff-coef'])

        steps1 = env.elapse(episodes=training_episodes, max_steps=max_steps, reset='all', step_count='yes',
                            termination='all', learning_method='every step')
        # if steps1 < 200:
        #     print('testing')
        # steps2 = env.elapse(episodes=test_episodes, max_steps=1000, reset='all',
        #                     termination='all', learning_method='none', step_count='yes')
        if not agents[active_agent_name].data_collector.is_active:
            agents[active_agent_name].data_collector.start()
        else:
            results[active_agent_name].append(agents[active_agent_name].data_collector.report(statistic=['diff-coef'])['diff-coef'])

            # print result of each run
            print('{}: {} {: 3.10f}'.format(i, steps2, results[active_agent_name][-1]))

        # save occasionally in case you don't lose data if you get bored of running the code!
    # env.save(filename=filename)

    # policy = agents[active_agent_name].data_collector.report(statistic=['states action'])['states action']
    # for state in sorted(policy.keys()):
    #     print(state.value, policy[state][0].value, policy[state][1])

    x = list(range(len(results[active_agent_name])))
    plt.plot(x, results[active_agent_name], 'b')
    plt.axis([0, len(x), 0, max(results[active_agent_name])])
    plt.show()

if __name__ == '__main__':
    windy()
