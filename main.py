# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:47:46 2018

@author: Sadjad
"""

import matplotlib.pyplot as plt

from rl.agents import QAgent, TD0Agent, ANNAgent, RandomAgent, PGAgent
from rl.environments import Environment

def cancer(**kwargs):
    from rl.subjects import CancerModel

    # set experiment variables
    runs = kwargs.get('runs', 100)
    training_episodes = kwargs.get('training_episodes', 100)

    # load the environment or create a new one
    filename = kwargs.get('filename', 'cancer_case')
    try:
        env = Environment(filename=filename)
        agents = env._agent
        subjects = env._subject
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
        agents['Doctor'] = QAgent(gamma=0.7, alpha=0.2, epsilon=0.4)
        # agents['Doctor'] = ANNAgent(gamma=0.7, alpha=0.2, epsilon=0.5, learning_rate=1e-3, batch_size=10,
        #     default_actions=subjects['Patient'].possible_actions, input_length=40, hidden_layer_sizes=(10, 5))
        # agents['Doctor'].report(items=['states action'])
        # assign agents to subjects
        assignment = [('Doctor', 'Patient')]

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    # env._agent['Doctor'].data_collector.start()
    # env._agent['Doctor'].data_collector.collect(statistic=['diff-agent'])

    for i in range(runs):
        # run and collect statistics
        steps = env.elapse(episodes=training_episodes, max_steps=250, learning_method='history', step_count='yes')
        print(i, steps)
        # print(agents['Doctor'].data_collector.report(statistic=['diff-agent'], update_data=True))

        # save occasionally in case you don't lose data if you get bored of running the code!
        env.save(filename=filename)

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

def mnk(**kwargs):
    from rl.subjects import MNKGame

    # set experiment variables
    runs = kwargs.get('runs', 100)
    training_episodes = kwargs.get('training_episodes', 100)
    test_episodes = kwargs.get('test_episodes', 0)

    # load the environment or create a new one
    filename = kwargs.get('filename', 'mnk333')

    try:
        env = Environment(filename=filename)
        agents = env._agent
        subjects = env._subject
    except FileNotFoundError:
        env = Environment()
        # initialize dictionaries
        agents = {}
        subjects = {}

        # define subjects
        subjects['Board A'] = MNKGame(m=3, n=3, k=3)
        default_actions = subjects['Board A'].possible_actions

        # define agents
        # agents['TD'] = TD0Agent(gamma=1, alpha=0.2, epsilon=0.1)
        # agents['Q'] = QAgent(gamma=1, alpha=0.2, epsilon=0.1)
        # agents['ANN'] = ANNAgent(gamma=1, alpha=0.2, epsilon=0.1, hidden_layer_sizes=(26,4))
        # agents['Opponent'] = QAgent()
        agents['PG'] = PGAgent(gamma=1, alpha=0.2, epsilon=0.1, hidden_layer_sizes=(26,4),
                               default_actions=default_actions, state_size=len(subjects['Board A'].state.binary_representation()))
        agents['Opponent'] = RandomAgent()
        # test_agent = RandomAgent()
        # agents['Opponent'] = QAgent()
        # agents['Opponent'].load(filename='mnk333_opponent')
        # agents['Q'].report(items=['states action'])
        # assign agents to subjects
        assignment = [('PG', 'Board A'), ('Opponent', 'Board A')]

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    results = {'ANN training win': [], 'ANN training draw': [], 'ANN training lose': [],
               'ANN testing win': [], 'ANN testing draw': [], 'ANN testing lose': []}
    for i in range(runs):
        # run and collect statistics

        # if agents['ANN'].data_collector.is_active:
        #     agents['ANN'].data_collector.collect()

        tally1 = env.elapse(episodes=training_episodes, reset='all',
                            termination='all', learning_method='history',
                            reporting='none', tally='yes')
        # if agents['ANN'].data_collector.is_active:
        #     print(agents['ANN'].data_collector.report())
        # else:
        #     agents['ANN'].data_collector.start()

        # switch agents for test
        # temp = env._agent['Opponent']
        # env._agent['Opponent'] = test_agent
        # tally2 = env.elapse(episodes=test_episodes, reset='all',
        #                     termination='all', learning_method='none',
        #                     reporting='none', tally='yes')
        # env._agent['Opponent'] = temp

        results['ANN training win'].append(tally1['PG'])
        results['ANN training lose'].append(tally1['Opponent'])
        results['ANN training draw'].append(training_episodes-tally1['PG']-tally1['Opponent'])
        results['ANN testing win'].append(0)  # tally2['ANN'])
        results['ANN testing lose'].append(0)  # tally2['Opponent'])
        results['ANN testing draw'].append(0)  # test_episodes-tally2['ANN']-tally2['Opponent'])

        # # print result of each run
        print('run {: }: TRAINING: win: {: } draw:{: } lose:{: } TESTING: win: {: } draw:{: } lose:{: }'
              .format(i, results['ANN training win'][-1], results['ANN training draw'][-1], results['ANN training lose'][-1],
                    results['ANN testing win'][-1], results['ANN testing draw'][-1], results['ANN testing lose'][-1]))


        # # save occasionally in case you don't lose data if you get bored of running the code!
        env.save(filename=filename)

    x = list(range(len(results['ANN training win'])))
    plt.plot(x, results['ANN training win'], 'b', x, results['ANN training draw'], 'g', x, results['ANN training lose'], 'r')
    plt.axis([0, len(x), 0, training_episodes])
    plt.show()

def windy(**kwargs):
    from rl.subjects import WindyGridworld

    # set experiment variables
    runs = kwargs.get('runs', 100)
    training_episodes = kwargs.get('training_episodes', 100)
    test_episodes = kwargs.get('test_episodes', 0)
    max_steps = kwargs.get('max_steps', 10000)

    # load the environment or create a new one
    filename = kwargs.get('filename', 'windy')

    try:
        env = Environment(filename=filename)
        agents = env._agent
        subjects = env._subject
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

    results = {active_agent_name: []}
    steps1 = 0
    steps2 = 0
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
    env.save(filename=filename)

    # policy = agents[active_agent_name].data_collector.report(statistic=['states action'])['states action']
    # for state in sorted(policy.keys()):
    #     print(state.value, policy[state][0].value, policy[state][1])

    x = list(range(len(results[active_agent_name])))
    plt.plot(x, results[active_agent_name], 'b')
    plt.axis([0, len(x), 0, max(results[active_agent_name])])
    plt.show()

if __name__ == '__main__':
    model = 'cancer'
    filename = 'test'
    runs = 1
    training_episodes = 10
    function = {'windy': windy, 'mnk': mnk, 'cancer': cancer}
    function[model.lower()](filename=filename, runs=runs, training_episodes=training_episodes)
