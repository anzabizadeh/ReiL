# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:47:46 2018

@author: Sadjad
"""

import matplotlib.pyplot as plt
from random import randint, random

# , TD0Agent, RandomAgent, PGAgent
from rl.agents import QAgent, ANNAgent, WarfarinQAgent, DQNAgent
from rl.environments import Environment


def cancer(**kwargs):
    from rl.subjects import ConstrainedCancerModel

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
        # state_range = [0, 0.0063, 0.0125, 0.025, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
        state_range = [0, 0.0076, 0.016, 0.0252, 0.0352, 0.046, 0.0576, 0.07, 0.0832, 0.0972, 0.112, 0.1276, 0.144, 0.1612, 0.1792, 0.198,
                       0.2176, 0.238, 0.2592, 0.2812, 0.304, 0.3276, 0.352, 0.3772, 0.4032, 0.43, 0.4576, 0.486, 0.5152, 0.5452, 0.576, 0.6076, 0.64,
                       0.6732, 0.7072, 0.742, 0.7776, 0.814, 0.8512, 0.8892]

        class drug_cap:
            def __init__(self, minimum=0, maximum=10, probability=0.01):
                self._prob = probability
                self._min = minimum
                self._max = maximum
                self._cap = randint(self._min, self._max)

            def __call__(self, x):
                if random() < self._prob:
                    self._cap = randint(self._min, self._max)
                    # print('Changed cap to {} on day {}'.format(self._cap, x['day']))
                return self._cap

        # state_range = list(x/1000 for x in range(0,9000))
        subjects['Patient'] = ConstrainedCancerModel(
            drug={'initial_value': 0, 'decay_rate': 1,
                  'normal_cell_kill_rate': 0.1, 'tumor_cell_kill_rate': 0.3, 'immune_cell_kill_rate': 0.2},
            normal_cells={'initial_value': 0.40,
                          'growth_rate': 1, 'carrying_capacity': 1},
            tumor_cells={'initial_value': 1.00,
                         'growth_rate': 1.5, 'carrying_capacity': 1},
            immune_cells={'initial_value': 0, 'influx_rate': 0.33,
                          'threshold_rate': 0.3, 'response_rate': 0.01, 'death_rate': 0.2},
            competition_term={'normal_from_tumor': 1, 'tumor_from_normal': 1,
                              'tumor_from_immune': 0.5, 'immune_from_tumor': 1},
            state_function=lambda x: {'value': (state_range.index(max(filter(
                lambda y: y <= x['tumor_cells'], state_range))), x['drug_cap']), 'min': (0, 0), 'max': (len(state_range), 10)},
            # state_function = lambda x: {'value': (round(x['normal_cells'], 3), round(x['tumor_cells'], 3), round(x['drug'], 3)), 'min':(0, 0, 0), 'max': {2, 2, 10}},
            reward_function=lambda new_x, old_x: - \
            new_x['tumor_cells']-0.01*new_x['drug'],
            termination_check=lambda x: x['tumor_cells'] <= 1e-5, u_max=10, u_steps=20,
            drug_cap=drug_cap(probability=0.05))

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

    env._agent['Doctor'].data_collector.start()
    env._agent['Doctor'].data_collector.collect(statistic=['diff-q'])

    for i in range(runs):
        # run and collect statistics
        steps = env.elapse(episodes=training_episodes, max_steps=250,
                           learning_method='history', step_count='yes')
        print(i, steps)
        print(agents['Doctor'].data_collector.report(
            statistic=['diff-q'], update_data=True))

        # save occasionally in case you don't lose data if you get bored of running the code!
        env.save(filename=filename)
        print(sum(len(a)
                  for a in agents['Doctor']._state_action_list.values()))

    env._subject['Patient'].set_params(drug_cap=lambda x: 10)
    history = env.trajectory()
    states = list(s.value[0] for i, s in enumerate(
        history['Doctor']) if (i % 3) == 0)
    actions = list(a.value[0] for i, a in enumerate(
        history['Doctor']) if (i % 3) == 1)
    rewards = list(r for i, r in enumerate(history['Doctor']) if (i % 3) == 2)
    print('Total reward: {}, total drug: {}'.format(sum(rewards), sum(actions)))
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
        agents['ANN'] = ANNAgent(
            gamma=1, alpha=0.2, epsilon=0.1, hidden_layer_sizes=(26, 4))
        agents['Opponent'] = QAgent()
        # agents['PG'] = PGAgent(gamma=1, alpha=0.2, epsilon=0.1, hidden_layer_sizes=(26, 4),
        #                        default_actions=default_actions, state_size=len(subjects['Board A'].state.binary_representation()))
        # agents['Opponent'] = RandomAgent()
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
        results['ANN training draw'].append(
            training_episodes-tally1['PG']-tally1['Opponent'])
        results['ANN testing win'].append(0)  # tally2['ANN'])
        results['ANN testing lose'].append(0)  # tally2['Opponent'])
        # test_episodes-tally2['ANN']-tally2['Opponent'])
        results['ANN testing draw'].append(0)

        # # print result of each run
        print('run {: }: TRAINING: win: {: } draw:{: } lose:{: } TESTING: win: {: } draw:{: } lose:{: }'
              .format(i, results['ANN training win'][-1], results['ANN training draw'][-1], results['ANN training lose'][-1],
                      results['ANN testing win'][-1], results['ANN testing draw'][-1], results['ANN testing lose'][-1]))

        # # save occasionally in case you don't lose data if you get bored of running the code!
        env.save(filename=filename)

    x = list(range(len(results['ANN training win'])))
    plt.plot(x, results['ANN training win'], 'b', x,
             results['ANN training draw'], 'g', x, results['ANN training lose'], 'r')
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
        # ('Q', 'Board Q'), ('TD', 'Board TD')
        assignment = [('ANN', 'Board ANN')]

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    results = {active_agent_name: []}
    steps1 = 0
    steps2 = 0
    for i in range(runs):
        # run and collect statistics
        if agents[active_agent_name].data_collector.is_active:
            agents[active_agent_name].data_collector.collect(
                statistic=['diff-coef'])

        steps1 = env.elapse(episodes=training_episodes, max_steps=max_steps, reset='all', step_count='yes',
                            termination='all', learning_method='every step')
        # if steps1 < 200:
        #     print('testing')
        # steps2 = env.elapse(episodes=test_episodes, max_steps=1000, reset='all',
        #                     termination='all', learning_method='none', step_count='yes')
        if not agents[active_agent_name].data_collector.is_active:
            agents[active_agent_name].data_collector.start()
        else:
            results[active_agent_name].append(agents[active_agent_name].data_collector.report(
                statistic=['diff-coef'])['diff-coef'])

            # print result of each run
            print('{}: {} {: 3.10f}'.format(
                i, steps2, results[active_agent_name][-1]))

        # save occasionally in case you don't lose data if you get bored of running the code!
    env.save(filename=filename)

    # policy = agents[active_agent_name].data_collector.report(statistic=['states action'])['states action']
    # for state in sorted(policy.keys()):
    #     print(state.value, policy[state][0].value, policy[state][1])

    x = list(range(len(results[active_agent_name])))
    plt.plot(x, results[active_agent_name], 'b')
    plt.axis([0, len(x), 0, max(results[active_agent_name])])
    plt.show()


def risk(**kwargs):
    from rl.subjects import Risk

    # set experiment variables
    runs = kwargs.get('runs', 1000)
    training_episodes = kwargs.get('training_episodes', 100)
    test_episodes = kwargs.get('test_episodes', 100)

    # load the environment or create a new one
    filename = kwargs.get('filename', 'risk')

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
        subjects['Board A'] = Risk(pieces=[15, 15])
        # default_actions = subjects['Board A'].possible_actions

        # define agents
        agents['Q'] = QAgent(gamma=1, alpha=0.2, epsilon=0.1)
        agents['Opponent'] = QAgent(gamma=1, alpha=0.2, epsilon=0.1)
        # agents['Opponent'].load(filename='mnk333_opponent')

        # assign agents to subjects
        assignment = [('Q', 'Board A'), ('Opponent', 'Board A')]

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    results = {'ANN training win': [], 'ANN training draw': [], 'ANN training lose': [],
               'ANN testing win': [], 'ANN testing draw': [], 'ANN testing lose': []}
    agents['Q'].data_collector.start()
    agents['Opponent'].data_collector.start()
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

        results['ANN training win'].append(tally1['Q'])
        results['ANN training lose'].append(tally1['Opponent'])
        results['ANN training draw'].append(
            training_episodes-tally1['Q']-tally1['Opponent'])
        results['ANN testing win'].append(0)  # tally2['ANN'])
        results['ANN testing lose'].append(0)  # tally2['Opponent'])
        # test_episodes-tally2['ANN']-tally2['Opponent'])
        results['ANN testing draw'].append(0)

        # # print result of each run
        print('run {: }: TRAINING: win: {: } draw:{: } lose:{: } TESTING: win: {: } draw:{: } lose:{: }'
              .format(i, results['ANN training win'][-1], results['ANN training draw'][-1], results['ANN training lose'][-1],
                      results['ANN testing win'][-1], results['ANN testing draw'][-1], results['ANN testing lose'][-1]))

        # # save occasionally in case you don't lose data if you get bored of running the code!
        env.save(filename=filename)

    print('State-actions q:')
    print('Q:')
    sa = agents['Q'].data_collector.report(
        statistic=['state-actions q'])['state-actions q']
    for s in sorted(sa, reverse=True):
        print(s[0].value, s[1].value, sa[s])
    print('Opponent:')
    sa = agents['Opponent'].data_collector.report(
        statistic=['state-actions q'])['state-actions q']
    for s in sorted(sa, reverse=True):
        print(s[0].value, s[1].value, sa[s])

    print('States action:')
    print('Q:')
    sa = agents['Q'].data_collector.report(
        statistic=['states action'])['states action']
    for s in sorted(sa, reverse=True):
        print(s.value, sa[s][0].value, sa[s][1])
    print('Opponent:')
    sa = agents['Opponent'].data_collector.report(
        statistic=['states action'])['states action']
    for s in sorted(sa, reverse=True):
        print(s.value, sa[s][0].value, sa[s][1])


def warfarin(**kwargs):
    from rl.subjects.warfarin_model_v3 import WarfarinModel

    # set experiment variables
    runs = kwargs.get('runs', 1000)
    training_episodes = kwargs.get('training_episodes', 100)
    test_episodes = kwargs.get('test_episodes', 100)

    age = kwargs.get('age', 74)
    CYP2C9 = kwargs.get('CYP2C9', '*2/*2')
    VKORC1 = kwargs.get('VKORC1', 'G/A')
    max_day = kwargs.get('max_day', 90)
    dose_history = kwargs.get('dose_history', 10)
    INR_history = kwargs.get('INR_history', 10)
    patient_selection = kwargs.get('patient_selection', '')
    randomized = kwargs.get('randomized', True)

    gamma = kwargs.get('gamma', 0.99)
    alpha = kwargs.get('alpha', 0.2)
    epsilon = kwargs.get('epsilon', 0.1)

    agent_type = kwargs.get('agent_type', 'Q')

    if agent_type.lower() == 'warfarinq':
        method = kwargs.get('method', 'fixed policy first')
        if method == 'fixed policy first':
            fixed_policy_attempts = kwargs.get('fixed_policy_attempts', 30)
            text = '{:2}'.format(fixed_policy_attempts)

    elif agent_type.lower() in ['ann', 'dqn']:
        learning_rate = kwargs.get('learning_rate', 1e-2)
        buffer_size = kwargs.get('buffer_size', 5000)
        batch_size = kwargs.get('batch_size', 1000)
        clear_buffer = kwargs.get('clear_buffer', False)
        input_length = kwargs.get('input_length', 79)
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (20, 20))
        validation_split = kwargs.get('validation_split', 0.3)
        text = '_'.join((str(hidden_layer_sizes),
                            'g', str(gamma),
                            'e', 'func' if callable(epsilon) else str(epsilon),
                            'lr', str(learning_rate),
                            'buff', str(buffer_size),
                            'clr', 'T' if clear_buffer else 'F',
                            'btch', str(batch_size),
                            'vld', str(validation_split)))
    else:
        text = ''

    # load the environment or create a new one
    try:
        filename = kwargs['filename']
    except KeyError:
        filename = '_'.join(('WARF',
                             str(age) if patient_selection != 'random' else '00',
                             {'*1/*1': '11', '*1/*2': '12', '*1/*3': '13', '*2/*2': '22', '*2/*3': '23', '*3/*3': '33'}[
                                 CYP2C9] if patient_selection != 'random' else '00',
                             {'A/A': 'AA', 'A/G': 'AG', 'G/A': 'AG', 'G/G': 'GG'}[
                                 VKORC1] if patient_selection != 'random' else 'XX',
                             'd_{:2}'.format(max_day),
                             'dose_{:2}'.format(dose_history),
                             'INR_{:2}'.format(INR_history),
                             'T' if randomized else 'F',
                             agent_type, text)).replace(' ', '')

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
        subjects['W'] = WarfarinModel(age=age,
                                      CYP2C9=CYP2C9,
                                      VKORC1=VKORC1,
                                      TTR_range=(2, 3),
                                      d_max=kwargs.get('d_max', 1),
                                      max_day=max_day,
                                      patient_selection=patient_selection,
                                      dose_history=dose_history,
                                      INR_history=INR_history,
                                      randomized=randomized)
        # define agents
        if agent_type.lower() == 'q':
            agents['protocol'] = QAgent(gamma=gamma,
                                        alpha=alpha,
                                        epsilon=epsilon)

        elif agent_type.lower() == 'warfarinq':
            agents['protocol'] = WarfarinQAgent(gamma=gamma,
                                                alpha=alpha,
                                                epsilon=epsilon,
                                                method=method,
                                                fixed_policy_attempts=fixed_policy_attempts,
                                                default_actions=subjects['W'].possible_actions)
        elif agent_type.lower() == 'ann':
            agents['protocol'] = ANNAgent(gamma=gamma,
                                          epsilon=epsilon,
                                          learning_rate=learning_rate,
                                          buffer_size=buffer_size,
                                          batch_size=batch_size,
                                          input_length=input_length,
                                          hidden_layer_sizes=hidden_layer_sizes,
                                          default_actions=subjects['W'].possible_actions)
        elif agent_type.lower() == 'dqn':
            agents['protocol'] = DQNAgent(gamma=gamma,
                                          epsilon=epsilon,
                                          learning_rate=learning_rate,
                                          buffer_size=buffer_size,
                                          clear_buffer=clear_buffer,
                                          batch_size=batch_size,
                                          input_length=input_length,
                                          validation_split=validation_split,
                                          hidden_layer_sizes=hidden_layer_sizes,
                                          default_actions=subjects['W'].possible_actions,
                                          tensorboard_path=filename)

        # assign agents to subjects
        assignment = [('protocol', 'W')]

        # update environment
        env.add(agents=agents, subjects=subjects)
        env.assign(assignment)

    # agents['protocol'].data_collector.start()
    for i in range(runs):
        print('run {: }'.format(i))
        env.elapse(episodes=training_episodes, reset='all',
                   termination='all', learning_method='history',
                   reporting='none', tally='no')

        # # save occasionally in case you don't lose data if you get bored of running the code!
        env.save(filename=filename)

        for row in env.trajectory()['protocol'].iterrows():
            # print('{}, {} \n {}'.format(row[0], row[1].state.value.loc['Doses'], row[1].reward))
            print('{}, {} \n {} \n {}'.format(row[0], row[1].state.value.loc['Doses'], row[1].state.value.loc['INRs'], row[1].reward))


def warfarin_results(**kwargs):
    # load the environment or create a new one
    filename = kwargs.get('filename', 'warfarin')

    env = Environment(filename=filename)
    # agents = env._agent
    subjects = env._subject

    for age in [74]:  # 70, 75, 80, 85]:
        # '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3']:
        for CYP2C9 in ['*2/*2']:
            for VKORC1 in ['G/A']:  # 'G/G', 'G/A', 'A/A']:
                subjects['W'].set_params(
                    age=age, CYP2C9=CYP2C9, VKORC1=VKORC1, patient_selection='')
                subjects['W'].reset()
                for t in env.trajectory().values():
                    for i, v in enumerate(t):
                        if (i+1) % 3 == 0:
                            print(v)
                        else:
                            print(v.value, end='\t')


if __name__ == '__main__':
    model = 'warfarin'
    # filename = 'WARF_74_22_GA_days90_hist10_DQN20x20'
    # filename = 'WARF_74_22_GA_days90_hist10_DQN10x10'
    runs = 100
    training_episodes = 100
    function = {'windy': windy, 'mnk': mnk, 'cancer': cancer, 'risk': risk,
                'warfarin': warfarin, 'warfarin_results': warfarin_results}
    function[model.lower()](runs=runs,
                            training_episodes=training_episodes,
                            randomized=True,
                            patient_selection='random',
                            # age=70,
                            # CYP2C9='*1/*3',
                            # VKORC1='A/A',
                            max_day=90,
                            dose_history=9,
                            INR_history=9,
                            gamma=0.95,
                            epsilon=lambda x: 1/(1+x/100),
                            agent_type='DQN',  # 'ANN',
                            # alpha=0.2,
                            input_length=30,
                            buffer_size=90*10,
                            batch_size=50,
                            validation_split=0.3,
                            hidden_layer_sizes=(20, 20),
                            clear_buffer=False
                            )
