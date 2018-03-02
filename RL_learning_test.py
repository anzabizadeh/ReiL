# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:23:38 2018

@author: Sadjad
"""

from agents import RLAgent, RandomAgent, UserAgent
from subjects import MNKGame
from environments import Environment
from matplotlib import pyplot as plt
import pickle
import time


def main():
    filename='mnk444'
    RND = RandomAgent()
    try:
        env = Environment(filename=filename)
        print('loaded')
    except (ModuleNotFoundError, FileNotFoundError):
        env = Environment()
        RLS1 = RLAgent(gamma=1, alpha=0.2, epsilon=0.3, Rplus=0, Ne=0)
        RLS2 = RLAgent(gamma=1, alpha=0.2, epsilon=0.3, Rplus=0, Ne=0)
        # RL1 = RLAgent(gamma=1, alpha=0.5, epsilon=0.1, Rplus=0.2, Ne=1)
        # RL2 = RLAgent(gamma=1, alpha=0.5, epsilon=0.1, Rplus=0.2, Ne=1)
        # RLR = RLAgent(gamma=1, alpha=0.5, epsilon=0.1, Rplus=0.2, Ne=1)
        agents = {'RLS 1': RLS1, 'RLS 2': RLS2}
                #   'RL 1': RL1, 'RL 2': RL2,
                #   'RLR': RLR, 'RND': RND}
        subjects = {'board RLS': MNKGame(m=4, n=4, k=4)}
                    # 'board RL': MNKGame(),
                    # 'board RND': MNKGame()}
        assignment = [('RLS 1', 'board RLS'), ('RLS 2', 'board RLS')]
                    #   ('RL 1', 'board RL'), ('RL 2', 'board RL'),
                    #   ('RLR', 'board RND'), ('RND', 'board RND')]

        env.add_agent(name_agent_pair=agents)
        env.add_subject(name_subject_pair=subjects)
        env.assign(assignment)

    runs = 1000
    training_episodes = 100
    test_episodes = 10
    results = {'RLS 1': [], 'RLS 2': []}
            #    'RL 1': [], 'RL 2': [],
            #    'RLR': [], 'RND': []}
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    li, = ax.plot([], [])
#    ax.relim()
#    ax.autoscale_view(True, True, True)
#    fig.canvas.draw()
#    plt.show(block=False)
    try:
        for i in range(runs):
        #    fig.canvas.draw()
        #    plt.draw_all()
        #    env._agent['a1'].status = 'training'
        #    env._agent['a2'].status = 'training'
            env.elapse(episodes=training_episodes, reset='all',
                       termination='all', learning_method='history',
                       reporting='none', tally='no')
            agent_temp = {'RLS 2': env._agent['RLS 2']}
                        #   'RL 2': env._agent['RL 2']}
            env._agent['RLS 2'] = RND
            # env._agent['RL 2'] = RND
#            env._agent['a1'].status = 'testing'
            tally = env.elapse(episodes=test_episodes, reset='all',
                               termination='all', learning_method='none',
                               reporting='none', tally='yes')
            for key in results:
                results[key].append(tally[key])
            state_count = len(env._agent['RLS 1']._state_action_list)
            Q = sum(s[0] for s in env._agent['RLS 1']._state_action_list.values())
            N = sum(s[1] for s in env._agent['RLS 1']._state_action_list.values())
            print('run {: }: win 1: {: }, win 2: {: }, state: #: {: } N: {: }, Q: {: 4.1f}, per! N:{: 4.1f}, Q:{: 4.3f}'
                .format(i, tally['RLS 1'], tally['RLS 2'], state_count, N, Q, N/state_count, Q/state_count))
            # print('run {: }: losing figures: RLS: {: }, RL: {: }, RLR: {: }'
            #       .format(i, tally['RLS 2'], tally['RL 2'], tally['RND']))
            env._agent['RLS 2'] = agent_temp['RLS 2']
            env.save(object_name='all', filename=filename)
            print('saved!')
            # env._agent['RL 2'] = agent_temp['RL 2']
        #    li.set_ydata(results['a1'])
        #    li.set_xdata(results['a1'])
        #    ax.relim()
        #    ax.autoscale_view(True, True, True)
        #    fig.canvas.draw()
        #    fig.canvas.flush_events()
        #    time.sleep(0.01)
    except KeyboardInterrupt:
        pass


#    print('\n', len(env._agent['a1']._state_action_list),
#          len(env._agent['a2']._state_action_list))
    print('saving...')
    env.save(object_name='all', filename=filename)
    with open('results.pkl', 'wb+') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
#    env._agent['a2'] = UserAgent()
#    env.elapse(episodes=5, reset='all', termination='any',
#               learning_method='history', reporting='all')


if __name__ == '__main__':
    main()