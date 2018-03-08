# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:23:38 2018

@author: Sadjad
"""

from agents import RLAgent, RandomAgent
from subjects import MNKGame
from environments import Environment
import time
import matplotlib.pyplot as plt


def main():
    m = 3
    n = 3
    k = 3
    filename='mnk' + str(m) + str(n) + str(k)
    RND = RandomAgent()
    try:
        env = Environment(filename=filename)
        print('{}: loaded'.format(time.ctime()))
    except (ModuleNotFoundError, FileNotFoundError):
        env = Environment()
        RLS1 = RLAgent(gamma=1, alpha=0.2, epsilon=0.3, Rplus=0, Ne=0)
        RLS2 = RLAgent(gamma=1, alpha=0.2, epsilon=0.3, Rplus=0, Ne=0)
        agents = {'RLS 1': RLS1, 'RLS 2': RLS2}
        subjects = {'board RLS': MNKGame(m=m, n=n, k=k)}
        assignment = [('RLS 1', 'board RLS'), ('RLS 2', 'board RLS')]

        env.add_agent(name_agent_pair=agents)
        env.add_subject(name_subject_pair=subjects)
        env.assign(assignment)

    runs = 50
    training_episodes = 100
    test_episodes = 100
    results = {'RLS 1': [], 'RLS 2': []}
    try:
        for i in range(runs):
            env.elapse(episodes=training_episodes, reset='all',
                       termination='all', learning_method='history',
                       reporting='none', tally='no')
            agent_temp = {'RLS 2': env._agent['RLS 2']}
            env._agent['RLS 2'] = RND
            tally = env.elapse(episodes=test_episodes, reset='all',
                               termination='all', learning_method='none',
                               reporting='none', tally='yes')
            for key in results:
                results[key].append(tally[key])
            state_count = len(env._agent['RLS 1']._state_action_list)
            Q = sum(s[0] for s in env._agent['RLS 1']._state_action_list.values())
            N = sum(s[1] for s in env._agent['RLS 1']._state_action_list.values())
            print('{}: run {: }: win 1: {: }, win 2: {: }, state: #: {: } N: {: }, Q: {: 4.1f}, per! N:{: 4.1f}, Q:{: 4.3f}'
                .format(time.ctime(), i, tally['RLS 1'], tally['RLS 2'], state_count, N, Q, N/state_count, Q/state_count))
            # print('{}: run {: }: win 1: {: }, win 2: {: }, state: #: {: }'
            #     .format(time.ctime(), i, tally['RLS 1'], tally['RLS 2'], state_count))
            env._agent['RLS 2'] = agent_temp['RLS 2']
            env.save(object_name='all', filename=filename)
            # print('saved!')
    except KeyboardInterrupt:
        pass
        
    plt.plot(results['RLS 1'])
    plt.axis([0, runs, 0, 100])
    plt.show()


if __name__ == '__main__':
    main()