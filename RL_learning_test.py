# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:23:38 2018

@author: Sadjad
"""

import time

import matplotlib.pyplot as plt

from rl.agents import ANNAgent, QAgent, RandomAgent
from rl.environments import Environment
from rl.subjects import MNKGame


def main():
    m = 3
    n = 3
    k = 3
    filename='mnk' + str(m) + str(n) + str(k) + '_ann'
    RND = RandomAgent()
    try:
        env = Environment(filename=filename)
        print('{}: loaded'.format(time.ctime()))
    except (ModuleNotFoundError, FileNotFoundError):
        env = Environment()
        board = MNKGame(m=m, n=n, k=k)
        RLS1 = ANNAgent(epsilon=0.05, hidden_layer_sizes=(100,),
                        default_actions=board.possible_actions, alpha=0.2)
        RLS2 = QAgent(gamma=1, alpha=0.2, epsilon=0.3, Rplus=0, Ne=0)
        # RLS2 = ANNAgent(hidden_layer_sizes=(20, 10), default_actions=board.possible_actions)
        # RLS1 = QAgent(gamma=1, alpha=0.2, epsilon=0.3, Rplus=0, Ne=0)
        agents = {'RLS 1': RLS1, 'RLS 2': RLS2}
        subjects = {'board RLS': board}
        assignment = [('RLS 1', 'board RLS'), ('RLS 2', 'board RLS')]

        env.add_agent(name_agent_pair=agents)
        env.add_subject(name_subject_pair=subjects)
        env.assign(assignment)

    runs = 100
    training_episodes = 10
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
            print('{}: run {: }: not lose 1: {: }'.format(time.ctime(), i, test_episodes - tally['RLS 2']))
            env._agent['RLS 2'] = agent_temp['RLS 2']
            env.save(object_name='all', filename=filename)
    except KeyboardInterrupt:
        print('caught Ctrl+C')
        
    plt.plot(list((test_episodes-r for r in results['RLS 2'])))
    plt.axis([0, runs, 0, 100])
    plt.show()


if __name__ == '__main__':
    main()
