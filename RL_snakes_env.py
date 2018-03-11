# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:23:38 2018

@author: Sadjad
"""

import time

from rl.agents import QAgent, RandomAgent
from rl.environments import Environment
from rl.subjects import Snake


def main():
    filename='snake01'
    try:
        env = Environment(filename=filename)
        print('{}: loaded'.format(time.ctime()))
    except (ModuleNotFoundError, FileNotFoundError):
        env = Environment()
        RLS1 = QAgent(gamma=0.6, alpha=0.2, epsilon=0.3, Rplus=0, Ne=0, default_actions=['left', 'none', 'right'])
        agents = {'snake': RLS1}
        subjects = {'board': Snake(m=10, n=10)}
        assignment = [('snake', 'board')]

        env.add_agent(name_agent_pair=agents)
        env.add_subject(name_subject_pair=subjects)
        env.assign(assignment)

    runs = 1
    training_episodes = 200
    test_episodes = 10
    results = {'snake': []}
    try:
        for i in range(runs):
            env.elapse(episodes=training_episodes, reset='all',
                       termination='any', learning_method='every step',
                       reporting='none', tally='no')
            # agent_temp = {'RLS 2': env._agent['RLS 2']}
            # env._agent['RLS 2'] = RND
            # tally = env.elapse(episodes=test_episodes, reset='all',
            #                    termination='all', learning_method='none',
            #                    reporting='none', tally='yes')
            # for key in results:
            #     results[key].append(tally[key])
            # state_count = len(env._agent['RLS 1']._state_action_list)
            # Q = sum(s[0] for s in env._agent['RLS 1']._state_action_list.values())
            # N = sum(s[1] for s in env._agent['RLS 1']._state_action_list.values())
            # print('{}: run {: }: win 1: {: }, win 2: {: }, state: #: {: } N: {: }, Q: {: 4.1f}, per! N:{: 4.1f}, Q:{: 4.3f}'
            #     .format(time.ctime(), i, tally['RLS 1'], tally['RLS 2'], state_count, N, Q, N/state_count, Q/state_count))
            # env._agent['RLS 2'] = agent_temp['RLS 2']
            # env.save(object_name='all', filename=filename)
            # print('saved!')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
