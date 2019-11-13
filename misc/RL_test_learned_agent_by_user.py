# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:23:38 2018

@author: Sadjad
"""

from rl.agents import QAgent, UserAgent
from rl.environments import Environment
from rl.subjects import MNKGame


def main():
    filename='mnk333'
    User = UserAgent()
    env = Environment(filename=filename)

    test_episodes = 1
    results = {'RLS 1': [], 'RLS 2': []}
    env._agent['RLS 2'] = User
    try:
        tally = env.elapse(episodes=test_episodes, reset='all',
                            termination='all', learning_method='none',
                            reporting='all', tally='yes')
        for key in results:
            results[key].append(tally[key])
        state_count = len(env._agent['RLS 1']._state_action_list)
        Q = sum(s[0] for s in env._agent['RLS 1']._state_action_list.values())
        N = sum(s[1] for s in env._agent['RLS 1']._state_action_list.values())
        print(f'win 1: {tally['RLS 1']}, win 2: {tally['RLS 2']}, state: #: {state_count} N: {N}, Q: {Q: 4.1f}, per! N:{N/state_count: 4.1f}, Q:{Q/state_count: 4.3f}')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
