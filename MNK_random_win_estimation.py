# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:23:38 2018

@author: Sadjad
"""

from agents import RandomAgent
from subjects import MNKGame
from environments import Environment
import numpy as np

def main():
    env = Environment()
    agents = {'RLS 1': RandomAgent(), 'RLS 2': RandomAgent()}
    subjects = {'board RLS': MNKGame()}
    assignment = [('RLS 1', 'board RLS'), ('RLS 2', 'board RLS')]

    env.add_agent(name_agent_pair=agents)
    env.add_subject(name_subject_pair=subjects)
    env.assign(assignment)

    run = 100
    test_episodes = 100
    results = {'RLS 1': [], 'RLS 2': []}
    stats = {'RLS 1': [], 'RLS 2': []}
    try:
        for i in range(run):
            tally = env.elapse(episodes=test_episodes, reset='all',
                                termination='all', learning_method='none',
                                reporting='none', tally='yes')
            for key in results:
                results[key].append(tally[key])
            print('run {: }: win 1: {: }, win 2: {: }'.format(i, tally['RLS 1'], tally['RLS 2']))
    except KeyboardInterrupt:
        pass

    for key in results:
        stats[key] = [np.percentile(results[key], 25),
                      np.percentile(results[key], 50),
                      np.percentile(results[key], 75)]
        print(stats[key])


if __name__ == '__main__':
    main()