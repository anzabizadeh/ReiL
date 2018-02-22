# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:47:46 2018

@author: Sadjad
"""

from rlagent import RLAgent, RandomAgent, UserAgent
from mnkgame import MNKGame
import pickle
import time
import Environment

def main():
    env = Environment.Environment()
    try:
        env.load(filename='mnk333env-agentbyagent-testbyrandom')
    except FileNotFoundError:
        agents = {'a1':
                  RLAgent(gamma=1, alpha=0.5, epsilon=0.05, Rplus=1, Ne=2),
                  'a2':
                  RLAgent(gamma=1, alpha=0.5, epsilon=0.05, Rplus=1, Ne=2)}
        subjects = {'board a': MNKGame(m=3, n=3, k=3)}
        assignment = [('a1', 'board a'), ('a2', 'board a')]
        env.add_agent(name_agent_pair=agents)
        env.add_subject(name_subject_pair=subjects)
        env.assign(assignment)

    test_agent = RandomAgent()
#    print(len(env._agent['a1']._state_action_list),
#          len(env._agent['a2']._state_action_list))
    runs = 5
    training_episodes = 500
    test_episodes = 100
    results = {'a1': [], 'a2': []}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    li, = ax.plot([], [])
    ax.relim()
    ax.autoscale_view(True, True, True)
    fig.canvas.draw()
    plt.show(block=False)
    try:
        for i in range(runs):
            fig.canvas.draw()
            plt.draw_all()
            env.elapse(episodes=training_episodes, reset='all',
                       termination='any', learning_method='history',
                       reporting='none', tally='no')
            training_agent = env._agent['a2']
            env._agent['a2'] = test_agent
            tally = env.elapse(episodes=test_episodes, reset='all',
                               termination='any', learning_method='history',
                               reporting='none', tally='yes')
            results['a1'].append(tally['a1'])
            results['a2'].append(tally['a2'])
#            print('run {: }: no lose rate: {: f} win: {: } draw: {: } lose: {: }'
#                  .format(i, 1 - tally['a2']/test_episodes,
#                          tally['a1'],
#                          test_episodes - tally['a1'] - tally['a2'],
#                          tally['a2']))
            env._agent['a2'] = training_agent
            li.set_ydata(results['a1'])
            li.set_xdata(results['a1'])
            ax.relim()
            ax.autoscale_view(True, True, True)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass


#    print('\n', len(env._agent['a1']._state_action_list),
#          len(env._agent['a2']._state_action_list))

    env.save(object_name='all', filename='mnk333env-agentbyagent-testbyrandom')
#    env._agent['a2'] = UserAgent()
#    env.elapse(episodes=5, reset='all', termination='any',
#               learning_method='history', reporting='all')
