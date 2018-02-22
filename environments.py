# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:23:38 2018

@author: Sadjad
"""

from agents import RLAgent, RandomAgent, UserAgent
from subjects import MNKGame
from matplotlib import pyplot as plt
import pickle
import time


def main():
    filename='several agents'
    RND = RandomAgent()
    try:
        env = Environment(filename=filename)
    except (ModuleNotFoundError, FileNotFoundError) as error:
        env = Environment()
        RLS = RLAgent(gamma=1, alpha=0.5, epsilon=0.1, Rplus=0.2, Ne=1)
        RL1 = RLAgent(gamma=1, alpha=0.5, epsilon=0.1, Rplus=0.2, Ne=1)
        RL2 = RLAgent(gamma=1, alpha=0.5, epsilon=0.1, Rplus=0.2, Ne=1)
        RLR = RLAgent(gamma=1, alpha=0.5, epsilon=0.1, Rplus=0.2, Ne=1)
        agents = {'RLS 1': RLS, 'RLS 2': RLS,
                  'RL 1': RL1, 'RL 2': RL2,
                  'RLR': RLR, 'RND': RND}
        subjects = {'board RLS': MNKGame(),
                    'board RL': MNKGame(),
                    'board RND': MNKGame()}
        assignment = [('RLS 1', 'board RLS'), ('RLS 2', 'board RLS'),
                      ('RL 1', 'board RL'), ('RL 2', 'board RL'),
                      ('RLR', 'board RND'), ('RND', 'board RND')]
        env.add_agent(name_agent_pair=agents)
        env.add_subject(name_subject_pair=subjects)
        env.assign(assignment)

#    test_agent = RandomAgent()
#    print(len(env._agent['a1']._state_action_list),
#          len(env._agent['a2']._state_action_list))
    runs = 10000000
    training_episodes = 1000
    test_episodes = 100
    results = {'RLS 1': [], 'RLS 2': [],
               'RL 1': [], 'RL 2': [],
               'RLR': [], 'RND': []}
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    li, = ax.plot([], [])
#    ax.relim()
#    ax.autoscale_view(True, True, True)
#    fig.canvas.draw()
#    plt.show(block=False)
    try:
        for i in range(runs):
#            fig.canvas.draw()
#            plt.draw_all()
#            env._agent['a1'].status = 'training'
#            env._agent['a2'].status = 'training'
            env.elapse(episodes=training_episodes, reset='all',
                       termination='all', learning_method='history',
                       reporting='none', tally='no')
            agent_temp = {'RLS 2': env._agent['RLS 2'],
                          'RL 2': env._agent['RL 2']}
            env._agent['RLS 2'] = RND
            env._agent['RL 2'] = RND
#            env._agent['a1'].status = 'testing'
            tally = env.elapse(episodes=test_episodes, reset='all',
                               termination='all', learning_method='none',
                               reporting='none', tally='yes')
            for key in results:
                results[key].append(tally[key])
            print('run {: }: losing figures: RLS: {: }, RL: {: }, RLR: {: }'
                  .format(i, tally['RLS 2'], tally['RL 2'], tally['RND']))
            env._agent['RLS 2'] = agent_temp['RLS 2']
            env._agent['RL 2'] = agent_temp['RL 2']
#            li.set_ydata(results['a1'])
#            li.set_xdata(results['a1'])
#            ax.relim()
#            ax.autoscale_view(True, True, True)
#            fig.canvas.draw()
#            fig.canvas.flush_events()
#            time.sleep(0.01)
    except KeyboardInterrupt:
        pass


#    print('\n', len(env._agent['a1']._state_action_list),
#          len(env._agent['a2']._state_action_list))

    env.save(object_name='all', filename=filename)
    with open('results.pkl', 'wb+') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
#    env._agent['a2'] = UserAgent()
#    env.elapse(episodes=5, reset='all', termination='any',
#               learning_method='history', reporting='all')


class Environment:
    def __init__(self, **kwargs):
        if 'filename' in kwargs:
            self.load(filename=kwargs['filename'])
            return

        self._agent = {}
        self._subject = {}
        self._assignment_list = {}
        # one episode is one full game till the end
        try:  # episodes
            self._default_episodes = kwargs['episodes']
        except KeyError:
            self._default_episodes = 1
        # termination: 'any' (terminate by any subject being terminated)
        #              'all' (terminate only if all subjects are terminated)
        try:  # termination
            self._termination = kwargs['termination']
        except KeyError:
            self._termination = 'any'
        # reset: 'any' (reset only subjects that are terminated)
        #        'all' (reset all subjects at each episode)
        try:  # reset
            self._reset = kwargs['reset']
        except KeyError:
            self._reset = 'any'

        # learning_method: 'every-step' (learns after every move)
        #                  'history' (learns after each episode)
        try:  # learning_method
            self._learning_method = kwargs['learning_method']
        except KeyError:
            self._learning_method = 'every step'

    def load(self, **kwargs):
        try:  # object_name
            object_name = kwargs['object_name']
        except KeyError:
            object_name = 'all'
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the input file not specified.')

        if object_name == 'all':
            with open(filename + '.pkl', 'rb') as f:
                self.__dict__ = pickle.load(f)
            for agent in self._agent:
                self._agent[agent].reset()
            for subject in self._subject:
                self._subject[subject].reset()
        elif object_name in self._agent:
            self._agent[object_name].save(filename)
            for agent in self._agent:
                self._agent[agent].reset()
        elif object_name in self._subject:
            self._subject[object_name].save(filename)
            for subject in self._subject:
                self._subject[subject].reset()

    def save(self, **kwargs):
        try:  # object_name
            object_name = kwargs['object_name']
        except KeyError:
            object_name = 'all'
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')

        if object_name == 'all':
            with open(filename + '.pkl', 'wb+') as f:
                pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        elif object_name in self._agent:
            self._agent[object_name].save(filename)
        elif object_name in self._subject:
            self._subject[object_name].save(filename)

    def add_agent(self, name_agent_pair):
        for name, agent in name_agent_pair.items():
            self._agent[name] = agent

    def remove_agent(self, agent_names):
        for name in agent_names:
            try:
                del self._agent[name]
            except KeyError:
                raise KeyError('Agent '+name+' not found')

    def add_subject(self, name_subject_pair):
        for name, subject in name_subject_pair.items():
            self._subject[name] = subject

    def remove_subject(self, subject_names):
        for name in subject_names:
            try:
                del self._subject[name]
            except KeyError:
                raise KeyError('Subject '+name+' not found')

    def assign(self, agent_subject_names):
        for agent_name, subject_name in agent_subject_names:
            if agent_name not in self._agent:
                raise ValueError('Agent ' + agent_name + ' not found.')
            if subject_name not in self._subject:
                raise ValueError('Subject ' + subject_name + ' not found.')
            _id = self._subject[subject_name].register(agent_name)
            self._assignment_list[agent_name] = (subject_name, _id)

    def elapse(self, **kwargs):
        # one episode is one full game till the end
        try:  # episodes
            episodes = kwargs['episodes']
        except KeyError:
            episodes = self._default_episodes
        # termination: 'any' (terminate by any subject being terminated)
        #              'all' (terminate only if all subjects are terminated)
        try:  # termination
            termination = kwargs['termination'].lower()
        except KeyError:
            termination = self._termination
        # reset: 'any' (reset only subjects that are terminated)
        #        'all' (reset all subjects at each episode)
        try:  # reset
            reset = kwargs['reset']
        except KeyError:
            reset = self._reset
        # learning_method: 'every step' (learns after every move)
        #                  'history' (learns after each episode)
        try:  # learning_method
            learning_method = kwargs['learning_method'].lower()
        except KeyError:
            learning_method = self._learning_method
        # reporting: 'all' (report every move)
        #            'none' (report nothing)
        #            'important' (report important parts only)
        try:  # reporting
            reporting = kwargs['reporting'].lower()
        except KeyError:
            reporting = 'none'

        try:  # tally
            if kwargs['tally'].lower() == 'yes':
                tally = True
                win_count = {}
                for agent in self._agent:
                    win_count[agent] = 0
            else:
                tally = False
        except KeyError:
            tally = False

        if learning_method == 'none':
            for agent in self._agent.values():
                agent.status = 'testing'
        else:
            for agent in self._agent.values():
                agent.status = 'training'

        for episode in range(episodes):
            if reporting != 'none':
                report_string = 'episode: {}'.format(episode+1)
            if learning_method == 'history':
                history = {}
                for agent_name in self._agent:
                    history[agent_name] = []
            done = False
            while not done:
                for agent_name, agent in self._agent.items():
                    if not done:
                        subject_name, _id = self._assignment_list[agent_name]
                        subject = self._subject[subject_name]
                        if not subject.is_terminated:
                            state = subject.state.copy()
                            possible_actions = subject.possible_actions
                            action = agent.act(state, actions=possible_actions,
                                               printable=subject.printable())
                            reward = subject.take_effect(_id, action)
                            if reporting == 'all':
#                                report_string += '\nagent {} plays {} on \
#                                                  subject {} and gets {}'.format(
#                                                  agent_name, action,
#                                                  subject_name, reward)
                                report_string += '\n'+subject.printable()+'\n'
                            if tally & (reward>0) & (subject.is_terminated):
                                win_count[agent_name] += 1

                            if learning_method == 'every step':
                                agent.learn(state=subject.state, reward=reward)
                            elif learning_method == 'history':
                                history[agent_name].append(state)
                                history[agent_name].append(action)
                                history[agent_name].append(reward)
                    if termination == 'all':
                        done = True
                        for sub in self._subject.values():
                            done = done & sub.is_terminated
                    elif termination == 'any':
                        done = False
                        for sub in self._subject.values():
                            done = done | sub.is_terminated

            if learning_method == 'history':
                for agent_name, agent in self._agent.items():
                    agent.learn(history=history[agent_name])

            if reset == 'all':
                for agent in self._agent.values():
                    agent.reset()
                for subject in self._subject.values():
                    subject.reset()
            elif reset == 'any':
                for agent_name, subject_name in self._assignment_list.items():
                    if self._subject[subject_name[0]].is_terminated:
                        self._agent[agent_name].reset()
                for agent_name, subject_name in self._assignment_list.items():
                    if self._subject[subject_name[0]].is_terminated:
                        self._subject[subject_name[0]].reset()

            if tally & (reporting != 'none'):
                report_string += '\n tally:'
                for agent in self._agent:
                    report_string += '\n {} {}'.format(agent, win_count[agent])

            if reporting != 'none':
                print(report_string)

        if tally:
            return win_count


if __name__ == '__main__':
    main()
