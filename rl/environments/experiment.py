# -*- coding: utf-8 -*-
'''
experiment class
=================

This `environment` class provides an experimentation environment for any reinforcement learning agent on any subject. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from math import log10, ceil
from pathlib import Path
# from zipfile import ZipFile 
from os import path
import pandas as pd
import numpy as np

from rl.environments import Environment
import rl.agents as agents
import rl.subjects as subjects


class Experiment(Environment):
    '''
    Provide an experimentation environment for agents and subjects.

    Attributes
    ----------

    Methods
    -------
        add: add a set of objects (agents/ subjects) to the environment.
        remove: remove objects (agents/ subjects) from the environment.
        assign: assign agents to subjects.
        run: run the experiment using agents on subjects.
        load: load an object (agent/ subject) or an environment.
        save: save an object (agent/ subject) or the current environment.

    Agents act on subjects and receive the reward of their action and the new state of subjects.
    '''

    def __init__(self, **kwargs):
        '''
        Create a new experiment.

        Arguments
        ---------
            filename: if given, Environment attempts to open a saved environment by openning the file. This argument cannot be used along other arguments.
            path: path of the file to be loaded (should be used with filename)
            name: the name of the environment
            episodes: number of episodes of run. (Default = 1)
            termination: when to terminate one episode. (Default = 'any')
                'any': terminate if any of the subjects is terminated.
                'all': terminate if all the subjects are terminated.
            reset: how to reset subjects after each episode. (Default = 'any')
                'any': reset only the subject that is terminated.
                'all': reset all subjects.
            learning_method: how to learn from each episode. (Default = 'every step')
                'every step': learn after every move.
                'history': learn after each episode.
        '''
        Environment.__init__(self, **kwargs)
        if 'filename' in kwargs:
            self.load(path=kwargs.get('path', '.'),
                      filename=kwargs['filename'])
            return

        Environment.set_defaults(self, agent={}, subject={}, assignment_list={},
                            number_of_subjects=1, max_steps=10000, save_subjects=True)
        Environment.set_params(self, **kwargs)

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._agent, self._subject, self._assignment_list = {}, {}, {}
            self._number_of_subjects, self._max_steps = 1, 10000
            self._save_subjects = 'all'


    def elapse(self, **kwargs):
        '''
        Move forward in time for a number of episodes.

        At each episode, agents are called sequentially to act on their respective subject(s).
        An episode ends if one (all) subject(s) terminates.
        Arguments
        ---------
            episodes: number of episodes of run.
            max_steps: maximum number of steps in each episode (Default = 10,000)
            termination: when to terminate one episode. (Default = 'any')
                'any': terminate if any of the subjects is terminated.
                'all': terminate if all the subjects are terminated.
            reset: how to reset subjects after each episode. (Default = 'any')
                'any': reset only the subject that is terminated.
                'all': reset all subjects.
            learning_method: how to learn from each episode. (Default = 'every step')
                'every step': learn after every move.
                'history': learn after each episode.
            reporting: what to report. (Default = 'none')
                'all': prints every move.
                'none' reports nothing.
                'important' reports important parts only.
            tally: count wins of each agent or not. (Default = 'no')
                'yes': counts the number of wins of each agent.
                'no': doesn't tally.
            step_count: count average number of steps in each episode. (Default = 'no')
                'yes': counts the average number of steps in each episode.
                'no': doesn't count.
        '''
        episodes = kwargs.get('episodes', self._episodes)
        max_steps = kwargs.get('max_steps', self._max_steps)
        termination = kwargs.get('termination', self._termination).lower()
        reset = kwargs.get('reset', self._reset)
        learning_method = kwargs.get(
            'learning_method', self._learning_method).lower()
        reporting = kwargs.get('reporting', 'none').lower()

        tally = kwargs.get('tally', 'no').lower() == 'yes'
        if tally:
            win_count = {}
            for agent in self._agent:
                win_count[agent] = 0

        step_count = kwargs.get('step_count', 'no').lower() == 'yes'

        if learning_method == 'none':
            for agent in self._agent.values():
                agent.status = 'testing'
        else:
            for agent in self._agent.values():
                agent.status = 'training'

        steps = 0
        for episode in range(episodes):
            if reporting != 'none':
                report_string = 'episode: {}'.format(episode+1)
            # if learning_method == 'history':
            history = {}
            for agent_name in self._agent:
                history[agent_name] = pd.DataFrame(columns=['state', 'action', 'reward'])
                # history[agent_name] = pd.DataFrame(columns=['state', 'action', 'reward'])
            done = False
            stopping_criterion = max_steps * (episode+1)
            while not done:
                if steps >= stopping_criterion:
                    break
                steps += 1
                for agent_name, agent in self._agent.items():
                    if not done:
                        subject_name, _id = self._assignment_list[agent_name]
                        subject = self._subject[subject_name]
                        if not subject.is_terminated:
                            state = subject.state
                            possible_actions = subject.possible_actions
                            action = agent.act(state, actions=possible_actions,
                                               episode=self._total_experienced_episodes,
                                               printable=subject.printable())
                            if reporting == 'all':
                                print('step: {: 4} episode: {:2} state: {} action: {} by:{}'
                                      .format(steps, episode, state, action, agent_name))
                            reward = subject.take_effect(_id, action)

                            history[agent_name].loc[len(history[agent_name].index)] = [state, action, reward]

                            # history[agent_name].append(state)
                            # history[agent_name].append(action)
                            # history[agent_name].append(reward)

                            if subject.is_terminated:
                                if tally & (reward > 0):
                                    win_count[agent_name] += 1
                                for affected_agent in self._agent.keys():
                                    if (self._assignment_list[affected_agent][0] == subject_name) & \
                                            (affected_agent != agent_name):
                                        history[affected_agent][-1] = -reward

                    if termination == 'all':
                        done = True
                        for sub in self._subject.values():
                            done = done & sub.is_terminated
                    elif termination == 'any':
                        done = False
                        for sub in self._subject.values():
                            done = done | sub.is_terminated

                if learning_method == 'every step':
                    for agent_name, agent in self._agent.items():
                        state = self._subject[self._assignment_list[agent_name][0]].state
                        # agent.learn(state=state, reward=history[agent_name][-1]['reward'])
                        # history[agent_name] = pd.DataFrame(columns=['state', 'action', 'reward'])
                        agent.learn(state=state, reward=history[agent_name][2])
                        history[agent_name] = []

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
                        self._subject[subject_name[0]].reset()

            if tally & (reporting != 'none'):
                report_string += '\n tally:'
                for agent in self._agent:
                    report_string += '\n {} {}'.format(agent, win_count[agent])

            if reporting != 'none':
                print(report_string)

            self._total_experienced_episodes += 1

        if tally:
            return win_count
        if step_count:
            return steps/episodes

    def generate_subjects(self, **kwargs):
        number_of_subjects = kwargs.get('number_of_subjects', self._number_of_subjects)
        self._number_of_subjects = number_of_subjects
        # save_subjects = kwargs.get('save_subjects', self._save_subjects)
        digits = ceil(log10(number_of_subjects))

        for subject_name, subject in self._subject.items():
            subject.reset()
            for subject_ID in range(number_of_subjects):
                filename = subject_name + str(subject_ID).rjust(digits, '0')
                if path.exists('./' + subject_name + '/' + filename + '.pkl'):
                    print('{} already exists! Skipping the file!'.format(filename))
                else:
                    subject.save(filename=filename, path='./' + subject_name)
                    subject.reset()
                # with ZipFile(subject_name,'w') as zip:
                #     for file in file_paths: 
                #         zip.write(file)

    def run(self, **kwargs):
        '''
        ...

        Arguments
        ---------
            max_steps: maximum number of steps in the episode (Default = 10,000)

        '''
        max_steps = kwargs.get('max_steps', self._max_steps)
        number_of_subjects = kwargs.get('number_of_subjects', self._number_of_subjects)
        # save_subjects = kwargs.get('save_subjects', self._save_subjects)
        digits = ceil(log10(number_of_subjects))

        for agent_name, agent in self._agent.items():
            print('Agent: {}'.format(agent_name))
            agent.status = 'testing'

            try:
                subject_name, _id = self._assignment_list[agent_name]
            except KeyError:
                continue

            subject = self._subject[subject_name]

            for subject_ID in range(number_of_subjects):
                history = pd.DataFrame(columns=['state', 'action', 'q', 'reward'])

                output_dir = Path('./'+subject_name+'/results')
                output_dir.mkdir(parents=True, exist_ok=True)

                filename = subject_name + str(subject_ID).rjust(digits, '0')
                print('Subject: {}'.format(filename))
                temp_agent_list = subject._agent_list
                subject.load(filename=filename, path='./' + subject_name)
                subject._agent_list = temp_agent_list

                steps = 0
                while not subject.is_terminated and steps < max_steps:
                    steps += 1

                    state = subject.state
                    possible_actions = subject.possible_actions
                    action = agent.act(state, actions=possible_actions)
                    try:
                        q = agent._q(state, action)
                    except AttributeError:
                        q = np.nan
                    reward = subject.take_effect(_id, action)

                    history.loc[len(history.index)] = [state, action, q, reward]

                if subject.is_terminated:
                    for affected_agent in self._agent.keys():
                        try:
                            if (self._assignment_list[affected_agent][0] == subject_name) & \
                                    (affected_agent != agent_name):
                                history[-1] = -reward
                        except KeyError:
                            pass

                history.to_pickle(output_dir / (agent_name + filename + '.pkl'))

    def __repr__(self):
        try:
            return 'Exp: \n Agents:\n' + \
                '\n\t'.join((a.__repr__() for a in self._agent.values())) + \
                '\nSubjects:\n' + \
                '\n\t'.join((s.__repr__() for s in self._subject.values()))
        except AttributeError:
            return 'Experiment: New'


if __name__ == "__main__":
    from rl.agents import RandomAgent
    from rl.subjects import WarfarinModel_v4

    exp = Experiment(agents={'R': RandomAgent()},
                     subjects={'W': WarfarinModel_v4},
                     assignment_list={'R': 'W'})

    exp.generate_subjects(number_of_subjects=5)
    exp.run()