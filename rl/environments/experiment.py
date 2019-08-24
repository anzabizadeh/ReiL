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

                history.to_pickle(output_dir / (agent_name + '@' + filename + '.pkl'))

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