# -*- coding: utf-8 -*-
'''
NOT IMPLEMENTED YET!!!

Experience replay environment class
=================

This `environment` class provides a learning environment for any reinforcement learning agent on any subject. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from pickle import load, dump, HIGHEST_PROTOCOL
import signal
import sys, inspect
import pathlib

from ..base import RLBase
import rl.agents as agents 
import rl.subjects as subjects
from .environment import Environment

class ExpReplayEnvironment(Environment):
    '''
    Provide an environment for buffering observations and learning from experience.

    Attributes
    ----------

    Methods
    -------
        add: add a set of objects (agents/ subjects) to the environment.
        remove: remove objects (agents/ subjects) from the environment.
        assign: assign agents to subjects.
        elapse: move forward in time and interact agents and subjects.
        trajectory: extract (state, action, reward) trajectory.
        load: load an object (agent/ subject) or an environment.
        save: save an object (agent/ subject) or the current environment.

    Agents act on subjects and receive the reward of their action and the new state of subjects.
    The S, A, r sequences are recorded in a buffer, then sampled to be learned by the agent.
    Note: the agent should be able to learn in batch.
    '''
    def __init__(self, **kwargs):
        '''
        Create a new environment.

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
            self.load(path=kwargs.get('path','.'), filename=kwargs['filename'])
            return

        Environment.set_defaults(self, agent={}, subject={}, assignment_list={},
                           episodes=1, max_steps=10000, termination='any', reset='all',
                           learning_method='every step',
                           allow_user_to_halt=True, save_on_exit=True)
        Environment.set_params(self, **kwargs)

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._agent, self._subject, self._assignment_list = {}, {}, {}
            self._episodes, self._max_steps = 1, 10000
            self._termination, self._reset, self._learning_method = 'any', 'all', 'every step'


    def elapse(self, **kwargs):
        '''
        Move forward in time for a number of episodes.
        
        At each episode, agents are called sequentially to act on their respective subject(s).
        An episode ends if one (all) subject(s) terminates.
        Arguments
        ---------
            episodes: number of episodes of run.
            buffer_size: how many records to keep in the buffer.
            batch_size: how many records to sample from the buffer for training.
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
        # one episode is one full game till the end
        try:  # episodes
            episodes = kwargs['episodes']
        except KeyError:
            episodes = self._episodes
        try:
            max_steps = kwargs['max_steps']
        except KeyError:
            max_steps = self._max_steps
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

        try:  # tally
            if kwargs['step_count'].lower() == 'yes':
                step_count = True
            else:
                step_count = False
        except KeyError:
            step_count = False

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
                history[agent_name] = []
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
                                               printable=subject.printable())
                            if reporting == 'all':
                                print('step: {: 4} episode: {:2} state: {} action: {} by:{}'
                                    .format(steps, episode, state, action, agent_name))
                            reward = subject.take_effect(_id, action)
                            # if reporting == 'all':
                            #     report_string += '\n'+subject.printable()+'\n'

                            history[agent_name].append(state)
                            history[agent_name].append(action)
                            history[agent_name].append(reward)
                            # print('{: 4d} {: 4d}'.format(episode, steps), subject._player_location)
                            # if ([*subject._player_location] == [*subject._goal]):
                            #     pass
                            # if reward>0:
                            #     print('Done')

                            if subject.is_terminated:
                                if tally & (reward>0):
                                    win_count[agent_name] += 1
                                for affected_agent in self._agent.keys():
                                    if (self._assignment_list[affected_agent][0] == subject_name) & \
                                        (affected_agent != agent_name):
                                        history[affected_agent][-1] = -reward
                                        # if learning_method == 'every step':
                                        #     self._agent[affected_agent].learn(state=subject.state,
                                        #         reward=reward if affected_agent == agent_name else -reward)
                                        # elif learning_method == 'history':

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

        if tally:
            return win_count
        if step_count:
            return steps/episodes

    def __repr__(self):
        return 'ExpReplayEnvironment'