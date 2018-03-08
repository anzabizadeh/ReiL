# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:23:38 2018

@author: Sadjad Anzabi Zadeh

This module contains classes that provides tools to load/save objects and environments,
add/remove agents/subjects, assign agents to subjects and run models.

Classes:
    Environment
"""

from agents import RLAgent, RandomAgent, UserAgent
from subjects import MNKGame
import valueset as vs
import pickle


def main():
    # An environment hosts agents and subjects and regulates their interactions
    filename='several agents'
    RND = RandomAgent()
    try:
        # You can load an environment from file either by class's constructor or using load() method
        env = Environment(filename=filename)
    except (ModuleNotFoundError, FileNotFoundError):
        env = Environment()
        RLS = RLAgent(gamma=1, alpha=0.5, epsilon=0.1, Rplus=0.2, Ne=1)
        agents = {'RLS 1': RLS, 'RLS 2': RLS}
        subjects = {'board RLS': MNKGame()}
        assignment = [('RLS 1', 'board RLS'), ('RLS 2', 'board RLS')]
        # add_agent method receives a dictionary of agents composed of name and object.
        env.add_agent(name_agent_pair=agents)
        # add_subject method receives a dictionary of subjects composed of name and object.
        env.add_subject(name_subject_pair=subjects)
        # assign method receives a list of agent subject tuples
        env.assign(assignment)

    runs = 100
    training_episodes = 100
    test_episodes = 100
    results = {'RLS 1': [], 'RLS 2': []}
    try:
        for i in range(runs):
            env.elapse(episodes=training_episodes, reset='all',
                       termination='all', learning_method='history',
                       reporting='none', tally='no')
            agent_temp = env._agent['RLS 2']
            env._agent['RLS 2'] = RND
            # env._agent['a1'].status = 'testing'
            tally = env.elapse(episodes=test_episodes, reset='all',
                               termination='all', learning_method='none',
                               reporting='none', tally='yes')
            for key in results:
                results[key].append(tally[key])
            print('run {: }: RLS loss: {: }'.format(i, tally['RLS 2']))
            env._agent['RLS 2'] = agent_temp
    except KeyboardInterrupt:
        pass

    env.save(object_name='all', filename=filename)
    with open('results.pkl', 'wb+') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


class Environment:
    '''
    Provides a learning environment including agents and subjects.
    Agents act on subjects and receive the reward of their action and the new state of subjects.
    Then agents learn based on this information to act better.
    '''
    def __init__(self, **kwargs):
        '''
        \nArguments:
        \n    filename: if given, Environment attempts to open a saved environment by openning the file. This argument cannot be used along other arguments.
        \n    episodes: number of episodes of run. Default is 1.
        \n    termination: 'any' means an episode terminates if any of the subjects terminates. 'all' runs until all episodes terminate. Default is 'any'.
        \n    reset: 'any' means that only the subject that is terminated should be reset. 'all' resets all subjects. Default is 'any'.
        \n    learning_method: 'every step' learns after every move, 'history' learns after each episode. Default is 'every step'
        '''
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
            self._reset = 'all'

        # learning_method: 'every step' (learns after every move)
        #                  'history' (learns after each episode)
        try:  # learning_method
            self._learning_method = kwargs['learning_method']
        except KeyError:
            self._learning_method = 'every step'

    def load(self, **kwargs):
        '''
        Loads an object of an environment.
        \nArguments:
        \n    object_name: if specified, that object (agent or subject) is being loaded from file. 'all' loads an environment. Default is 'all'.
        \n    filename: the name of the file to be loaded.
        '''
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
        '''
        Saves an object (agent or subject) or the environment.
        \nArguments:
        \n    object_name: if specified, that object (agent or subject) is being saved to file. 'all' saves the environment. Default is 'all'.
        \n    filename: the name of the file to save.
        '''
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
        '''
        Adds agents to the environment.
        \nArguments:
        \n    name_agent_pair: a dictionary consist of agent name and agent object. Names should be unique, otherwise overwritten.
        '''
        for name, agent in name_agent_pair.items():
            self._agent[name] = agent

    def remove_agent(self, agent_names):
        '''
        Removes agents from the environment.
        \nArguments:
        \n    agent_names: a list of agent names to be deleted. If not found, KeyError exception is raised.
        '''
        for name in agent_names:
            try:
                del self._agent[name]
            except KeyError:
                raise KeyError('Agent '+name+' not found')

    def add_subject(self, name_subject_pair):
        '''
        Adds subjects to the environment.
        \nArguments:
        \n    name_subject_pair: a dictionary consist of subject name and subject object. Names should be unique, otherwise overwritten.
        '''
        for name, subject in name_subject_pair.items():
            self._subject[name] = subject

    def remove_subject(self, subject_names):
        '''
        Removes subjects from the environment.
        \nArguments:
        \n    subject_names: a list of subject names to be deleted. If not found, KeyError exception is raised.
        '''
        for name in subject_names:
            try:
                del self._subject[name]
            except KeyError:
                raise KeyError('Subject '+name+' not found')

    def assign(self, agent_subject_names):
        '''
        Assigns agents to subjects. An agent can be assigned to act on multiple subjects and a subject can be affected by multiple agents. 
        \nArguments:
        \n    agent_subject_names: a list of agent subject tuples. If any agent or subject not found, ValueError exception is raised.
        '''
        for agent_name, subject_name in agent_subject_names:
            if agent_name not in self._agent:
                raise ValueError('Agent ' + agent_name + ' not found.')
            if subject_name not in self._subject:
                raise ValueError('Subject ' + subject_name + ' not found.')
            _id = self._subject[subject_name].register(agent_name)
            self._assignment_list[agent_name] = (subject_name, _id)

    def elapse(self, **kwargs):
        '''
        Moves forward in time for a number of episodes. At each episode agents are called sequentially to act on their respective subject(s).
        An episode ends if one (all) subject(s) terminates.
        \nArguments:
        \n    episodes: number of episodes of run.
        \n    termination: 'any' means an episode terminates if any of the subjects terminates. 'all' runs until all episodes terminate.
        \n    reset: 'any' means that only the subject that is terminated should be reset. 'all' resets all subjects.
        \n    learning_method: 'every step' learns after every move, 'history' learns after each episode. 'none' initiates 'testing' state of agents.
        \n    reporting: 'all' prints every move, 'none' reports nothing, 'important' reports important parts only.
        \n    tally: 'yes' counts the number of wins of each agent. 'no' doesn't tally. Default is 'no'.
        '''
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
                            state = subject.state
                            possible_actions = subject.possible_actions
                            action = agent.act(state, actions=possible_actions,
                                               printable=subject.printable())
                            reward = subject.take_effect(_id, action)
                            if reporting == 'all':
                                report_string += '\n'+subject.printable()+'\n'
                            if subject.is_terminated:
                                if tally & (reward>0):
                                    win_count[agent_name] += 1
                                for affected_agent in self._agent.keys():
                                    if learning_method == 'every step':
                                        self._agent[affected_agent].learn(state=subject.state,
                                            reward=reward if affected_agent == agent_name else -reward)
                                    elif learning_method == 'history':
                                        history[affected_agent].append(state)
                                        history[affected_agent].append(action)
                                        history[affected_agent].append(reward if affected_agent == agent_name else -reward)
                            else:
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
