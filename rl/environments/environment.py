# -*- coding: utf-8 -*-
'''
environment class
=================

This `environment` class provides a learning environment for any reinforcement learning agent on any subject. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from ..base import RLBase


def main():
    from ..agents import QAgent, RandomAgent
    from ..subjects import MNKGame
    # An environment hosts agents and subjects and regulates their interactions
    filename='several agents'
    RND = RandomAgent()
    try:
        # You can load an environment from file either by class's constructor or using load() method
        env = Environment(filename=filename)
    except (ModuleNotFoundError, FileNotFoundError):
        env = Environment()
        RLS = QAgent(gamma=1, alpha=0.5, epsilon=0.1, Rplus=0.2, Ne=1)
        agents = {'RLS 1': RLS, 'RLS 2': RLS}
        subjects = {'board RLS': MNKGame()}
        assignment = [('RLS 1', 'board RLS'), ('RLS 2', 'board RLS')]
        # add method receives a dictionary of agents/ subjects composed of name and object.
        env.add(agents=agents, subjects=subjects)
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


class Environment(RLBase):
    '''
    Provide a learning environment for agents and subjects.

    Attributes
    ----------

    Methods
    -------
        add: add a set of objects (agents/ subjects) to the environment.
        remove: remove objects (agents/ subjects) from the environment.
        assign: assign agents to subjects
        elapse: move forward in time and interact agents and subjects

    Agents act on subjects and receive the reward of their action and the new state of subjects.
    Then agents learn based on this information to act better.
    '''
    def __init__(self, **kwargs):
        '''
        Create a new environment.

        Arguments
        ---------
            filename: if given, Environment attempts to open a saved environment by openning the file. This argument cannot be used along other arguments.
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
        if 'filename' in kwargs:
            self.load(filename=kwargs['filename'])
            return

        RLBase.__init__(self, **kwargs)
        RLBase.set_defaults(self, agent={}, subject={}, assignment_list={},
                           default_episodes=1, termination='any', reset='all',
                           learning_method='every step')
        RLBase.set_params(self, **kwargs)

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._agent, self._subject, self._assignment_list = {}, {}, {}
            self._default_episodes = 1
            self._termination, self._reset, self._learning_method = 'any', 'all', 'every step'

    def add(self, **kwargs):
        '''
        Add agents to the environment.

        Arguments
        ---------
            agents: a dictionary consist of agent name and agent object. Names should be unique, otherwise overwritten.
            subjects: a dictionary consist of subject name and subject object. Names should be unique, otherwise overwritten.
        '''
        try:
            for name, agent in kwargs['agents'].items():
                self._agent[name] = agent
        except IndexError:
            pass

        try:
            for name, subject in kwargs['subjects'].items():
                self._subject[name] = subject
        except IndexError:
            pass

    def remove(self, **kwargs):
        '''
        Remove agents from the environment.

        Arguments
        ---------
            agents: a list of agent names to be deleted.
            subjects: a list of subject names to be deleted.

        Raises KeyError if the agent is not found.
        '''
        for name in kwargs['agents']:
            try:
                del self._agent[name]
            except KeyError:
                raise KeyError('Agent '+name+' not found')

        for name in kwargs['subjects']:
            try:
                del self._subject[name]
            except KeyError:
                raise KeyError('Subject '+name+' not found')

    def assign(self, agent_subject_names):
        '''
        Assign agents to subjects.
        
        Arguments
        ---------
            agent_subject_names: a list of agent subject tuples.

        Raises ValueError if an agent or subject is not found.
        Note: An agent can be assigned to act on multiple subjects and a subject can be affected by multiple agents. 
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
        Move forward in time for a number of episodes.
        
        At each episode, agents are called sequentially to act on their respective subject(s).
        An episode ends if one (all) subject(s) terminates.
        Arguments
        ---------
            episodes: number of episodes of run.
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
