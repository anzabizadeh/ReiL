# -*- coding: utf-8 -*-
'''
WarfarinAgent class
=================

An agent for warfarin modeling based on the doses define in Ravvaz et al (2017)

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''


import os
from pickle import HIGHEST_PROTOCOL, dump, load
from random import choice, random
from time import time

import numpy as np

from .agent import Agent
from ..rldata import RLData


class WarfarinAgent(Agent):
    '''
    An agent for warfarin modeling based on the doses define in Ravvaz et al (2017).

    Constructor Arguments
    ---------------------

        Note: This class doesn't have any data_collectors.
    
    Methods
    -------
        act: return an action based on the given state.
        learn: learn using either history or action, reward, and state.
        reset: Not Implemented Yet!
    '''

    def __init__(self, **kwargs):
        '''
        Initialize a warfarin agent.
        '''
        Agent.__init__(self, **kwargs)
        Agent.set_defaults(self)
        Agent.set_params(self, **kwargs)
        self.data_collector.available_statistics = {}
        self.data_collector.active_statistics = []

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._day = 1

    def learn(self, **kwargs):
        '''
        Learn based on history.

        Note: Since this agent implements fixed policies, it does not learn.
        '''
        pass

    def act(self, state, **kwargs):
        '''
        return the best action for a given state.

        Arguments
        ---------
            state: the state for which an action is chosen.
            actions: a set of possible actions. If not provided, default actions are used.

        Note: If in 'training', the action is chosen randomly with probability of epsilon. In in 'test', the action is greedy.
        '''
        self._previous_state = state
        possible_actions = kwargs.get('actions', self._default_actions)
        episode = kwargs.get('episode', 0)
        try:
            epsilon = self._epsilon(episode)
        except TypeError:
            epsilon = self._epsilon

        if (self._training_flag) & (random() < epsilon):
            result = possible_actions
        else:
            q_values = self._q(state, possible_actions)
            max_q = np.max(q_values)
            result = tuple(possible_actions[i] for i in np.nonzero(q_values==max_q)[0])
            # result = max(((possible_actions[i], q_values[i]) for i in range(len(possible_actions))), key=lambda x: x[1])
            # action = result[0]

        action = choice(result)

        self._previous_action = action
        return action

    def action(self, method, patient):
        if method.lower() == 'aurora':
            if day <= 2:
                dose = 10 if patient_characteristics['age'] < 65 else 5
            else:
                pass


    # def load(self, **kwargs):
    #     '''
    #     Load an object from a file.

    #     Arguments
    #     ---------
    #         filename: the name of the file to be loaded.

    #     Note: tensorflow part is saved in filename.tf folder

    #     Raises ValueError if the filename is not specified.
    #     '''
    #     Agent.load(self, **kwargs)
    #     tf.reset_default_graph()
    #     self._model = keras.models.load_model(kwargs.get(
    #         'path', self._path) + '/' + kwargs['filename'] + '.tf/' + kwargs['filename'])
    #     self._tensorboard = keras.callbacks.TensorBoard(
    #         log_dir=self._tensorboard_path)

    # def save(self, **kwargs):
    #     '''
    #     Save the object to a file.

    #     Arguments
    #     ---------
    #         filename: the name of the file to be saved.

    #     Note: tensorflow part should be in filename.tf folder

    #     Raises ValueError if the filename is not specified.
    #     '''

    #     pickle_data = tuple(key for key in self.__dict__ if key not in [
    #                         '_model', '_tensorboard', 'data_collector'])
    #     path, filename = Agent.save(self, **kwargs, data=pickle_data)
    #     try:
    #         self._model.save(kwargs.get('path', self._path) + '/' +
    #                          kwargs['filename'] + '.tf/' + kwargs['filename'])
    #     except OSError:
    #         os.makedirs(kwargs.get('path', self._path) +
    #                     '/' + kwargs['filename'] + '.tf/')
    #         self._model.save(kwargs.get('path', self._path) + '/' +
    #                          kwargs['filename'] + '.tf/' + kwargs['filename'])
    #     return path, filename

    def _report(self, **kwargs):
        '''
        generate and return the requested report.

        Arguments
        ---------
            statistic: the list of items to report.

        Note: this function is not implemented!
        '''
        raise NotImplementedError

    def __repr__(self):
        return 'WarfarinAgent'
