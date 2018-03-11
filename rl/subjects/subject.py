# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 21:35:10 2018

@author: Sadjad Anzabi Zadeh

This module contains classes of different types of subjects.

Classes:
    Subject (Super Class)
    MNKBoard (Super Class): this is not a subject, but one of the super classes of MNKGame class.
    MNKGame

mnkboard(m=3, n=3, k=3, players=2)
    set_piece(player, **kwargs)
    get_board(format_='vector')
    get_action_set(format_='vector')
    reset()
    printable()

mnkgame(mnkboard)(m=3, n=3, k=3, players=2)
    board_status
    is_terminated
    register(player_name): registers a player and gives back an ID
    possible_actions(): returns a list of possible actions
    take_effect(ID, action): returns reward
    reset()
    set_piece(player, *argv, **kwargs)

"""

import pickle


def main():
    pass


class Subject:
    '''
    Super class of all subject classes. This class provides basic methods including is_terminated, register, take_effect, reset, load, save, and printable.
    Register, load and save are implemented.
    '''
    def __init__(self):
        self._agent_list = {}

    @property
    def state(self):
        '''
        Returns the state.
        '''
        raise NotImplementedError

    @property
    def is_terminated(self):
        pass

    @property
    def possible_actions(self):
        pass

    def register(self, agent_name):
        '''
        Registers an agent and returns its ID. If the agent is new, a new ID is generated and the agent_name is added to agent_list.
        \nArguments:
        \n    agent_name: the name of the agent to be registered.
        '''
        try:
            return self._agent_list[agent_name]
        except KeyError:
            _id = len(self._agent_list) + 1
            self._agent_list[agent_name] = _id
            return _id

    def take_effect(self, _id, action):
        pass

    def reset(self):
        pass

    def load(self, **kwargs):
        '''
        Loads an agent.
        \nArguments:
        \n    filename: the name of the file to be loaded.
        '''
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')
        with open(filename + '.pkl', 'rb') as f:
            self.__dict__ = pickle.load(f)

    def save(self, **kwargs):
        '''
        Saves an agent.
        \nArguments:
        \n    filename: the name of the file to be saved.
        '''
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')
        with open(filename + '.pkl', 'wb+') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def printable(self):
        pass


if __name__ == '__main__':
    main()
