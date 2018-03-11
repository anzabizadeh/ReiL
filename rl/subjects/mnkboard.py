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
from ..valueset.valueset import ValueSet

def main():
    pass


class MNKBoard:
    def __init__(self, **kwargs):
        '''
        Initializes an instance of mnkgame.
        \nArguments:
        \n    m: number of rows
        \n    n: number of columns
        \n    k: winning criteria
        \n    players: number of players
        '''

        try:  # m
            self._m = kwargs['m']
        except KeyError:
            self._m = 3
        try:  # n
            self._n = kwargs['n']
        except KeyError:
            self._n = 3
        try:  # k
            self._k = kwargs['k']
        except KeyError:
            self._k = 3
        try:  # players
            self._players = kwargs['players']
        except KeyError:
            self._players = 2

        self._board = [0]*(self._m*self._n)

    def set_piece(self, player, **kwargs):
        '''
        Sets a piece for a player. An attempt to set a piece for an undefined player
        or in an unspecified location will result in ValueError.
        \nArguments:
        \n    player: ID of the player whose piece will be set on the board.
        \n    index: If provided, the piece is set using 'index'. Index starts from 0 and assumes the board to be a list.
        \n    row: The row in which the piece is set.
        \n    column: The column in which the piece is set.
        '''
        if (player <= 0) | (player > self._players):
            raise ValueError('player not found.')
        if not kwargs:
            raise ValueError('No row-column pair or index found.')
        try:
            self._board[kwargs['index']] = player
        except KeyError:
            self._board[kwargs['row']*self._n + kwargs['column']] = player

    def clear_piece(self, **kwargs):
        '''
        Clears a location on the board.
        \nArguments:
        \n    index: If provided, the piece is set using 'index'. Index starts from 0 and assumes the board to be a list.
        \n    row: The row in which the piece is set.
        \n    column: The column in which the piece is set.
        '''
        if not kwargs:
            raise ValueError('No row-column pair or index found.')
        try:
            self._board[kwargs['index']] = 0
        except KeyError:
            self._board[kwargs['row']*self._n + kwargs['column']] = 0

    @property
    def state(self):
        '''
        Returns the state of the board as a list.
        '''
        s = ValueSet(*self._board)
        s.min = 0
        s.max = self._players
        return s

    def get_board(self, format_='vector'):
        '''
        Returns the board.
        \nArguments:
        \n    format_: 'vector' returns the board as a list. 'matrix' returns the board as a 2D list. Default is 'vector'. Other values result in ValueError Exception.
        '''
        if format_.lower() == 'vector':
            return self._board
        elif format_.lower() == 'matrix':
            return self._matrix(self._board)
        else:
            raise ValueError('format should be either vector or matrix')

    def get_action_set(self, format_='vector'):
        '''
        Returns a list of indexes of empty squares.
        \nArguments:
        \n    format_: 'vector' returns actions in index format. 'matrix' returns actions in row column format. Default is 'vector'. Other values result in ValueError Exception.
        '''  
        index = (i for i in range(self._m*self._n) if self._board[i] == 0)
        for action in index:
            if format_.lower() == 'vector':
                yield action
            elif format_.lower() == 'matrix':
                yield [action // self._n, action % self._n]
            else:
                raise TypeError('format should be either vector or matrix')

    def reset(self):
        '''
        Empties the board.
        '''
        self._board = [0]*(self._m*self._n)

    def printable(self):
        '''
        Returns a printable format string of the board.
        ''' 
        return ('\n'.join([''.join(['{:4}'.format(item) for item in row])
                           for row in self._matrix()]))

    def _matrix(self, *args):
        '''
        Gets a board and returns it in 2D list format.
        \nArguments:
        \n    args[0]: the board to be converted. If no arguments is supplied, current object's board is returned.
        '''
        if not args:
            vector = self._board
        else:
            vector = args[0]
        i = 0
        matrix = [([0] * self._n) for row in range(self._m)]
        for row in range(self._m):
            for col in range(self._n):
                matrix[row][col] = vector[i]
                i = i + 1
        return matrix


if __name__ == '__main__':
    main()
