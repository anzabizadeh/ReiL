# -*- coding: utf-8 -*-
'''
MNKBoard class
==============

This class creates a board for players to play mnk game.
It serves as a super class for MNKGame.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import pickle

from rl.valueset import ValueSet


def main():
    # create a board and set piece for each player and print the board
    board = MNKBoard(m=3, n=3, k=3, players=3)
    board.set_piece(1, row=0, column=0)
    board.set_piece(2, index=4)
    board.set_piece(3, index=8)
    print(board.printable())


class MNKBoard:
    '''
    Provide an m-by-n board to play.

    Attributes
    ----------
        state: the state of the board as a ValueSet.

    Methods
    -------
        set_piece: set a piece of the specified player on the specified square of the board.
        clear_square: clear the specified square of the board.
        get_board: return the board either as a vector or as a matrix.
        get_action_set: return a list of empty squares.
        reset: clear the board.
        printable: format the board state for more readability. 
    '''
    def __init__(self, **kwargs):
        '''
        Initialize an instance of mnkgame.

        Arguments
        ---------
            m: number of rows (default=3)
            n: number of columns (default=3)
            k: winning criteria (default=3)
            players: number of players (default=2)
            can_recapture: whether a piece can be put on an occupied square
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
        try:  # players
            self._can_recapture = kwargs['can_recapture']
        except KeyError:
            self._can_recapture = True

        self._board = [0]*(self._m*self._n)

    def set_piece(self, player, **kwargs):
        '''
        Set a piece for a player.
        
        Arguments
        ---------
            player: ID of the player whose piece will be set on the board.
            index: If provided, the piece is set using 'index'. Index starts from 0 and assumes the board to be a list.
            row: The row in which the piece is set.
            column: The column in which the piece is set.

        Raises ValueError if the player or the location is out of range or niether index nor row-column is provided.
        '''
        try:
            can_recapture = kwargs['can_recapture']
        except KeyError:
            can_recapture = self._can_recapture

        if (player <= 0) | (player > self._players):
            raise ValueError('player not found.')
        if not kwargs:
            raise ValueError('No row-column pair or index found.')
        try:
            if (self._board[kwargs['index']] != 0) & (not can_recapture):
                raise ValueError('The square is already occupied.')
            self._board[kwargs['index']] = player
        except KeyError:
            if (self._board[kwargs['row']*self._n + kwargs['column']] != 0) & (not can_recapture):
                raise ValueError('The square is already occupied.')
            self._board[kwargs['row']*self._n + kwargs['column']] = player

    def clear_square(self, **kwargs):
        '''
        Clear a square on the board.

        Arguments
        ---------
            index: If provided, the piece is set using 'index'. Index starts from 0 and assumes the board to be a list.
            row: The row in which the piece is set.
            column: The column in which the piece is set.

        Raises ValueError if the player or the location is out of range or niether index nor row-column is provided
        '''
        if not kwargs:
            raise ValueError('No row-column pair or index found.')
        try:
            self._board[kwargs['index']] = 0
        except KeyError:
            self._board[kwargs['row']*self._n + kwargs['column']] = 0

    @property
    def state(self):
        ''' Return the state of the board as a ValueSet.'''
        return ValueSet(self._board, min=0, max=self._players)

    def get_board(self, format_='vector'):
        '''
        Return the board.
        
        Arguments
        ---------
            format_: 'vector' returns the board as a list. 'matrix' returns the board as a 2D list. (Default='vector').
            
        Raises ValueError if undefined format is provided.
        '''
        if format_.lower() == 'vector':
            return self._board
        elif format_.lower() == 'matrix':
            return self._matrix(self._board)
        else:
            raise ValueError('format should be either vector or matrix')

    def get_action_set(self, format_='vector'):
        '''
        Return a list of indexes of empty squares.

        Arguments
        ---------
            format_: 'vector' returns the board as a list. 'matrix' returns the board as a 2D list. (Default='vector').
            
        Raises ValueError if undefined format is provided.
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
        '''Clear the board.'''
        self._board = [0]*(self._m*self._n)

    def printable(self):
        '''Return a printable format string of the board.''' 
        return ('\n'.join([''.join(['{:4}'.format(item) for item in row])
                           for row in self._matrix()]))

    def _matrix(self, *args):
        '''
        Get a board and returns it in 2D list format.

        Arguments
        ---------
            args[0]: the board to be converted. If no arguments is supplied, current object's board is returned.
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

    def __repr__(self):
        return 'MNKBoard'

if __name__ == '__main__':
    main()
