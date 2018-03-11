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

from ..valueset import ValueSet
from .mnkboard import MNKBoard
from .subject import Subject


def main():
    pass


class MNKGame(MNKBoard, Subject):
    '''
    Builds an m by n board (using mnkboard super class)
    in which p players can play. Winner is the player who can put k pieces
    in on row, column, or diagonal.
    '''
    # _board is a row vector. (row, column) and index start from 0
    # _board_status: None: no winner yet,
    #                1..players: winner,
    #                0: stall,
    #               -1: illegal board
    def __init__(self, **kwargs):
        '''
        Initializes an instance of mnkgame.
        \nArguments:
        \n    m: number of rows
        \n    n: number of columns
        \n    k: winning criteria
        \n    players: number of players
        '''
        MNKBoard.__init__(self, **kwargs)
        Subject.__init__(self)
        self._board_status = None

    @property
    def is_terminated(self):
        '''
        Returns True if no moves is left (either the board is full or a player has won).
        '''
        if self.board_status is None:
            return False
        return True

    @property
    def possible_actions(self):
        '''
        Returns a list of indexes of empty squares.
        '''
        actions = []
        for a in list(self.get_action_set()):
            temp = ValueSet(a)
            temp.max = len(self._board) - 1
            temp.min = 0
            actions.append(temp)
        return actions

    def register(self, player_name):
        '''
        Registers an agent and returns its ID. If the agent is new, a new ID is generated and the agent_name is added to agent_list.
        \nArguments:
        \n    agent_name: the name of the agent to be registered.
        '''
        return Subject.register(self, player_name)

    def take_effect(self, _id, action):
        '''
        Sets a piece for the given player on the board.
        \nArguments:
        \n    _id: ID of the player who sets the piece.
        \n    action: the location in which the piece is set. Can be either in index format or row column format.
        ''' 
        self.set_piece(_id, index=int(action.value[0]), update='yes')
        if self.board_status is None:
            return 0
        if self.board_status == _id:
            return 1
        if self.board_status > 0:
            return -1
        return 0

    def reset(self):
        '''
        Empties the board.
        '''
        MNKBoard.reset(self)
        self._board_status = None

    def set_piece(self, player, **kwargs):
        '''
        This function sets a piece for a player.
        You can either specify the location by index using 'index' named
        argument or row and column using 'row' and 'column'. \n
        Note that the board starts from index=0 (row=0, column=0). \n
        An attempt to set a piece for an undefined player or in an
        unspecified location will result in ValueError.
        \nArguments:
        \n    player: ID of the player whose piece will be set on the board.
        \n    update: if 'yes', the board status is updated after the move. Default is 'yes'.
        \n    index: If provided, the piece is set using 'index'. Index starts from 0 and assumes the board to be a list.
        \n    row: The row in which the piece is set.
        \n    column: The column in which the piece is set.
        '''

        MNKBoard.set_piece(self, player, **kwargs)
        try:  # update
            update = kwargs['update'].lower()
        except KeyError:
            update = 'yes'
        if update != 'yes':
            return
        if self._board_status is None:
            self._board_status = self._update_board_status(player, **kwargs)
        elif self._board_status > 0:
            self._board_status = -1

    @property
    def board_status(self):
        '''
        Returns board status. If there is a winner, the winner's ID is returned. If the game is not finished, None is returned.
        If the game is a draw, 0 is returned.
        ''' 
        return self._board_status

    def _update_board_status(self, player, **kwargs):
        # player wins: player | doesn't win: None | draw: 0
        '''
        Gets a player and the location of the latest change and tries to find a sequence of length k of the specified
        player around that location. \n
        sequence found (win): return player \n
        sequence found and board is full (stall): return 0 \n
        sequence found and board is not full (ongoing): return None
        \nArguments:
        \n    player: ID of the player whose piece will be set on the board.
        \n    index: If provided, the piece is set using 'index'. Index starts from 0 and assumes the board to be a list.
        \n    row: The row in which the piece is set.
        \n    column: The column in which the piece is set.
        '''
        if not kwargs:
            raise TypeError('No (row, column) or index found')
        try:
            r = kwargs['row']
            c = kwargs['column']
        except KeyError:
            r = kwargs['index'] // self._n
            c = kwargs['index'] % self._n

        ul_r = max(r-self._k+1, 0)
        ul_c = max(c-self._k+1, 0)
        lr_r = min(r+self._k, self._m-1)
        lr_c = min(c+self._k, self._n-1)
        m = self._matrix()

        # Vertical sequence
        pointer = ul_r
        counter = 0
        while pointer <= lr_r:
            if m[pointer][c] == player:
                counter += 1
            else:
                counter = 0
            if counter == self._k:
                return player
            pointer += 1

        # Horizontal sequence
        pointer = ul_c
        counter = 0
        while pointer <= lr_c:
            if m[r][pointer] == player:
                counter += 1
            else:
                counter = 0
            if counter == self._k:
                return player
            pointer += 1

        # Diagonal \
        min_d = min(r-ul_r, c-ul_c)
        pointer_r = r - min_d
        pointer_c = c - min_d
        counter = 0
        while (pointer_r <= lr_r) & (pointer_c <= lr_c):
            if m[pointer_r][pointer_c] == player:
                counter += 1
            else:
                counter = 0
            if counter == self._k:
                return player
            pointer_r += 1
            pointer_c += 1

        # Diagonal /
        min_d = min(r-ul_r, lr_c-c)
        pointer_r = r - min_d
        pointer_c = c + min_d
        counter = 0
        while (pointer_r <= lr_r) & (pointer_c >= ul_c):
            if m[pointer_r][pointer_c] == player:
                counter += 1
            else:
                counter = 0
            if counter == self._k:
                return player
            pointer_r += 1
            pointer_c -= 1

        if min(self._board) > 0:
            return 0

        return None


if __name__ == '__main__':
    main()
