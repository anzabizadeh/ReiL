# -*- coding: utf-8 -*-
'''
MNKGame class
==============

This `subject` class emulates mnk game. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''


from typing import Optional, Tuple
from reil.utils.mnkboard import MNKBoard
from reil.subjects.subject import Subject
from reil.datatypes import reildata
import random

def main():
    board = MNKGame()
    player = {}
    p = 0
    player['P1'] = board.register('P1')
    player['P2'] = board.register('P2')
    while not board.is_terminated():
        current_player = ['P1', 'P2'][p]
        print(p, current_player)
        actions = board.possible_actions(player[current_player])
        board.take_effect(random.choice(actions), player[current_player])
        print(f'{board}\n', board.reward(
            'default', player['P1']), board.reward('default', player['P2']))
        p = (p + 1) % 2


class MNKGame(MNKBoard, Subject):
    '''
    Build an m-by-n board (using mnkboard super class) in which p players can play.
    Winner is the player who can put k pieces in on row, column, or diagonal.

    ### Attributes
    is_terminated: whether the game finished or not.

    possible_actions: a list of possible actions.

    ### Methods
    register: register a new player and return its ID or return ID of an existing player.

    take_effect: set a piece of the specified player on the specified square of the board.

    set_piece: set a piece of the specified player on the specified square of the board.

    reset: clear the board.
    '''
    # _board is a row vector. (row, column) and index start from 0
    # _board_status: None: no winner yet,
    #                1..players: winner,
    #                0: stall,
    #               -1: illegal board
    def __init__(self, **kwargs):
        '''
        Initialize an instance of mnkgame.

        ### Arguments
        m: number of rows (default=3)

        n: number of columns (default=3)

        k: winning criteria (default=3)

        players: number of players (default=2)
        '''
        self._board_status = None
        self.set_params(**kwargs)
        MNKBoard.__init__(self, **kwargs, can_recapture=False)
        Subject.__init__(self, **kwargs)

    def is_terminated(self, _id: Optional[int] = None) -> bool:
        '''Return True if no moves is left (either the board is full or a player has won).'''
        return self._board_status is not None

    def possible_actions(self, _id: Optional[int] = None) -> Tuple[reildata.ReilData, ...]:
        '''Return a list of indexes of empty squares.'''
        return tuple(reildata.ReilData({'name': 'square', 'value': v,
                                    'lower': 0, 'upper': len(self._board)-1})
                                    for v in self.get_action_set())

    def take_effect(self, action: reildata.ReilData, _id: Optional[int] = None) -> None:
        '''
        Set a piece for the given player on the board.

        Arguments
        ---------
            _id: ID of the player who sets the piece.
            action: the location in which the piece is set. Can be either in index format or row column format.
        ''' 
        self._set_piece(_id, index=int(
            action.value['square']), update='yes')

    def default_reward(self, _id: Optional[int] = None) -> reildata.ReilData:
        if self._board_status is None:
            r = 0
        elif self._board_status == _id:
            r = 1
        elif self._board_status > 0:
            r = -1
        else:
            r = 0

        return reildata.ReilData.single_member(name='reward', value=r)

    def reset(self):
        '''Clear the board and update board_status.'''
        MNKBoard.reset(self)
        self._board_status = None

    def _set_piece(self, player, **kwargs):
        '''
        Set a piece for a player.

        Arguments
        ---------
            player: ID of the player whose piece will be set on the board.
            index: If provided, the piece is set using 'index'. Index starts from 0 and assumes the board to be a list.
            row: The row in which the piece is set.
            column: The column in which the piece is set.
            update: if 'yes', the board status is updated after the move. (Default = 'yes')

        Raises ValueError if the player or the location is out of range or niether index nor row-column is provided.
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

    def _update_board_status(self, player, **kwargs):
        # player wins: player | doesn't win: None | draw: 0
        '''
        Get a player and the location of the latest change and try to find a sequence of length k of the specified player.

        Arguments
        ---------
            player: ID of the player whose latest move is being evaluated.
            index: If provided, the piece is set using 'index'. Index starts from 0 and assumes the board to be a list.
            row: The row in which the piece is set.
            column: The column in which the piece is set.

        Return
        ------
            0: sequence not found and the board is full (stall)
            player: sequence found (win)
            None: sequence not found and the board is not full (ongoing)
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

    def __repr__(self):
        return self.__class__.__qualname__

if __name__ == '__main__':
    main()
