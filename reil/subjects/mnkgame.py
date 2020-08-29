# -*- coding: utf-8 -*-
'''
MNKGame class
==============

This `subject` class emulates mnk game. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''


from ..utils.mnkboard import MNKBoard
from ..subjects.subject import Subject
# from reil.valueset import ValueSet
from ..rldata import RLData


def main():
    board = MNKGame()
    player = {}
    player['P1'] = board.register('P1')
    player['P2'] = board.register('P2')
    board.take_effect(RLData(0), player['P1'])  # ValueSet(0))
    board.take_effect(RLData(1), player['P2'])  # ValueSet(1))
    print(f'{board}')


class MNKGame(MNKBoard, Subject):
    '''
    Build an m-by-n board (using mnkboard super class) in which p players can play.
    Winner is the player who can put k pieces in on row, column, or diagonal.

    Attributes
    ----------
        is_terminated: whether the game finished or not.
        possible_actions: a list of possible actions.

    Methods
    -------
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

        Arguments
        ---------
            m: number of rows (default=3)
            n: number of columns (default=3)
            k: winning criteria (default=3)
            players: number of players (default=2)
        '''
        self.set_defaults(board_status=None)
        self.set_params(**kwargs)
        super().__init__(**kwargs, can_recapture=False)
        super().__init__(**kwargs)

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._board_status = None

    @property
    def is_terminated(self):
        '''Return True if no moves is left (either the board is full or a player has won).'''
        if self._board_status is None:
            return False
        return True

    @property
    def possible_actions(self):
        '''Return a list of indexes of empty squares.'''
        # return ValueSet(list(self.get_action_set()), min=0, max=len(self._board)-1).as_valueset_array()
        return RLData(list(self.get_action_set()), lower=0, upper=len(self._board)-1).as_rldata_array()

    def register(self, player_name):
        '''
        Register an agent and return its ID.
        
        If the agent is new, a new ID is generated and the agent_name is added to agent_list.
        Arguments
        ---------
            agent_name: the name of the agent to be registered.
        '''
        return Subject.register(self, player_name)

    def take_effect(self, action, _id):
        '''
        Set a piece for the given player on the board.

        Arguments
        ---------
            _id: ID of the player who sets the piece.
            action: the location in which the piece is set. Can be either in index format or row column format.
        ''' 
        self._set_piece(_id, index=int(action.value[0]), update='yes')
        if self._board_status is None:
            return 0
        if self._board_status == _id:
            return 1
        if self._board_status > 0:
            return -1
        return 0

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
        return 'MNKGame'

if __name__ == '__main__':
    main()
