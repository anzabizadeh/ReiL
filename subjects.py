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
    '''
    Constructs a board and fills it randomly until games finishes.
    '''
    import random as r
    row, col, criteria, players = 6, 8, 4, 4
    board = MNKGame(m=row, n=col, k=criteria, p=players)
    while board.board_status is None:
        player = r.randint(1, players)
        i = r.choice(list(board.get_action_set()))
        print(player, i)
        board.set_piece(player, index=i)
        print(board.printable(), '\n', board.board_status)
    board.reset()


class Subject:
    '''
    Super class of all subject classes. This class provides basic methods including is_terminated, register, take_effect, reset, load, save, and printable.
    Register, load and save are implemented.
    '''
    def __init__(self):
        self._agent_list = {}

    @property
    def is_terminated(self):
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

    @property
    def state(self):
        '''
        Returns the state of the board as a list.
        '''
        return self._board

    @property
    def possible_actions(self):
        '''
        Returns a list of indexes of empty squares.
        '''  
        return list(self.get_action_set())

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
        self.set_piece(_id, index=action, update='yes')
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
