# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 21:35:10 2018

@author: Sadjad

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


def main():
    ''' constructs a board and fills it randomly until games finishes.  '''
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
    def __init__(self):
        self._player_list = {}

    @property
    def is_terminated(self):
        pass

    def register(self, player_name):
        try:
            return self._player_list[player_name]
        except KeyError:
            _id = len(self._player_list) + 1
            self._player_list[player_name] = _id
            return _id

    def take_effect(self, _id, action):
        pass

    def reset(self):
        pass

    def load(self, **kwargs):
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')
        with open(filename + '.pkl', 'rb') as f:
            self.__dict__ = pickle.load(f)

    def save(self, **kwargs):
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
        ''' Initializes an instance of mnkgame. \n
            m: number of rows \n
            n: number of columns \n
            k: winning criteria \n
            players: number of players'''

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
        ''' This function sets a piece for a player.
            You can either specify the location by index using 'index' named
            argument or row and column using 'row' and 'column'. \n
            Note that the board starts from index=0 (row=0, column=0). \n
            An attempt to set a piece for an undefined player or in an
            unspecified location will result in ValueError.'''

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
        return self._board

    @property
    def possible_actions(self):
        return list(self.get_action_set())

    def get_board(self, format_='vector'):
        if format_.lower() == 'vector':
            return self._board
        elif format_.lower() == 'matrix':
            return self._matrix(self._board)
        else:
            raise TypeError('format should be either vector or matrix')

    def get_action_set(self, format_='vector'):
        index = (i for i in range(self._m*self._n) if self._board[i] == 0)
        for action in index:
            if format_.lower() == 'vector':
                yield action
            elif format_.lower() == 'matrix':
                yield [action // self._n, action % self._n]
            else:
                raise TypeError('format should be either vector or matrix')

    def reset(self):
        self._board = [0]*(self._m*self._n)

    def printable(self):
        return ('\n'.join([''.join(['{:4}'.format(item) for item in row])
                           for row in self._matrix()]))

    def _matrix(self, *args):
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
    '''mnkgame class builds an m by n board (using mnkboard superclass)
       in which p players can play. Winner is the player who can put k pieces
       in on row, column, or diagonal.
       '''
    # _board is a row vector. (row, column) and index start from 0
    # _board_status: None: no winner yet,
    #                1..players: winner,
    #                0: stall,
    #               -1: illegal board
    def __init__(self, **kwargs):
        ''' Initializes an instance of mnkgame. \n
            m: number of rows \n
            n: number of columns \n
            k: winning criteria \n
            players: number of players'''
        MNKBoard.__init__(self, **kwargs)
        Subject.__init__(self)
        self._board_status = None

    @property
    def is_terminated(self):
        if self.board_status is None:
            return False
        return True

    def register(self, player_name):
        return Subject.register(self, player_name)

    def take_effect(self, _id, action):
        self.set_piece(_id, index=action, update='yes')
        if self.board_status is None:
            return 0
        if self.board_status == _id:
            return 1
        if self.board_status > 0:
            return -1
        return 0

    def reset(self):
        MNKBoard.reset(self)
        self._board_status = None

    def set_piece(self, player, **kwargs):
        ''' This function sets a piece for a player.
            You can either specify the location by index using 'index' named
            argument or row and column using 'row' and 'column'. \n
            Note that the board starts from index=0 (row=0, column=0). \n
            An attempt to set a piece for an undefined player or in an
            unspecified location will result in ValueError.'''

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
        return self._board_status

    def _update_board_status(self, player, **kwargs):
        # player wins: player | doesn't win: None | Stall: 0
        '''_update_board_status gets a player and the location of the latest
            change and tries to find a sequence of length k of the specified
            player around that location. \n
            sequence found (win): return player \n
            sequence found and board is full (stall): return 0 \n
            sequence found and board is not full (ongoing): return None'''
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
