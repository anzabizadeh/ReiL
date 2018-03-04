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
        return self._board

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

    @property
    def possible_actions(self):
        '''
        Returns a list of indexes of empty squares.
        '''  
        return list(self.get_action_set())

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
        self.set_piece(_id, index=int(action), update='yes')
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


import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint

class Snake(MNKBoard, Subject):
    def __init__(self, **kwargs):
        try:
            self._m = kwargs['m']
        except KeyError:
            self._m = 10
        try:
            self._n = kwargs['n']
        except KeyError:
            self._n = 10

        MNKBoard.__init__(self, m=self._m, n=self._n, players=2)  # player 1 is the snake, player 2 is the fruit!
        self.reset()

    @property
    def is_terminated(self):
        return (self._snake[0] in self._snake[1:])
    

    @property
    def possible_actions(self):
        '''
        Returns a list of indexes of empty squares.
        '''  
        return ['left', 'none', 'right']

    def take_effect(self, _id, action):
        self._win.border(0)
        self._win.addstr(0, 2, 'Score : ' + str(self._score) + ' ')                # Printing 'Score' and
        self._win.addstr(0, 27, ' SNAKE ')                                   # 'SNAKE' strings
        self._win.timeout(150 - int(len(self._snake)/5 + len(self._snake)/10)%120)          # Increases the speed of Snake as its length increases
        
        # prevKey = self._key                                                  # Previous key pressed
        _ = self._win.getch()
        if action == 'left':
            if self._key == KEY_LEFT:
                self._key = KEY_DOWN
            elif self._key == KEY_DOWN:
                self._key = KEY_RIGHT
            elif self._key == KEY_RIGHT:
                self._key = KEY_UP
            elif self._key == KEY_UP:
                self._key = KEY_LEFT
        elif action == 'right':
            if self._key == KEY_LEFT:
                self._key = KEY_UP
            elif self._key == KEY_UP:
                self._key = KEY_RIGHT
            elif self._key == KEY_RIGHT:
                self._key = KEY_DOWN
            elif self._key == KEY_DOWN:
                self._key = KEY_LEFT

        # if key == ord(' '):                                            # If SPACE BAR is pressed, wait for another
        #     key = -1                                                   # one (Pause/Resume)
        #     while key != ord(' '):
        #         key = win.getch()
        #     key = prevKey
        #     continue

        # if key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, 27]:     # If an invalid key is pressed
        #     key = prevKey

        # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases.
        # This is taken care of later at [1].
        self._snake.insert(0, [self._snake[0][0] + (self._key == KEY_DOWN and 1) + (self._key == KEY_UP and -1), self._snake[0][1] + (self._key == KEY_LEFT and -1) + (self._key == KEY_RIGHT and 1)])
        if self._snake[0][0] == 0: self._snake[0][0] = self._m - 2
        if self._snake[0][1] == 0: self._snake[0][1] = self._n - 2
        if self._snake[0][0] == self._m: self._snake[0][0] = 1
        if self._snake[0][1] == self._n: self._snake[0][1] = 1
        # print(self._snake[0][0], self._snake[0][1], (self._snake[0][0] * 6 + self._snake[0][1]))
        self.set_piece(1, row=self._snake[0][0], column=self._snake[0][1])
        if self._snake[0] == self._food:                                            # When snake eats the food
            self._food = []
            self._score += 1
            while self._food == []:
                self._food = [randint(1, self._m-2), randint(1, self._n-2)]                 # Calculating next food's coordinates
                if self._food in self._snake: self._food = []
            self._win.addch(self._food[0], self._food[1], '*')
            self.set_piece(2, row=self._food[0], column=self._food[1])
        else:
            last = self._snake.pop()                                          # [1] If it does not eat the food, length decreases
            self._win.addch(last[0], last[1], ' ')
            self.clear_piece(row=last[0], column=last[1])
        self._win.addch(self._snake[0][0], self._snake[0][1], '#')
        return self._score

    def reset(self):
        MNKGame.reset(self)
        curses.initscr()
        self._win = curses.newwin(20, 60, 0, 0)
        self._win.keypad(1)
        curses.noecho()
        curses.curs_set(0)
        self._win.border(0)
        self._win.nodelay(1)
        self._key = KEY_RIGHT                                                    # Initializing values
        self._score = 0
        self._snake = [[self._m // 2, self._n // 2 + 1],
                       [self._m // 2, self._n // 2],
                       [self._m // 2, self._n // 2 - 1]]                                     # Initial snake co-ordinates
        self._food = [self._m // 2 + 1, self._n // 2]                                                     # First food co-ordinates
        self._win.addch(self._food[0], self._food[1], '*')                       # Prints the food
        Subject.__init__(self)
        for location in self._snake:
            self.set_piece(1, row=location[0], column=location[1])
        self.set_piece(2, row=self._food[0], column=self._food[1])

    def printable(self):
        pass

         

if __name__ == '__main__':
    main()
