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

import curses
from curses import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP
from random import randint

from ..valueset import ValueSet
from .mnkboard import MNKBoard
from .mnkgame import MNKGame
from .subject import Subject


def main():
    pass


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
        return ValueSet('left', 'none', 'right')

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
