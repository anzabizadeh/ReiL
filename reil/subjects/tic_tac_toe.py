# -*- coding: utf-8 -*-
'''
TicTacToe class
===============

The standard Tic-Tac-Toe game.
'''
import random
from typing import Any, Optional

from reil import subjects
from reil.datatypes import ReilData


class TicTacToe(subjects.MNKGame):
    '''
    Build a 3-by-3 board in which 2 players can play.
    Winner is the player who can put 3 pieces in one row, column, or diagonal.
    '''
    # _board is a row vector. (row, column) and index start from 0
    # _board_status: None: no winner yet,
    #                1..players: winner,
    #                0: stall,
    #               -1: illegal board

    def __init__(self, **kwargs: Any):
        super().__init__(m=3, n=3, k=3, players=2, **kwargs)

    def default_state(self, _id: Optional[int] = None) -> ReilData:
        def modify(i, _id) -> float:
            if i == _id:
                return 1
            if i == 0:
                return 0
            return -1

        return ReilData.single_numerical(
            name='state',
            value=tuple(modify(i, _id)
                        for i in self._board),
            lower=-1, upper=1)

    def __repr__(self):
        return self.__class__.__qualname__


if __name__ == '__main__':
    board = TicTacToe()
    player = {}
    p = 0
    player['P1'] = board.register('P1')
    player['P2'] = board.register('P2')
    while not board.is_terminated():
        board.state
        current_player = ['P1', 'P2'][p]
        print(p, current_player)
        actions = board.possible_actions(player[current_player])
        board.take_effect(random.choice(actions), player[current_player])
        print(f'{board}\n', board.reward(
            'default', player['P1']), board.reward('default', player['P2']))
        p = (p + 1) % 2
