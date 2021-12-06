# -*- coding: utf-8 -*-
'''
TicTacToe class
===============

The standard Tic-Tac-Toe game.
'''
import random
from typing import Any, Dict

from reil.datatypes.dataclasses import Index_FeatureArray
from reil.datatypes.feature import FeatureGenerator
from reil.subjects.mnkgame import MNKGame


class TicTacToe(MNKGame):
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
        self._state_gen = FeatureGenerator.numerical(
            name='state', lower=-1, upper=1)


if __name__ == '__main__':
    board = TicTacToe()
    player: Dict[str, int] = {}
    p = 0
    player['P1'] = board.register('P1')
    player['P2'] = board.register('P2')
    while not board.is_terminated():
        current_player = ['P1', 'P2'][p]
        print(p, current_player)
        actions = board.possible_actions(
            'square', player[current_player]) or ()
        index = random.randrange(0, len(actions))
        board.take_effect(
            Index_FeatureArray(index, actions[index]),
            player[current_player])
        print(f'{board}\n',
              board.reward('default', player['P1']),
              board.reward('default', player['P2']))
        p = (p + 1) % 2
