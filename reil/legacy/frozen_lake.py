# -*- coding: utf-8 -*-
# TODO: Components (state, possible actions, etc.) should be updated
# according to the new implementation of `Subject`.

'''
FrozenLake class
=================

This class creates a frozen lake (board) in which one square is the goal.
The `agent` starts from a location and should find the fastest route to
the goal. Some locations are holes.
'''

from random import choice
from typing import Any, Dict, List, Optional, Tuple

from reil.utils.mnkboard import MNKBoard
from reil.subjects.subject import Subject
from reil.datatypes.feature import Feature, FeatureArray


class FrozenLake(MNKBoard, Subject):
    '''
    Build an m-by-n grid (using mnkboard super class) in which 1 player can
    play. Player wins if it can get to the goal square. Each element in the
    graph can be:
        S: starting point, safe
        F: frozen surface, safe
        H: hole (end with reward -1)
        G: goal (end with reward 1)

    Attributes
    ----------
        is_terminated: whether the game finished or not.
        possible_actions: a list of possible actions.

    Methods
    -------
        register: register the player and return its ID or return ID of current
        player.

        take_effect: moves the player on the grid.

        reset: clears the grid.
    '''
    def __init__(self, map: Optional[List[List[str]]] = None, **kwargs: Any):
        '''
        Arguments
        ---------
        map:
            the map to be used
        '''

        default_map = [
            ['S', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'H'],
            ['F', 'F', 'F', 'H'],
            ['H', 'F', 'F', 'G']]

        def locate(map: List[List[Any]], element: Any) -> Tuple[int, int]:
            row = [element in m_i
                   for m_i in map].index(True)
            col = map[row].index(element)
            return (row, col)

        self._map = map if map is not None else default_map

        self._dim = (len(self._map), len(self._map[0]))
        self._start = locate(self._map, 'S')
        self._goal = locate(self._map, 'G')

        moves = ('U', 'D', 'R', 'L')
        self._default_moves = tuple(
            Feature.categorical(name='move', value=m, categories=moves)
            for m in moves)

        Subject.__init__(self, max_entity_count=1, **kwargs)
        MNKBoard.__init__(self, m=self._dim[0], n=self._dim[1], players=1)
        self.reset()

    def is_terminated(self, _id: Optional[int] = None) -> bool:
        '''Return True if the player get to the goal.'''
        return self._player_location == self._goal

    def possible_actions(self, _id: int = 0) -> Tuple[Feature, ...]:
        '''Return the set of possible moves.'''
        return self._default_moves

    def default_reward(self, _id: int = 0) -> Feature:
        return Feature.numerical(
            name='reward',
            value=(float(self._player_location == self._goal) - 0.5) * 2)

    def _take_effect(self, action: FeatureArray, _id: int = 0) -> None:
        '''
        Move according to the action.

        Arguments
        ---------
            _id: ID of the player.
            action: the location in which the piece is set. Can be either
            in index format or row column format.
        '''
        Subject.take_effect(self, action, _id)

        row, column = self._player_location
        max_row = self._dim[0] - 1
        max_column = self._dim[1] - 1
        MNKBoard.clear_square(self, row=row, column=column)

        a = str(*action.value)
        temp = (
            row - (a == ['U', 'UR', 'UL']) + (a in ['D', 'DR', 'DL']),
            column - (a in ['L', 'UL', 'DL']) + (a in ['R', 'UR', 'DR'])
        )

        self._player_location = (
            min(max(temp[0], 0), max_row),
            min(max(temp[1], 0), max_column)
        )

        if self._map[
                self._player_location[0]][self._player_location[1]] == 'H':
            self._player_location = self._start

        MNKBoard.set_piece(
            self, player=1,
            row=self._player_location[0],
            column=self._player_location[1])

    def reset(self):
        '''Clear the board and update board_status.'''
        Subject.reset(self)
        MNKBoard.reset(self)

        self._player_location = self._start
        MNKBoard.set_piece(self, player=1,
                           row=self._player_location[0],
                           column=self._player_location[1])

    def _generate_state_components(self) -> None:
        def full_map(**kwargs: Any) -> Dict[str, Any]:
            return {'name': 'full_map',
                    'value': self._map,
                    'categories': ('S', 'F', 'H', 'G')}

        self._available_state_components = {
            'full_map': full_map,
        }


if __name__ == '__main__':
    board = FrozenLake()
    _ = board.register('P1')
    for _ in range(10):
        print(board.state)
        my_action = FeatureArray(
            choice(board.possible_actions))  # type: ignore
        board.take_effect(my_action, 1)
        print(my_action.value)
        print(f'{board}')
