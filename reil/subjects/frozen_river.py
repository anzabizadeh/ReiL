# -*- coding: utf-8 -*-
# TODO: Components (state, possible actions, etc.) should be updated
# according to the new implementation of `Subject`.

'''
FrozenRiver class
=================

(1D FrozenLake game).
On a straight path, you can either move one or two units to the right or left,
until you reach the goal. There are holes that you should avoid.
'''

from random import choice
from typing import Any, List, Optional, Tuple

from reil.datatypes.dataclasses import Index_FeatureArray
from reil.datatypes.feature import Feature, FeatureArray, FeatureGenerator
from reil.subjects.subject import Subject


class FrozenRiver(Subject):
    '''
    Build a 1-by-n grid in which 1 player can
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
    def __init__(
            self, _map: Optional[List[str]] = None,
            terminate_on_hole: bool = False,
            max_moves: int = -1, map_size: Optional[int] = None,
            **kwargs: Any):
        '''
        Arguments
        ---------
        map:
            the map to be used
        '''
        # default_map = [
        #     'S', 'F', 'H', 'H', 'F', 'H', 'F', 'F',
        #     'F', 'H', 'F', 'H', 'H', 'F', 'G']
        self._default_actions_values = {
            'N0': 0,
            'R1': 1,
            'R2': 2,
            'R3': 3,
            'L1': -1,
            'L2': -2}

        Subject.__init__(self, max_entity_count=1, **kwargs)
        self._terminate_on_hole = terminate_on_hole
        self._max_moves = max_moves
        if _map is None and map_size is None:
            raise ValueError('Either provide a map, or set the map size.')

        self._default_map = _map
        self._map_size = map_size

        self._board_gen = FeatureGenerator.categorical(
            name='board', categories=('S', 'G', 'F', 'H', 'P'))
        self._board_gen.__dict__['normalizer'] = {
            'S': (0.2,),
            'G': (1.0,),
            'F': (0.6,),
            'H': (0.0,),
            'P': (0.4,)}
        self._action_gen = FeatureGenerator.categorical(
            name='move', categories=tuple(self._default_actions_values))
        self._default_actions = tuple(
            FeatureArray(self._action_gen(m))
            for m in self._default_actions_values)
        self._default_actions_dict = {
            action.value['move']: Index_FeatureArray(i, action)
            for i, action in enumerate(self._default_actions)
        }

        self.state.add_definition(
            'full_map', ('full_map', {}))
        for before, after in (
                (0, 1), (0, 2), (0, 3),
                (1, 1), (1, 2),
                (2, 2), (2, 3), (3, 3)):
            self.state.add_definition(
                f'neighbor{before}{after}',
                ('neighborhood', {'before': before, 'after': after}))

        self.possible_actions.add_definition(
            'moves', self._actions, 'neighbor01')

        self.reset()

    def _actions(self, board: Feature) -> Tuple[FeatureArray, ...]:
        return self._default_actions

    def is_terminated(self, _id: Optional[int] = None) -> bool:
        '''Return True if the player get to the goal or falls into a hole.'''
        temp = self._board[self._player_loc]
        return (
            temp == 'G' or (self._terminate_on_hole and temp == 'H') or
            (self._max_moves > 0 and self._move_counter >= self._max_moves))

    def _default_reward_definition(
            self, _id: Optional[int] = None) -> float:
        if self._player_loc == self._goal:
            return 10.0
        if self._terminate_on_hole and self._board[self._player_loc] == 'H':
            return -1.0
        if self._player_loc > self._previous_loc:
            return self._goal - self._player_loc

        # return -0.1
        return -(self._goal - self._player_loc)

    def _take_effect(
            self, action: Index_FeatureArray, _id: int) -> Index_FeatureArray:
        '''
        Move according to the action.

        Arguments
        ---------
            _id: ID of the player.
            action: the location in which the piece is set. Can be either
            in index format or row column format.
        '''
        a: str = action.feature['move'].value  # type: ignore
        direction = a[0]
        loc = self._player_loc
        self._previous_loc = loc

        self._move_counter += 1

        if direction == 'L' and loc - int(a[1:]) < 0:
            return self._default_actions_dict['N0']
        if direction == 'R' and loc + int(a[1:]) > self._goal:
            return self._default_actions_dict['N0']

        self._player_loc += self._default_actions_values[a]
        if (not self._terminate_on_hole and
                self._board[self._player_loc] == 'H'):
            self._player_loc = 0

        return action

    def reset(self):
        '''Clear the board and update board_status.'''
        Subject.reset(self)
        if self._default_map:
            self._board = self._default_map
        else:
            self._board = self._generate_map()

        self._goal = len(self._board) - 1
        self._player_loc: int = 0
        self._previous_loc: int = 0
        self._move_counter: int = 0

    def _generate_map(self) -> List[str]:
        map_size: int = self._map_size  # type: ignore
        temp = ['H'] * map_size
        moves = [
            int(a[1])
            for a in self._default_actions_values
            if a[0] == 'R']

        i = 0
        while i < map_size:
            temp[i] = 'F'
            i += choice(moves)

        temp[0] = 'S'
        temp[-1] = 'G'

        return temp

    def _sub_comp_full_map(
            self, _id: int, **kwargs: Any) -> Feature:
        board_list = tuple(
            self._board[:self._player_loc] +
            ['P'] +
            self._board[self._player_loc+1:]
        )

        return self._board_gen(board_list)

    def _sub_comp_neighborhood(
            self, _id: int, before: int = 1, after: int = 1,
            **kwargs: Any) -> Feature:
        loc = self._player_loc

        temp = ['H'] * before + self._board + ['H'] * after
        board_list = temp[loc:loc+before+after+1]
        del board_list[before]

        return self._board_gen(tuple(board_list))

    def __repr__(self) -> str:
        return str(
            self._board[:self._player_loc] +
            ['P'] +
            self._board[self._player_loc+1:])


if __name__ == '__main__':
    board = FrozenRiver()
    _ = board.register('P1')
    while not board.is_terminated():
        my_action = Index_FeatureArray(
            0, choice(board.possible_actions('moves')))  # type: ignore
        board.take_effect(my_action, 1)
        print(my_action.feature.value)
        print(f'{board}')
