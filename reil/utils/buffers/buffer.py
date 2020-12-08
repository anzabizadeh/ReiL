# -*- coding: utf-8 -*-
'''
Buffer class
============

The base class for all buffers in `reil`.


'''

from typing import Dict, Generic, List, Optional, Tuple, TypeVar, cast

import numpy as np
from reil import reilbase
from typing_extensions import Literal

T = TypeVar('T')
PickModes = Literal['all', 'random', 'recent', 'old']


class Buffer(reilbase.ReilBase, Generic[T]):  # pylint: disable=unsubscriptable-object
    '''
    The base class for all buffers in `reil`.

    Methods
-----------
    setup: sets up the buffer by defining its size, queue names, and pick mode.

    add: adds a new item to the buffer.

    pick: picks the `count` number of items from the buffer.

    reset: resets the buffer.

    _pick_old: returns `count` oldest items in the buffer.

    _pick_recent: returns `count` most recent items in the buffer.

    _pick_random: returns `count` number of items in the buffer selected randomly.

    _pick_all: returns everything stored in the buffer.
    '''

    _buffer_size = None
    _buffer_names = None
    _pick_mode = None
    _buffer_index = -1
    _count = 0

    def __init__(self,
                 buffer_size: Optional[int] = None,
                 buffer_names: Optional[List[str]] = None,
                 pick_mode: Optional[PickModes] = None) -> None:
        '''
        Initializes the buffer.

        Arguments
-----------
        buffer_size: the size of the buffer.

        buffer_names: a list containing the names of buffer queues.

        pick_mode: the default mode to pick items from the list.
        '''
        self.setup(buffer_size, buffer_names, pick_mode)

    def setup(self,
              buffer_size: Optional[int] = None,
              buffer_names: Optional[List[str]] = None,
              pick_mode: Optional[PickModes] = None) -> None:
        '''
        Sets up the buffer.

        Arguments
-----------
        buffer_size: the size of the buffer.

        buffer_names: a list containing the names of buffer elements.

        pick_mode: the default mode to pick items from the list.

        Note: `setup` should be used only for attributes of the buffer that are
        not defined. Attempt to use `setup` to modify size, names or mode will
        result in an exception.
        '''
        if buffer_size is not None:
            if self._buffer_size is not None:
                raise ValueError(
                    'Cannot modify buffer_size. The value is already set.')
            else:
                self._buffer_size = buffer_size

        if buffer_names is not None:
            if self._buffer_names is not None:
                raise ValueError(
                    'Cannot modify buffer_names. The value is already set.')
            else:
                self._buffer_names = buffer_names

        if self._buffer_size is not None and self._buffer_names is not None:
            self._buffer = cast(Dict[str, List[T]],
                                dict((name, [0.0]*self._buffer_size)
                                     for name in self._buffer_names))
        else:
            self._buffer = None

        if pick_mode is not None:
            self._pick_mode = pick_mode

        self.reset()

    def add(self, data: Dict[str, T]) -> None:
        '''
        Adds a new item to the buffer.

        Note: this implementation of `add` does not check if the buffer is full
        or if the provided names exist in the buffer queues. As a result, this
        situations will result in exceptions by the system.

        Arguments
-----------
        data: a dictionary with the name of buffer queues as keys.
        '''
        self._buffer_index += 1
        for key, v in data.items():
            self._buffer[key][self._buffer_index] = v
        self._count += 1

    def pick(self,
             count: Optional[int] = None,
             mode: Optional[PickModes] = None) -> Dict[str, Tuple[T, ...]]:
        '''
        Returns items from the buffer.

        Arguments
-----------
        count: the number of items to return. If omitted, the number of items in
        the buffer is used. `count` will be ignored if `mode` is 'all'.

        mode: how to pick items. If omitted, the default `pick_mode` specified
        during initialization or setup is used.

        Raises `ValueError` if the number of items requested is greater than the
        number of items in the buffer.
        Raises `ValueError` if mode is not on of 'all', 'random', 'recent' and 'old'.
        '''
        _mode = mode.lower() if mode is not None else self._pick_mode
        _count = count if count is not None else self._count

        if _count > self._count:
            raise ValueError('Not enough data in the buffer.')

        if _mode == 'old':
            return self._pick_old(_count)
        elif _mode == 'recent':
            return self._pick_recent(_count)
        elif _mode == 'random':
            return self._pick_random(_count)
        elif _mode == 'all':
            return self._pick_all()
        else:
            raise ValueError(
                'mode should be one of all, old, recent, or random.')

    def reset(self) -> None:
        '''
        Resets the buffer.
        '''
        self._buffer_index = -1
        self._count = 0

    def _pick_old(self, count: int) -> Dict[str, Tuple[T, ...]]:
        '''
        Returns the oldest items in the buffer.

        Arguments
-----------
        count: the number of items to return.
        '''
        return dict((name, tuple(buffer[:count]))
                    for name, buffer in self._buffer.items())

    def _pick_recent(self, count: int) -> Dict[str, Tuple[T, ...]]:
        '''
        Returns the most recent items in the buffer.

        Arguments
-----------
        count: the number of items to return.
        '''
        return dict((name, tuple(buffer[self._buffer_index+1-count:self._buffer_index+1]))
                    for name, buffer in self._buffer.items())

    def _pick_random(self, count: int) -> Dict[str, Tuple[T, ...]]:
        '''
        Returns a random sample of items in the buffer.

        Arguments
-----------
        count: the number of items to return.
        '''
        index = np.random.choice(
            self._count, count, replace=False)
        return dict((name, tuple(buffer[i] for i in index))
                    for name, buffer in self._buffer.items())

    def _pick_all(self) -> Dict[str, Tuple[T, ...]]:
        '''
        Returns all items in the buffer.
        '''
        return dict((name, tuple(buffer[:self._buffer_index+1]))
                    for name, buffer in self._buffer.items())
