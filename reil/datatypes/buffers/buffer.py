# -*- coding: utf-8 -*-
'''
Buffer class
============

The base class for all buffers in `reil`.
'''

from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
from reil import reilbase
from typing_extensions import Literal

T = TypeVar('T')
PickModes = Literal['all', 'random', 'recent', 'old']


class Buffer(reilbase.ReilBase, Generic[T]):
    '''
    The base class for all buffers in `reil`.
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
        Arguments
        ---------
        buffer_size:
            The size of the buffer.

        buffer_names:
            A list containing the names of buffer queues.

        pick_mode:
            The default mode to pick items from the list.
        '''
        self.setup(buffer_size, buffer_names, pick_mode)

    def setup(self,
              buffer_size: Optional[int] = None,
              buffer_names: Optional[List[str]] = None,
              pick_mode: Optional[PickModes] = None) -> None:
        '''
        Set up the buffer.

        Arguments
        ---------
        buffer_size:
            The size of the buffer.

        buffer_names:
            A list containing the names of buffer elements.

        pick_mode:
            The default mode to pick items from the list.

        Raises
        ------
        ValueError:
            Cannot modify `buffer_size`. The value is already set.

        ValueError:
            Cannot modify `buffer_names`. The value is already set.

        Notes
        -----
        `setup` should be used only for attributes of the buffer that are
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
            self._buffer = {name: [0.0]*self._buffer_size
                            for name in self._buffer_names}
        else:
            self._buffer = None

        if pick_mode is not None:
            self._pick_mode = pick_mode

        self.reset()

    def add(self, data: Dict[str, Any]) -> None:
        '''
        Append a new item to the buffer.

        Arguments
        ---------
        data:
            A dictionary with the name of buffer queues as keys.

        Notes
        -----
        This implementation of `add` does not check if the buffer is full
        or if the provided names exist in the buffer queues. As a result, this
        situations will result in exceptions by the system.
        '''
        if self._buffer is None:
            raise RuntimeError('Buffer is not set up!')

        self._buffer_index += 1
        for key, v in data.items():
            self._buffer[key][self._buffer_index] = v

        self._count += 1

    def pick(self,
             count: Optional[int] = None,
             mode: Optional[PickModes] = None) -> Dict[str, Tuple[Any, ...]]:
        '''
        Return items from the buffer.

        Arguments
        ---------
        count:
            The number of items to return. If omitted, the number of items in
            the buffer is used. `count` will be ignored if `mode` is 'all'.

        mode:
            How to pick items. If omitted, the default `pick_mode` specified
            during initialization or setup is used.

        Returns
        -------
        :
            A dictionary with buffer names as keys and picked items as values.

        Raises
        ------
        ValueError:
            The number of items requested is greater than the number of items
            in the buffer.

        ValueError:
            `mode` is not one of 'all', 'random', 'recent' and 'old'.
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
        Reset the buffer.
        '''
        self._buffer_index = -1
        self._count = 0

    def _pick_old(self, count: int) -> Dict[str, Tuple[Any, ...]]:
        '''
        Return the oldest items in the buffer.

        Arguments
        ---------
        count:
            The number of items to return.

        Returns
        -------
        :
            A dictionary with buffer names as keys and `count` oldest items
            as values.
        '''
        s = slice(count)
        return {name: tuple(buffer[s])
                for name, buffer in self._buffer.items()}

    def _pick_recent(self, count: int) -> Dict[str, Tuple[Any, ...]]:
        '''
        Return the most recent items in the buffer.

        Arguments
        ---------
        count:
            The number of items to return.

        Returns
        -------
        :
            A dictionary with buffer names as keys and `count` most recent
            items as values.
        '''
        s = slice(self._buffer_index - count + 1, self._buffer_index + 1)
        return {name: tuple(buffer[s])
                for name, buffer in self._buffer.items()}

    def _pick_random(self, count: int) -> Dict[str, Tuple[Any, ...]]:
        '''
        Return a random sample of items in the buffer.

        Arguments
        ---------
        count:
            The number of items to return.

        Returns
        -------
        :
            A dictionary with buffer names as keys and `count` randomly picked
            items as values.
        '''
        index = np.random.choice(
            self._count, count, replace=False)
        return {name: tuple(buffer[i] for i in index)
                for name, buffer in self._buffer.items()}

    def _pick_all(self) -> Dict[str, Tuple[Any, ...]]:
        '''
        Return all items in the buffer.

        Returns
        -------
        :
            A dictionary with buffer names as keys and all items as values.
        '''
        s = slice(self._buffer_index + 1)
        return {name: tuple(buffer[s])
                for name, buffer in self._buffer.items()}
