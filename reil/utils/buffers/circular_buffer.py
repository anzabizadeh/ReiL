# -*- coding: utf-8 -*-
'''
CircularBuffer class
===================

A `Buffer` that overflows!


'''

from typing import Dict, Tuple

from reil.utils.buffers import Buffer, T


class CircularBuffer(Buffer[T]):
    '''
    A `Buffer` that overflows.

    Extends `Buffer` class.

    Methods
-----------
    setup: sets up the buffer by defining its size, queue names and pick mode.

    add: adds a new item to the buffer. Extends `Buffer.add`.

    pick: picks the `count` number of items from the buffer.

    reset: resets the buffer. Extends `Buffer.reset`.

    _pick_old: returns `count` oldest items in the buffer. Extends `Buffer._pick_old`.

    _pick_recent: returns `count` most recent items in the buffer. Extends `Buffer._pick_old`.

    _pick_all: returns everything stored in the buffer. Extends `Buffer._pick_old`.
    '''
    _buffer_full = False

    def add(self, data: Dict[str, T]) -> None:
        '''
        Adds a new item to the buffer.

        If the buffer is full, new items will be writen over the oldest one.

        Arguments
-----------
        data: a dictionary with the name of buffer queues as keys.
        '''
        try:
            super().add(data)
        except IndexError:
            self._buffer_full = True
            self._buffer_index = -1
            super().add(data)

        # the size does not change if buffer is full.
        self._count -= self._buffer_full

    def _pick_old(self, count: int) -> Dict[str, Tuple[T, ...]]:
        '''
        Returns the oldest items in the buffer.

        Arguments
-----------
        count: the number of items to return.
        '''
        if self._buffer_full:
            return dict((name, tuple(
                buffer[self._buffer_index+1:self._buffer_index+count+1] +
                buffer[:max(0, count - (self._buffer_size - self._buffer_index) + 1)]))
                for name, buffer in self._buffer.items())
        else:
            return super()._pick_old(count)

    def _pick_recent(self, count: int) -> Dict[str, Tuple[T, ...]]:
        '''
        Returns the most recent items in the buffer.

        Arguments
-----------
        count: the number of items to return.
        '''
        if count - self._buffer_index <= 1 or not self._buffer_full:
            return super()._pick_recent(count)
        else:
            return dict((name, tuple(
                buffer[-(count - self._buffer_index - 1):] +
                buffer[:self._buffer_index + 1]))
                for name, buffer in self._buffer.items())

    def _pick_all(self) -> Dict[str, Tuple[T, ...]]:
        '''
        Returns all items in the buffer.
        '''
        if self._buffer_full:
            return dict((name, tuple(
                         buffer[self._buffer_index+1:] +
                         buffer[:self._buffer_index+1]))
                        for name, buffer in self._buffer.items())
        else:
            return super()._pick_all()

    def reset(self) -> None:
        '''
        Resets the buffer.
        '''
        super().reset()
        self._buffer_full = False
