# -*- coding: utf-8 -*-
'''
CircularBuffer class
====================

A `Buffer` that overflows!
'''

from typing import Dict, List, Optional, Tuple, Union

from reil.datatypes.buffers.buffer import Buffer, PickModes, T_1, T_2


class CircularBuffer(Buffer[T_1, T_2]):
    '''
    A `Buffer` that overflows.

    Extends `Buffer` class.
    '''

    def __init__(
            self, buffer_size: Optional[int] = None,
            buffer_names: Optional[List[str]] = None,
            pick_mode: Optional[PickModes] = None) -> None:
        self._buffer_full: bool = False
        super().__init__(buffer_size=buffer_size, buffer_names=buffer_names,
                         pick_mode=pick_mode)

    def add(self, data: Dict[str, Union[T_1, T_2]]) -> None:
        '''
        Add a new item to the buffer.

        Arguments
        ---------
        data:
            A dictionary with the name of buffer queues as keys.

        Notes
        -----
        If the buffer is full, new items will be writen over the oldest one.
        '''
        try:
            super().add(data)
        except IndexError:
            self._buffer_full = True
            self._buffer_index = -1
            super().add(data)

        # the size does not change if buffer is full.
        self._count -= self._buffer_full

    def _pick_old(
        self, count: int
    ) -> Dict[str, Union[Tuple[T_1, ...], Tuple[T_2, ...]]]:
        '''
        Return the oldest items in the buffer.

        Arguments
        ---------
        count:
            The number of items to return.
        '''
        if self._buffer_full:
            self._buffer_size: int
            slice_pre = slice(self._buffer_index + 1,
                              self._buffer_index + count + 1)
            slice_post = slice(
                max(0, count - (self._buffer_size - self._buffer_index) + 1))

            return {name: tuple(
                buffer[slice_pre] + buffer[slice_post])  # type: ignore
                for name, buffer in self._buffer.items()}
        else:
            return super()._pick_old(count)

    def _pick_recent(
        self, count: int
    ) -> Dict[str, Union[Tuple[T_1, ...], Tuple[T_2, ...]]]:
        '''
        Return the most recent items in the buffer.

        Arguments
        ---------
        count:
            The number of items to return.
        '''
        if count - self._buffer_index <= 1 or not self._buffer_full:
            return super()._pick_recent(count)
        elif self._buffer is not None:
            slice_pre = slice(-(count - self._buffer_index - 1), None)
            slice_post = slice(self._buffer_index + 1)
            return {name: tuple(
                buffer[slice_pre] + buffer[slice_post])  # type: ignore
                for name, buffer in self._buffer.items()}
        else:
            raise RuntimeError('Buffer is not set up.')

    def _pick_all(self) -> Dict[str, Union[Tuple[T_1, ...], Tuple[T_2, ...]]]:
        '''
        Return all items in the buffer.
        '''
        if self._buffer_full:
            self._buffer: Dict[str, Union[List[T_1], List[T_2]]]
            slice_pre = slice(self._buffer_index + 1, None)
            slice_post = slice(self._buffer_index + 1)
            return {name: tuple(
                buffer[slice_pre] + buffer[slice_post])  # type: ignore
                for name, buffer in self._buffer.items()}
        else:
            return super()._pick_all()

    def reset(self) -> None:
        '''
        Reset the buffer.
        '''
        super().reset()
        self._buffer_full = False
