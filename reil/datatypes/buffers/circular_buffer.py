# -*- coding: utf-8 -*-
'''
CircularBuffer class
====================

A `Buffer` that overflows!
'''

from typing import Callable, Iterator

from reil.datatypes.buffers.buffer import (T1, T2, Buffer, Funcs, PickModes,
                                           funcs_dict)


class CircularBuffer(Buffer[T1, T2]):
    '''
    A `Buffer` that overflows.

    Extends `Buffer` class.
    '''

    def __init__(
            self, buffer_size: int | None = None,
            buffer_names: list[str] | None = None,
            pick_mode: PickModes | None = None) -> None:
        self._buffer_full: bool = False
        super().__init__(
            buffer_size=buffer_size, buffer_names=buffer_names,
            pick_mode=pick_mode)
        self._buffer_index = 0

    def add(self, data: dict[str, T1 | T2]) -> None:
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
            super().add(data)  # type: ignore
        except IndexError:
            self._buffer_full = True
            self._buffer_index = -1
            super().add(data)  # type: ignore

        # the size does not change if buffer is full.
        self._count -= self._buffer_full

    def add_iter(self, iter: Iterator[dict[str, T1 | T2]]) -> None:
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

        for data in iter:
            self._buffer_index += 1
            try:
                for key, v in data.items():
                    self._buffer[key][self._buffer_index] = v  # type: ignore
                self._count += 1
            except IndexError:
                self._buffer_full = True
                self._buffer_index = 0
                for key, v in data.items():
                    self._buffer[key][self._buffer_index] = v  # type: ignore
        if self._buffer_full:
            self._count = self._buffer_size

    def aggregate(
        self, func: Funcs | Callable[[list[T1] | list[T2]], T1 | T2],
        names: str | list[str] | None = None
    ) -> dict[str, T1 | T2]:
        if not self._buffer_full:
            return {}

        _names: list[str]
        if names is None:
            _names = self._buffer_names  # type: ignore
        elif isinstance(names, str):
            _names = [names]
        else:
            _names = names

        fn = funcs_dict[func] if isinstance(func, str) else func

        return {
            name: fn(self._buffer[name])  # type: ignore
            for name in _names
        }

    def _pick_old(self, count: int) -> dict[str, tuple[T1, ...] | tuple[T2, ...]]:
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
        else:  # filling starts from the first item, not 0th
            picks = super()._pick_old(count + 1)
            return {name: value[1:] for name, value in picks.items()}

    def _pick_recent(
        self, count: int
    ) -> dict[str, tuple[T1, ...] | tuple[T2, ...]]:
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

    def _pick_all(self) -> dict[str, tuple[T1, ...] | tuple[T2, ...]]:
        '''
        Return all items in the buffer.
        '''
        if self._buffer_full:
            self._buffer: dict[str, list[T1] | list[T2]]
            slice_pre = slice(self._buffer_index + 1, None)
            slice_post = slice(self._buffer_index + 1)
            return {name: tuple(
                buffer[slice_pre] + buffer[slice_post])  # type: ignore
                for name, buffer in self._buffer.items()}
        else:  # filling starts from the first item, not 0th
            picks = super()._pick_all()
            return {name: value[1:] for name, value in picks.items()}

    def reset(self) -> None:
        '''
        Reset the buffer.
        '''
        super().reset()
        self._buffer_index = 0
        self._buffer_full = False
