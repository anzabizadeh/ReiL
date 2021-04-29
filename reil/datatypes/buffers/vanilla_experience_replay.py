# -*- coding: utf-8 -*-
'''
VanillaExperienceReplay class
=============================

A `Buffer` with random pick that picks only if it is full.
'''

from typing import Dict, List, Optional, Tuple

from reil.datatypes.buffers import CircularBuffer, T


class VanillaExperienceReplay(CircularBuffer[T]):
    '''
    A `Buffer` with random pick that picks only if it is full.

    Extends `CircularBuffer` class.
    '''

    def __init__(self,
                 buffer_size: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 buffer_names: Optional[List[str]] = None,
                 clear_buffer: bool = False) -> None:
        '''
        Initialize the buffer.

        Arguments
        -----------
        buffer_size:
            The size of the buffer.

        batch_size:
            The number of items to return at each `pick`.

        buffer_names:
            A list containing the names of buffer queues.

        clear_buffer:
            Whether to clear the buffer when `reset` is called.
        '''
        self._batch_size = None
        self._clear_buffer = False

        super().__init__(buffer_size=buffer_size,
                         buffer_names=buffer_names, pick_mode='random')

        self.setup(buffer_size=buffer_size,
                   batch_size=batch_size,
                   buffer_names=buffer_names,
                   clear_buffer=clear_buffer)

    def setup(self, **kwargs) -> None:
        '''
        Set up the buffer.

        Arguments
        ---------
        buffer_size:
            The size of the buffer.

        batch_size:
            The number of items to return at each `pick`.

        buffer_names:
            A list containing the names of buffer elements.

        clear_buffer:
            Whether to clear the buffer when `reset` is called.

        Notes
        -----
        `setup` should be used only for attributes of the buffer that are
        not defined. Attempt to use `setup` to modify size, names or mode will
        result in an exception.
        '''
        buffer_size = kwargs.get('buffer_size')
        batch_size = kwargs.get('batch_size')
        buffer_names = kwargs.get('buffer_names')
        clear_buffer = kwargs.get('clear_buffer')

        super().setup(buffer_size=buffer_size,
                      buffer_names=buffer_names)

        if buffer_size is not None and buffer_size < 1:
            raise ValueError('buffer_size should be at least 1.')

        if self._buffer_size is not None:
            if batch_size is not None and self._buffer_size < batch_size:
                raise ValueError('buffer_size should be >= batch_size.')

        if batch_size is not None:
            if (self._buffer_size is not None and
                    self._buffer_size < batch_size):
                raise ValueError('buffer_size should be >= batch_size.')
            if self._batch_size not in (None, batch_size):
                raise ValueError(
                    'Cannot modify batch_size. The value is already set.')
            else:
                self._batch_size = batch_size

        self._clear_buffer = clear_buffer

    def pick(self) -> Dict[str, Tuple[T, ...]]:
        '''
        Return `batch_size` number of items from the buffer randomly.
        If the buffer is not full, return empty tuples.
        '''
        if self._buffer_full:
            return super().pick(self._batch_size, 'random')
        else:
            return {name: () for name in self._buffer}

    def reset(self) -> None:
        '''
        Reset the buffer if `clear_buffer` is set to `True`.
        '''
        if self._clear_buffer:
            super().reset()
            self._buffer_full = False
