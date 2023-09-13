# -*- coding: utf-8 -*-
'''
Sink class
==========

A dummy buffer that does nothing!
'''

from typing import Iterator

from typing_extensions import Unpack

from reil.datatypes.buffers.buffer import AnyTs, Buffer, PickModes, Ts


class Sink(Buffer[Unpack[Ts]]):
    '''
    A sink class.
    '''
    def __init__(self, buffer_names: list[str] | None = None) -> None:
        '''
        Arguments
        ---------
        buffer_names:
            A list containing the names of buffer queues.
        '''
        self._buffer_names = None
        self.setup(buffer_names=buffer_names)

    def setup(
            self, buffer_size: int | None = None,
            buffer_names: list[str] | None = None,
            pick_mode: PickModes | None = None,
            clear_buffer: bool | None = None) -> None:
        '''
        Set up the buffer.

        Arguments
        ---------
        buffer_size:
            The size of the buffer. This argument is only available for
            signature consistency. Assigning it has no effect.

        buffer_names:
            A list containing the names of buffer elements.

        pick_mode:
            The default mode to pick items from the list. This argument is
            only available for signature consistency. Assigning it has
            no effect.

        clear_buffer:
            Whether to clear the buffer when `reset` is called. This argument
            is only available for signature consistency. Assigning it has no
            effect.

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
        super().setup(buffer_names=buffer_names)

    def add(self, data: dict[str, AnyTs]) -> None:
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
        return

    def add_iter(self, iter: Iterator[dict[str, AnyTs]]) -> None:
        return

    def pick(
        self, count: int | None = None, mode: PickModes | None = None
    ) -> dict[str, tuple[AnyTs, ...]]:
        '''
        Raises an exception.

        Arguments
        ---------
        count:
            The number of items to return. If omitted, the number of items in
            the buffer is used. `count` will be ignored if `mode` is 'all'.

        mode:
            How to pick items. If omitted, the default `pick_mode` specified
            during initialization or setup is used.

        Raises
        ------
        TypeError:
            Cannot pick from a `Sink`.
        '''
        raise TypeError('Cannot pick from a Sink.')

    def reset(self) -> None:
        '''
        Reset the buffer.
        '''
        pass
