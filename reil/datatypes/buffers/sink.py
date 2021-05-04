# -*- coding: utf-8 -*-
'''
Sink class
==========

A dummy buffer that does nothing!
'''

from typing import Any, Dict, List, Optional

from reil.datatypes.buffers import Buffer, PickModes


class Sink(Buffer):
    '''
    A sink class.
    '''
    def __init__(self, buffer_names: Optional[List[str]] = None) -> None:
        '''
        Arguments
        ---------
        buffer_names:
            A list containing the names of buffer queues.
        '''
        self._buffer_names = None
        self.setup(buffer_names)

    def setup(self, buffer_names: Optional[List[str]] = None) -> None:
        '''
        Set up the buffer.

        Arguments
        ---------
        buffer_names:
            A list containing the names of buffer elements.

        Raises
        ------
        ValueError:
            Cannot modify `buffer_names`. The value is already set.

        Notes
        -----
        `setup` should be used only for attributes of the buffer that are
        not defined. Attempt to use `setup` to modify size, names or mode will
        result in an exception.
        '''
        if buffer_names is not None:
            if self._buffer_names is not None:
                raise ValueError(
                    'Cannot modify buffer_names. The value is already set.')
            else:
                self._buffer_names = buffer_names

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
        return

    def pick(self,
             count: Optional[int] = None,
             mode: Optional[PickModes] = None) -> None:
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
