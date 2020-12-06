# -*- coding: utf-8 -*-
'''
EndlessBuffer class
===================

A `Buffer` without size limit.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from typing import Dict, List, Optional, cast

from reil.utils.buffers import Buffer, PickModes, T

class EndlessBuffer(Buffer[T]):
    '''
    A `Buffer` without size limit.

    Extends `Buffer` class.

    ### Methods
    setup: sets up the buffer by defining its queue names and pick mode.

    add: adds a new item to the buffer. Overrides `Buffer.add`.

    pick: picks the `count` number of items from the buffer.

    reset: resets the buffer. Overrides `Buffer.reset`.
    '''

    def __init__(self,
                 buffer_names: Optional[List[str]] = None,
                 pick_mode: Optional[PickModes] = None) -> None:
        '''
        Initializes the buffer.

        ### Arguments
        buffer_names: a list containing the names of buffer queues.

        pick_mode: the default mode to pick items from the list.
        '''
        self.setup(buffer_names, pick_mode)

    def setup(self,
              buffer_names: Optional[List[str]] = None,
              pick_mode: Optional[PickModes] = None) -> None:
        '''
        Sets up the buffer.

        ### Arguments
        buffer_names: a list containing the names of buffer elements.

        pick_mode: the default mode to pick items from the list.

        Note: `setup` should be used only for attributes of the buffer that are
        not defined. Attempt to use `setup` to modify size, names or mode will
        result in an exception.
        '''
        super().setup(buffer_names=buffer_names, pick_mode=pick_mode)

    def add(self, data: Dict[str, T]) -> None:
        '''
        Adds a new item to the buffer.

        ### Arguments
        data: a dictionary with the name of buffer queues as keys.
        '''
        self._buffer_index += 1
        for key, v in data.items():
            self._buffer[key].append(v)
        self._count += 1

    def reset(self) -> None:
        '''
        Resets the buffer.
        '''
        super().reset()
        if self._buffer_names is not None:
            self._buffer = cast(Dict[str, List[T]],
                                dict((name, [])
                                     for name in self._buffer_names))
        else:
            self._buffer = None
