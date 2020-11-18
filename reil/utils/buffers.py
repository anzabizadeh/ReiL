from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np


class Buffer:
    _buffer_size = None
    _buffer_names = None
    _pick_mode = None
    _buffer_index = -1
    _count = 0

    def __init__(self,
                 buffer_size: Optional[int] = None,
                 buffer_names: Optional[List[str]] = None,
                 pick_mode: Optional[str] = None) -> None:
        self.setup(buffer_size, buffer_names, pick_mode)

    def setup(self,
              buffer_size: Optional[int] = None,
              buffer_names: Optional[List[str]] = None,
              pick_mode: Optional[str] = None) -> None:
        # self._assign('buffer_size', buffer_size)
        # self._assign('buffer_names', buffer_names)
        if buffer_size is not None:
            if self._buffer_size is not None: 
                raise ValueError('Cannot modify buffer_size. The value is already set.')
            else:
                self._buffer_size = buffer_size

        if buffer_names is not None:
            if self._buffer_names is not None:
                raise ValueError(
                    'Cannot modify buffer_names. The value is already set.')
            else:
                self._buffer_names = buffer_names

        if self._buffer_size is not None and self._buffer_names is not None:
            self._buffer = dict((name, [0.0]*self._buffer_size)
                                for name in self._buffer_names)
        else:
            self._buffer = None

        if pick_mode is not None:
            self._pick_mode = pick_mode

        self.reset()

    # def _assign(self, attribute_name: str, value: Any) -> None:
    #     attr_name = '_' + attribute_name
    #     if attr_name not in self.__dict__:
    #         raise AttributeError(f'Attempt to assign value to {attribute_name} failed.'
    #                              ' Attribute does not exist.')

    #     if value is not None:
    #         if self.__dict__[attr_name] is not None:
    #             raise ValueError(
    #                 f'Cannot modify {attribute_name}. The value is already set.')
    #         else:
    #             self.__dict__[attr_name] = value

    def add(self, data: Dict[str, Any]) -> None:
        self._buffer_index += 1
        for key, v in data.items():
            self._buffer[key][self._buffer_index] = v
        self._count += 1

    def pick(self, count: Optional[int] = None, mode: Optional[str] = None) -> Dict[str, List[Any]]:
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
            raise ValueError('mode should be one of all, old, recent, or random.')

    def _pick_old(self, count: int) -> Dict[str, List[Any]]:
        return dict((name, buffer[:count])
                    for name, buffer in self._buffer.items())

    def _pick_recent(self, count: int) -> Dict[str, List[Any]]:
        return dict((name, buffer[self._buffer_index+1-count:self._buffer_index+1])
                    for name, buffer in self._buffer.items())

    def _pick_random(self, count: int) -> Dict[str, List[Any]]:
        index = np.random.choice(
            self._count, count, replace=False)
        return dict((name, [buffer[i] for i in index])
                    for name, buffer in self._buffer.items())

    def _pick_all(self) -> Dict[str, List[Any]]:
        return dict((name, buffer[:self._buffer_index+1])
                    for name, buffer in self._buffer.items())

    def reset(self) -> None:
        self._buffer_index = -1
        self._count = 0


class CircularBuffer(Buffer):
    _buffer_full = False

    def add(self, data: Dict[str, Any]) -> None:
        try:
            super().add(data)
        except IndexError:
            self._buffer_full = True
            self._buffer_index = -1
            super().add(data)

        self._count -= self._buffer_full  # the size does not change if buffer is full.

    def _pick_old(self, count: int) -> Dict[str, List[Any]]:
        if self._buffer_full:
            raise NotImplementedError
            # return dict(
            #     (name, list(buffer[self._buffer_index+1:self._buffer_index+count+1])
            #           + list(buffer[:max(0, count - (self._buffer_size - self._buffer_index))]))
            #     for name, buffer in self._buffer.items())
        else:
            return super()._pick_old(count)

    def _pick_recent(self, count: int) -> Dict[str, List[Any]]:
        if self._buffer_full:
            raise NotImplementedError
            # return dict((name, list(buffer[-self._buffer_index-count:-self._buffer_index]))
            #             for name, buffer in self._buffer.items())
        else:
            return super()._pick_recent(count)

    def _pick_all(self) -> Dict[str, List[Any]]:
        if self._buffer_full:
            return dict((name,
                         buffer[self._buffer_index+1:] +
                         buffer[:self._buffer_index+1])
                        for name, buffer in self._buffer.items())
        else:
            return super()._pick_all()

    def reset(self) -> None:
        super().reset()
        self._buffer_full = False


class EndlessBuffer(Buffer):
    def __init__(self,
                 buffer_names: Optional[List[str]] = None,
                 pick_mode: Optional[str] = None) -> None:
        self.setup(buffer_names, pick_mode)

    def setup(self,
              buffer_names: Optional[List[str]] = None,
              pick_mode: Optional[str] = None) -> None:

        if self._buffer_names is not None:
            raise ValueError(
                'Cannot modify buffer_names. The value is already set.')
        else:
            self._buffer_names = buffer_names

        if pick_mode is not None:
            self._pick_mode = pick_mode

        self.reset()

    def add(self, data: Dict[str, Any]) -> None:
        self._buffer_index += 1
        for key, v in data.items():
            self._buffer[key].append(v)
        self._count += 1

    def reset(self) -> None:
        super().reset()
        if self._buffer_names is not None:
            self._buffer = dict((name, [])
                                for name in self._buffer_names)
        else:
            self._buffer = None


class VanillaExperienceReplay(CircularBuffer):
    _batch_size = None
    _clear_buffer = False

    def __init__(self,
                 buffer_size: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 buffer_names: Optional[List[str]] = None,
                 clear_buffer: bool = False) -> None:

        self.setup(buffer_size=buffer_size,
                   batch_size=batch_size,
                   buffer_names=buffer_names,
                   clear_buffer=clear_buffer)

        super().setup(pick_mode='random')

    def setup(self,
              buffer_size: Optional[int] = None,
              batch_size: Optional[int] = None,
              buffer_names: Optional[List[str]] = None,
              clear_buffer: bool = False) -> None:

        if buffer_size is not None and  buffer_size < 1:
            raise ValueError('buffer_size should be at least 1.')

        super().setup(buffer_size=buffer_size, buffer_names=buffer_names)

        if self._buffer_size is not None:
            if batch_size is not None and self._buffer_size < batch_size:
                raise ValueError('buffer_size should be >= batch_size.')

        if batch_size is not None:
            if self._batch_size is not None:
                raise ValueError(
                    'Cannot modify batch_size. The value is already set.')
            else:
                self._batch_size = batch_size

        self._clear_buffer = clear_buffer

        self.reset()

    def pick(self) -> Dict[str, List[Any]]:
        if self._buffer_full:
            return super().pick(self._batch_size, 'random')
        else:
            return dict((name, []) for name in self._buffer)

    def reset(self) -> None:
        if self._clear_buffer:
            super().reset()
            self._buffer_full = False
