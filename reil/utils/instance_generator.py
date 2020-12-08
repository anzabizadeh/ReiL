# -*- coding: utf-8 -*-
'''
InstanceGenerator class
=======================

`InstanceGenerator` takes any object derived form `ReilBase` and returns an iterator. 


'''

from __future__ import annotations

import pathlib
from typing import Any, Callable, Generic, Tuple, TypeVar, Union

from reil import reilbase

T = TypeVar('T', bound=reilbase.ReilBase)


class InstanceGenerator(Generic[T], reilbase.ReilBase):  # pylint: disable=unsubscriptable-object
    '''
    Makes any ReilBase object an iterable.

    The initializer accepts, among other arguments, an instance of an object to
    iterate, and `instance_counter_stops`, which is a tuple of the instance
    numbers where the instance generator should stop.

    Attributes
-----------
    is_finite: `True` if `auto_rewind` is `False` and `instance_counter_stops`
    does not contain -1.

    Methods
-----------
    rewind: Rewinds the iterator object.

    is_terminated: returns `True` if no new instances can be generated.
    '''

    def __init__(self,
                 object: T,
                 instance_counter_stops: Tuple[int] = (-1,),  # -1: infinite
                 first_instance_number: int = 0,
                 auto_rewind: bool = False,
                 save_instances: bool = False,
                 overwrite_instances: bool = False,
                 use_existing_instances: bool = True,
                 save_path: Union[pathlib.Path, str] = '',
                 filename_pattern: str = '{n:04}',
                 **kwargs: Any):
        '''
        Initialize the InstanceGenerator

        Attributes
-----------
        object: an instance of an object.

        instance_counter_stops: a tuple of the instance numbers where the instance
            generator should stop. A value of -1 means infinite (Default = (-1,)).

        first_instance_number: the number of the first instance to be generated.
            (Default = 0)

        auto_rewind: whether to rewind after the generator hits the last stop (Default = False).

        save_instances: whether to save instances of the `object` or not (Default = False).

        overwrite_instances: whether to overwrite instances of the `object` or not.
            This flag is useful only if `save_instances` is set to True (Default = False).

        use_existing_instances: whether try to load instances before attempting
            to create them (Default = True).

        save_path: the path where instances should be saved to/ loaded from
            (Default = '').

        filename_pattern: a string that uses "n" as the instance number, and is
            used for saving and loading instances (Default = '{n:04}').
        '''
        super().__init__(**kwargs)

        self._object = object
        self._save_instances = save_instances
        self._use_existing_instances = use_existing_instances
        self._overwrite_instances = overwrite_instances
        self._save_path = save_path
        self._filename_pattern = filename_pattern

        self._auto_rewind = auto_rewind
        self._first_instance_number = first_instance_number
        self._instance_counter_stops = instance_counter_stops
        self._last_stop_index = len(self._instance_counter_stops) - 1

        self.is_finite = -1 not in instance_counter_stops and not auto_rewind

        self.rewind()

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[int, T]:
        try:
            end = self._instance_counter_stops[self._stops_index]
        except IndexError:
            if self._auto_rewind:
                self.rewind()
                end = self._instance_counter_stops[self._stops_index]
            else:
                raise StopIteration

        self._instance_counter += 1

        if self._stop_check(self._instance_counter, end):
            self._instance_counter -= 1
            self._stops_index += 1

            if self._stops_index <= self._last_stop_index:
                if self._instance_counter_stops[self._stops_index] == -1:
                    self._stop_check: Callable[[int, int],
                                               bool] = lambda current, end: False
                else:
                    self._stop_check: Callable[[int, int],
                                               bool] = lambda current, end: current > end
            raise StopIteration
        else:
            current_instance = self._filename_pattern.format(
                n=self._instance_counter)
            new_instance = True
            if self._use_existing_instances:
                try:
                    self._object.load(path=self._save_path,
                                      filename=current_instance)
                    new_instance = False
                except FileNotFoundError:
                    self._object.reset()
            else:
                self._object.reset()

            if self._save_instances and new_instance:
                if self._overwrite_instances:
                    self._object.save(path=self._save_path,
                                      filename=current_instance)
                else:
                    raise FileExistsError(
                        f'File {current_instance} already exists.')

        return (self._instance_counter, self._object)

    def rewind(self) -> None:
        '''
        Rewinds the iterator object.
        '''
        self._instance_counter = self._first_instance_number - 1
        self._stops_index = 0
        if self._instance_counter_stops[self._stops_index] == -1:
            self._stop_check: Callable[[int, int],
                                       bool] = lambda current, end: False
        else:
            self._stop_check: Callable[[int, int],
                                       bool] = lambda current, end: current > end

    def is_terminated(self) -> bool:
        return self._stops_index > self._last_stop_index

    def __repr__(self) -> str:
        try:
            return f'{self.__class__.__qualname__} -> {self._object.__repr__()}'
        except AttributeError:
            return self.__class__.__qualname__
