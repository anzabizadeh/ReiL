# -*- coding: utf-8 -*-
'''
learning_rate_schedulers module
===============================

A module that provides different learning rate schedulers.

classes
-------
LearningRateScheduler:
    The base class of all schedulers. Accepts a bi-variate function.

ConstantLearningRate:
    A learning rate scheduler with constant rate.
'''
from typing import Callable
from reil import reilbase


class LearningRateScheduler(reilbase.ReilBase):
    '''
    A class that gets an initial learning rate and a function to determine the
    rate based on epoch and previous rate.
    '''
    def __init__(self, initial_lr: float,
                 new_rate_function: Callable[[int, float], float]) -> None:
        '''
        Arguments
        ---------
        initial_lr:
            The initial learning rate.

        new_rate_function:
            A function that accepts epoch and current learning
            rate and returns a new learning rate.
        '''
        self.initial_lr = initial_lr
        self._lambda_func = new_rate_function

    @classmethod
    def _empty_instance(cls):
        return cls(0.0, lambda e, lr: lr)

    def new_rate(self, epoch: int, current_lr: float) -> float:
        '''
        Determine the new rate based on `epoch` and current learning rate.

        Arguments
        ---------
        epoch:
            The current training epoch.

        current_lr:
            The current learning rate.

        Returns
        -------
        :
            The new learning rate.

        '''
        return self._lambda_func(epoch, current_lr)


class ConstantLearningRate(LearningRateScheduler):
    '''
    A `LearningRateScheduler` with constant rate.
    '''
    def __init__(self, initial_lr: float) -> None:
        '''
        Arguments
        ---------
        initial_lr:
            The initial learning rate.
        '''
        super().__init__(initial_lr, lambda e, lr: initial_lr)
