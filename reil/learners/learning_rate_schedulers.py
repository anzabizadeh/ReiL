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
        Initialize the scheduler.

        ### Arguments
        initial_lr: initial learning rate.
        new_rate_function: a function that accepts epoch and current learning
        rate and returns a new learning rate.
        '''
        self.initial_lr = initial_lr
        self._lambda_func = new_rate_function

    def new_rate(self, epoch: int, current_lr: float) -> float:
        '''
        Returns a new rate based on epoch and current learning rate.
        '''
        return self._lambda_func(epoch, current_lr)


class ConstantLearningRate(LearningRateScheduler):
    '''
    A LearningRateScheduler with constant rate.
    '''
    def __init__(self, initial_lr: float) -> None:
        '''
        Initialize the scheduler.

        ### Arguments
        initial_lr: initial learning rate.
        '''
        super().__init__(initial_lr, lambda e, lr: initial_lr)
