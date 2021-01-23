# -*- coding: utf-8 -*-
'''
Learner class
=============

The base class for all `learner` classes.
'''
from typing import Any, Generic, Tuple, TypeVar

from reil import learners, reilbase
from reil.datatypes import ReilData

LabelType = TypeVar('LabelType')


class Learner(reilbase.ReilBase, Generic[LabelType]):
    '''
    The base class for all `learner` classes.
    '''
    def __init__(self,
                 learning_rate: learners.LearningRateScheduler,
                 **kwargs: Any) -> None:
        '''
        Arguments
        ---------
        learning_rate:
            A `LearningRateScheduler` object that determines the learning rate
            based on epoch. If any scheduler other than constant is provided,
            the model uses the `new_rate` method of the scheduler to determine
            the learning rate at each epoch.
        '''
        super().__init__(**kwargs)
        self._learning_rate = learning_rate

    @classmethod
    def _empty_instance(cls):
        return cls(learning_rate=learners.ConstantLearningRate(0.0))

    def predict(self, X: Tuple[ReilData, ...]) -> Tuple[LabelType, ...]:
        '''
        predict `y` for a given input list `X`.

        Arguments
        ---------
        X:
            A list of `ReilData` as inputs to the prediction model.

        Returns
        -------
        :
            The predicted `y`.
        '''
        raise NotImplementedError

    def learn(self, X: Tuple[ReilData, ...], Y: Tuple[LabelType, ...]) -> None:
        '''
        Learn using the training set `X` and `Y`.

        Arguments
        ---------
        X:
            A list of `ReilData` as inputs to the learning model.

        Y:
            A list of float labels for the learning model.
        '''
        raise NotImplementedError
