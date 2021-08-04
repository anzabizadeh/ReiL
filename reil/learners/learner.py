# -*- coding: utf-8 -*-
'''
Learner class
=============

The base class for all `learner` classes.
'''
from abc import abstractmethod
from typing import Any, Protocol, Tuple, TypeVar

from reil import reilbase
from reil.datatypes.feature import FeatureArray
from reil.learners.learning_rate_schedulers import (ConstantLearningRate,
                                                    LearningRateScheduler)

LabelType = TypeVar('LabelType')


class LearnerProtocol(Protocol[LabelType]):
    '''
    The base class for all `learner` classes.
    '''
    _learning_rate: LearningRateScheduler

    @abstractmethod
    def predict(self, X: Tuple[FeatureArray, ...]) -> Tuple[LabelType, ...]:
        '''
        predict `y` for a given input list `X`.

        Arguments
        ---------
        X:
            A list of `FeatureArray` as inputs to the prediction model.

        Returns
        -------
        :
            The predicted `y`.
        '''
        raise NotImplementedError

    @abstractmethod
    def learn(
            self, X: Tuple[FeatureArray, ...], Y: Tuple[LabelType, ...]
    ) -> None:
        '''
        Learn using the training set `X` and `Y`.

        Arguments
        ---------
        X:
            A list of `FeatureArray` as inputs to the learning model.

        Y:
            A list of float labels for the learning model.
        '''
        raise NotImplementedError


class Learner(reilbase.ReilBase, LearnerProtocol[LabelType]):
    '''
    The base class for all `learner` classes.
    '''
    def __init__(
            self, learning_rate: LearningRateScheduler, **kwargs: Any) -> None:
        '''
        Arguments
        ---------
        learning_rate:
            A `LearningRateScheduler` object that determines the learning rate
            based on iteration. If any scheduler other than constant is
            provided, the model uses the `new_rate` method of the scheduler to
            determine the learning rate at each iteration.
        '''
        super().__init__(**kwargs)
        self._learning_rate = learning_rate

    @classmethod
    def _empty_instance(cls):
        return cls(learning_rate=ConstantLearningRate(0.0))
