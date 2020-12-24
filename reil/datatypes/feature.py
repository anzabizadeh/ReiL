# -*- coding: utf-8 -*-
'''
Feature class
=============

A datatype that accepts initial value and feature generator, and generates
new values.
'''
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar, Union

FeatureType = TypeVar('FeatureType')


@dataclasses.dataclass
class Feature(Generic[FeatureType]):
    '''
    A datatype that accepts initial value and feature generator, and generates
    new values.

    Attributes
    ----------
    is_numerical:
        Is the feature numerical?

    value:
        The currect value of the feature.

    randomized:
        Whether the generator should produce random values.

    generator:
        A function that accepts feature characteristics and generates a new
        value

    lower:
        The lower bound for numerical features.

    upper:
        The upper bound for numerical features.

    mean:
        Mean of the distribution for numerical features.

    stdev:
        Standard Deviation of the distribution for numerical features.

    categories:
        A list of possible values for categorical features.

    probabilities:
        A list of probabilities corresponding to each possible value
        for categorical features.
    '''
    is_numerical: Optional[bool] = True
    value: Optional[Any] = None
    randomized: Optional[bool] = True
    generator: Optional[Callable[["Feature"], Any]] = lambda x: x.value

    def __post_init__(self):
        if self.is_numerical:
            self.lower: Optional[float] = None
            self.upper: Optional[float] = None
            self.mean: Optional[float] = None
            self.stdev: Optional[float] = None
        else:
            self.categories: Optional[Tuple[Any, ...]] = None
            self.probabilities: Optional[Tuple[float, ...]] = None

    @classmethod
    def categorical(cls,
                    value: Optional[Any] = None,
                    categories: Optional[Tuple[Any, ...]] = None,
                    probabilities: Optional[Tuple[float, ...]] = None,
                    randomized: Optional[bool] = None,
                    generator: Optional[Callable[[Feature], Any]] = None):
        '''
        Create a categorical Feature.

        Arguments
        ---------
        value:
            The initial value of the feature.

        randomized:
            Whether the generator should produce random values.

        generator:
            A function that gets feature characteristics and generates a new
            value

        categories:
            A list of possible values.

        probabilities:
            A list of probabilities corresponding to each possible value.
        '''
        instance = cls(is_numerical=False,
                       value=value,
                       randomized=randomized,
                       generator=generator)

        instance.categories = categories
        instance.probabilities = probabilities

        instance._categorical_validator()

        return instance

    @classmethod
    def numerical(cls,
                  value: Optional[Union[int, float]] = None,
                  lower: Optional[Union[int, float]] = None,
                  upper: Optional[Union[int, float]] = None,
                  mean: Optional[Union[int, float]] = None,
                  stdev: Optional[Union[int, float]] = None,
                  generator: Optional[Callable[[Feature], Any]] = None,
                  randomized: Optional[bool] = None):
        '''
        Create a numerical Feature.

        Arguments
        ---------
        value:
            The currect value of the feature.

        randomized:
            Whether the generator should produce random values.

        generator:
            A function that gets feature characteristics and generates a new
            value

        lower:
            The lower bound.

        upper:
            The upper bound.

        mean:
            Mean of the distribution.

        stdev:
            Standard Deviation of the distribution.
        '''
        instance = cls(is_numerical=True,
                       value=value,
                       generator=generator,
                       randomized=randomized)

        instance.lower = lower
        instance.upper = upper
        instance.mean = mean
        instance.stdev = stdev

        instance._numerical_validator()

        return instance

    def _numerical_validator(self) -> None:
        '''
        Check if the value is in the defined range.
        '''
        if self.value is not None:
            if self.lower is not None and self.value < self.lower:
                raise ValueError(
                    f'value={self.value} is less than lower={self.lower}.')

            if self.upper is not None and self.value > self.upper:
                raise ValueError(
                    f'value={self.value} is greater than upper={self.upper}.')

    def _categorical_validator(self) -> None:
        '''
        Check if the value is in the defined categories and probabilities add
        up to one.
        '''
        if self.value is not None:
            if (self.categories is not None and
                    self.value not in self.categories):
                raise ValueError(
                    f'value={self.value} is in '
                    f'the categories={self.categories}.')

        if self.probabilities is not None:
            if abs(sum(self.probabilities) - 1.0) > 1e-6:
                raise ValueError('probabilities should add up to 1.0.'
                                 f'Got {sum(self.probabilities)}')
            if self.categories is None:
                raise ValueError(
                    'probabilities cannot be set for None categories.')
            if len(self.probabilities) != len(self.categories):
                raise ValueError('Size mismatch. '
                                 f'{len(self.categories)} categories vs. '
                                 f'{len(self.categories)} probabilities')

    def generate(self) -> None:
        '''
        Generate a new value using the generator.
        '''
        if self.generator is None:
            return

        if self.randomized or self.value is None:
            self.value = self.generator(self)
