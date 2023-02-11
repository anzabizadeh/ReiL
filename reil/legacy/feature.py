# -*- coding: utf-8 -*-
# type: ignore
'''
Feature class
=============

A datatype that accepts initial value and feature generator, and generates
new values.
'''
from __future__ import annotations

import dataclasses
from typing import Any, Callable

from reil.datatypes import reildata


@dataclasses.dataclass(frozen=True)
class Feature_old(reildata.FeatureGenerator):
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
    # value: Any | None = None

    def __post_init__(self):
        super().__post_init__()
        if not self.is_numerical and self.probabilities is not None:
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

    @classmethod
    def categorical(
            cls,
            name,
            # value: Any | None = None,
            categories: tuple[Any, ...] | None = None,
            probabilities: tuple[float, ...] | None = None,
            randomized: bool | None = None,
            generator: Callable[[Feature_old], Any] | None = None):
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
        instance = cls(
            name=name, is_numerical=False, categories=categories,
            probabilities=probabilities, randomized=randomized,
            generator=generator)

        return instance

    @classmethod
    def numerical(
            cls,
            name,
            lower: int | float | None = None,
            upper: int | float | None = None,
            mean: int | float | None = None,
            stdev: int | float | None = None,
            generator: Callable[[Feature_old], Any] | None = None,
            randomized: bool | None = None):
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
        instance = cls(
            name=name, is_numerical=True, lower=lower, upper=upper,
            mean=mean, stdev=stdev, generator=generator, randomized=randomized)

        return instance

    def generate(self):
        '''
        Generate a new value using the generator.
        '''
        try:
            return self.generator(self)
        except TypeError:
            return self.generator
