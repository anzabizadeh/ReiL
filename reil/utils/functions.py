# -*- coding: utf-8 -*-
'''
functions module
================

Contains some useful functions.
'''

import math
import random
from reil.datatypes.feature import FeatureArray
from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar

import numpy as np
from reil.datatypes import FeatureGenerator
from scipy import stats


def random_choice(f: Any):
    '''
    This function allows `yaml` config files to use `random.choice`
    as part of `reil` module.
    '''
    return random.choice(f)


def random_truncated_normal(f: FeatureGenerator) -> float:
    return min(max(
        np.random.normal(f.mean, f.stdev), f.lower),
        f.upper)


def random_uniform(f: FeatureGenerator):
    return np.random.uniform(f.lower, f.upper)


def random_categorical(f: FeatureGenerator):
    if f.probabilities is None:
        return random.choice(f.categories)  # type:ignore
    else:
        return np.random.choice(
            f.categories, 1, p=f.probabilities)[0]


def random_truncated_lnorm(f: FeatureGenerator) -> float:
    # capture 50% of the data.
    # This restricts the log values to a "reasonable" range
    if f.randomized:
        quartileRange = (0.25, 0.75)
        lnorm = stats.lognorm(f.stdev, scale=math.exp(f.mean))  # type:ignore
        qValues = lnorm.ppf(quartileRange)
        values = list(v for v in lnorm.rvs(size=1000)
                      if (v > qValues[0]) & (v < qValues[1]))

        return random.sample(values, 1)[0]

    return math.exp(f.mean + (f.stdev ** 2)/2)


def square_dist(x: float, y: Iterable[float]) -> float:
    return sum((x - yi) ** 2
               for yi in y)


def in_range(r: Tuple[float, float], x: Iterable[float]) -> int:
    return sum(r[0] <= xi <= r[1]
               for xi in x)


def interpolate(start: float, end: float, steps: int) -> Iterable[float]:
    return (start + (end - start) / steps * j
            for j in range(1, steps + 1))


T = TypeVar('T')


def generate_modifier(
        operation: Callable[[T], T],
        condition: Optional[Callable[[FeatureArray], bool]] = None
) -> Callable[[FeatureArray, T], T]:
    '''Generate a modifier function for states or actions

    Parameters
    ----------
    operation:
        What should happen to the input.

    condition:
        A function that accepts a state `FeatureArray`, and based on that
        determines if the `operation` should be applied to the input.

    Returns
    -------
    :
        A function that accepts `condition_state` and `input` and returns the
        modified `input`.
    '''
    if condition is None:
        def no_condition_modifier(condition_state: FeatureArray,
                                  input: T) -> T:
            return operation(input)

        return no_condition_modifier

    def modifier(condition_state: FeatureArray,
                 input: T) -> T:
        if condition(condition_state):
            return operation(input)

        return input

    return modifier
