# -*- coding: utf-8 -*-
'''
functions module
================

Contains some useful functions.
'''

import math
import random
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
from reil.datatypes.feature import FeatureArray, FeatureGenerator
from scipy.stats import lognorm


def random_choice(f: Any):
    '''
    This function allows `yaml` config files to use `random.choice`
    as part of `reil` module.
    '''
    return random.choice(f)


def random_uniform(f: FeatureGenerator[float]) -> float:
    if f.randomized:
        return np.random.uniform(f.lower, f.upper)

    if f.mean is not None:
        return f.mean

    if f.upper is None or f.lower is None:
        raise ValueError('mean, or upper and lower should be numbers.')

    return (f.upper - f.lower) / 2.0


def random_normal(f: FeatureGenerator[float]) -> float:
    if f.randomized:
        return np.random.normal(f.mean, f.stdev)

    if f.mean is None:
        raise ValueError('mean should be a number.')

    return f.mean


def random_normal_truncated(f: FeatureGenerator[float]) -> float:
    if f.randomized:
        return min(max(
            np.random.normal(f.mean, f.stdev), f.lower),
            f.upper)

    if f.mean is None:
        raise ValueError('mean should be a number.')

    return f.mean


def random_lognormal(f: FeatureGenerator[float]) -> float:
    try:
        exp_mu = math.exp(f.mean)  # type: ignore
    except TypeError:
        raise ValueError('mean should be a number.')

    if f.randomized:
        return lognorm.rvs(s=f.stdev, scale=exp_mu)

    return exp_mu


def random_lognormal_truncated(f: FeatureGenerator[float]) -> float:
    # capture 50% of the data.
    # This restricts the log values to a "reasonable" range
    try:
        exp_mu = math.exp(f.mean)  # type: ignore
    except TypeError:
        raise ValueError('mean should be a number.')

    if f.randomized:
        quartileRange = (0.25, 0.75)
        lnorm = lognorm(f.stdev, scale=exp_mu)
        qValues: Tuple[float, float] = lnorm.ppf(quartileRange)
        values: List[float] = list(
            v for v in lnorm.rvs(size=1000)
            if (v > qValues[0]) & (v < qValues[1]))

        return random.sample(values, 1)[0]

    return exp_mu


Categorical = TypeVar('Categorical')


def random_categorical(f: FeatureGenerator[Categorical]) -> Categorical:
    if (categories := f.categories) is None:
        raise TypeError('No categories found!')

    if f.randomized:
        if (probs := f.probabilities) is None:
            return random.choice(categories)
        else:
            return np.random.choice(categories, 1, p=probs)[0]

    return categories[0]


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
        def no_condition_modifier(
                condition_state: FeatureArray, input: T) -> T:
            return operation(input)

        return no_condition_modifier
    else:
        def modifier(
                condition_state: FeatureArray, input: T) -> T:
            if condition(condition_state):  # type: ignore
                return operation(input)

            return input

    return modifier
