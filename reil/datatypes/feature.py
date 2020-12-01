from __future__ import annotations

import dataclasses
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar, Union

FeatureType = TypeVar('FeatureType')


@dataclasses.dataclass
class Feature(Generic[FeatureType]):
    is_numerical: Optional[bool] = True
    value: Optional[Any] = None
    randomized: Optional[bool] = True
    generator: Optional[Callable[[FeatureType], Any]] = lambda x: x.value

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

    def _numerical_validator(self):
        if self.value is not None:
            if self.lower is not None and self.value < self.lower:
                raise ValueError(
                    f'value={self.value} is less than lower={self.lower}.')

            if self.upper is not None and self.value > self.upper:
                raise ValueError(
                    f'value={self.value} is greater than upper={self.upper}.')

    def _categorical_validator(self):
        if self.value is not None:
            if self.categories is not None and self.value not in self.categories:
                raise ValueError(
                    f'value={self.value} is in the categories={self.categories}.')

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

    def generate(self):
        if self.generator is None:
            return

        if self.randomized or self.value is None:
            self.value = self.generator(self)
