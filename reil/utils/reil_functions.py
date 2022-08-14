from __future__ import annotations

import dataclasses
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from reil.logger import Logger
from reil.datatypes.feature import FeatureSet
from reil.utils.functions import dist, in_range, interpolate, square_dist

# NOTE:
# `retrospective` was meant to be used to indicate regular reward computation
# vs lookahead. However, lookahead should be done by an `Environment`. So,
# I removed `retrospective` to simplify the implementation and improve
# performance.

reil_func_logger = Logger('reil_functions')

TypeY = TypeVar('TypeY')
TypeX = TypeVar('TypeX')


@dataclasses.dataclass
class ReilFunction(Generic[TypeY, TypeX]):
    name: str
    y_var_name: str
    x_var_name: Optional[str] = None
    length: int = -1
    multiplier: float = 1.0
    interpolate: bool = True

    def __post_init__(self):
        self._fn = self._inter if self.interpolate else self._no_inter

    def __call__(self, args: FeatureSet) -> float:
        temp = args.value
        fn_args: Dict[str, Any] = {'y': temp[self.y_var_name]}
        if self.x_var_name:
            fn_args['x'] = temp[self.x_var_name]

        try:
            result = self.multiplier * self._fn(**fn_args)
        except NotImplementedError:
            result = self.multiplier * self._default_function(**fn_args)

        return result

    # just for the compatibility with old saved models. Should not be used.
    def _retro_inter(self, y: List[TypeY], x: List[TypeX]) -> float:
        raise NotImplementedError

    # just for the compatibility with old saved models. Should not be used.
    def _retro_no_inter(self, y: List[TypeY], x: List[TypeX]) -> float:
        raise NotImplementedError

    def _inter(self, y: List[TypeY], x: List[TypeX]) -> float:
        raise NotImplementedError

    def _no_inter(self, y: List[TypeY]) -> float:
        raise NotImplementedError

    def _default_function(
            self, y: List[TypeY], x: Optional[List[TypeX]] = None) -> float:
        raise NotImplementedError


@dataclasses.dataclass
class NormalizedSquareDistance(ReilFunction[float, int]):
    center: float = 0.0
    band_width: float = 1.0
    amplifying_factor: float = 1.0
    exclude_first: bool = False

    def _default_function(
            self, y: List[float], x: Optional[List[int]] = None) -> float:
        len_y = len(y)
        _x = x or [1] * (len_y - 1)

        if len_y != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0] + y
        else:
            _y = y

        result = sum(
            (self.amplifying_factor ** i) * square_dist(
                self.center, interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x))

        # normalize
        result *= (2.0 / self.band_width) ** 2

        return result


@dataclasses.dataclass
class NormalizedDistance(ReilFunction[float, int]):
    center: float = 0.0
    band_width: float = 1.0
    exclude_first: bool = False

    def _default_function(
            self, y: List[float], x: Optional[List[int]] = None) -> float:
        len_y = len(y)
        _x = x or [1] * (len_y - 1)

        if len_y != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0] + y
        else:
            _y = y

        result = sum(dist(
            self.center, interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x))

        # normalize
        result *= (2.0 / self.band_width) ** 2

        return result


@dataclasses.dataclass
class PercentInRange(ReilFunction[float, int]):
    acceptable_range: Tuple[float, float] = (0.0, 1.0)
    exclude_first: bool = False

    def _default_function(
            self, y: List[float], x: Optional[List[int]] = None) -> float:
        len_y = len(y)
        _x = x or [1] * (len_y - 1)
        if len_y != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0] + y
        else:
            _y = y

        result = sum(
            in_range(
                self.acceptable_range,
                interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x))

        total_intervals = sum(_x)

        return result / total_intervals


@dataclasses.dataclass
class NotEqual(ReilFunction[float, int]):
    interpolate: bool = False

    def _no_inter(
            self, y: List[float], x: Optional[List[int]] = None) -> float:
        if x:
            reil_func_logger.info(
                'x is provided, but is not used in `NotEqual` function.')

        try:
            result = sum(
                y1 != y2
                for y1, y2 in zip(y[:-1], y[1:])
            ) / (len(y) - 1)
        except ZeroDivisionError:  # NotEqual for one observation is 0.
            result = 0

        return result


# TODO: not implemented yet!
# @dataclasses.dataclass
# class Delta(ReilFunction):
#     '''
#     Get changes in the series.

#     available `op`s:
#         count: counts the number of change points in y.
#         sum: sum of value changes
#         average: average value change

#     available `interpolation_method`s:
#         linear
#         post: y = y[i] at x[i]
#         pre: y = y[i] at x[i-1]
#     '''
#     exclude_first: bool = False
#     op: str = 'count'
#     interpolation_method: str = 'linear'

# def _default_function(
#         self, y: List[Any], x: Optional[List[Any]] = None) -> float:
#     if self.op == 'count':
#         result = sum(yi != y[i+1]
#                     for i, yi in enumerate(y[:-1]))

#     return result


# class Functions:
#     @staticmethod
#     def dose_change_count(dose_list: List[float],
#                           intervals: Optional[List[int]] = None) -> int:
#         # assuming dose is fixed during each interval
#         return sum(x != dose_list[i+1]
#                    for i, x in enumerate(dose_list[:-1]))

#     @staticmethod
#     def delta_dose(dose_list: List[float],
#                    intervals: Optional[List[int]] = None) -> float:
#         # assuming dose is fixed during each interval
#         return sum(abs(x-dose_list[i+1])
#                    for i, x in enumerate(dose_list[:-1]))

#     @staticmethod
#     def total_dose(dose_list: List[float],
#                    intervals: Optional[List[int]] = None) -> float:
#         if intervals is None:
#             result = sum(dose_list)
#         else:
#             if len(dose_list) != len(intervals):
#                 raise ValueError(
#                     'dose_list and intervals should '
#                     'have the same number of items.')

#             result = sum(dose*interval
#                          for dose, interval in zip(dose_list, intervals))

#         return result

#     @staticmethod
#     def average_dose(dose_list: List[float],
#                      intervals: Optional[List[int]] = None) -> float:
#         total_dose = Functions.total_dose(dose_list, intervals)
#         total_interval = len(
#             dose_list) if intervals is None else sum(intervals)

#         return total_dose / total_interval
