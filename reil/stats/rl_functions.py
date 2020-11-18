from collections import namedtuple
import dataclasses
import functools
from typing import Any, Generator, Iterable, List, Optional, Tuple

from reil import rldata

# SOME THOUGHTS!
# Lookahead is not something a subject can/should do!
# The environment should take a copy of the subject, apply some policy (random,
# current policy, fixed, etc.) for a number of steps (RLFunction.length) for
# a number of times (sample size), then provide the outcome to the RLFunction
# and finally add that to the reward from the subject!


Arguments = namedtuple('Arguments', ('y', 'x'), defaults=(None,))


def square_dist(x: float, y: Iterable) -> float:
    return sum((x - yi) ** 2 for yi in y)


def in_range(r: Tuple[float, float], x: Iterable) -> int:
    return sum(r[0] <= xi <= r[1] for xi in x)


def interpolate(start: float, end: float, steps: int) -> Generator[float, None, None]:
    return (start + (end - start) / steps * j for j in range(1, steps + 1))


@dataclasses.dataclass
class RLFunction:
    name: str
    arguments: Arguments
    length: int = -1
    multiplier: float = 1.0
    retrospective: bool = True
    interpolate: bool = True

    def __post_init__(self):
        if not isinstance(self.arguments, Arguments):
            self.arguments = Arguments(*self.arguments)  # type: ignore

    def __call__(self, args: rldata.RLData) -> float:
        temp = args.value
        y = temp[self.arguments.y]
        x = temp[self.arguments.x] if self.arguments.x is not None else None

        fn = [functools.partial(self._no_retro_no_inter, y),
              functools.partial(self._no_retro_inter, y, x),
              functools.partial(self._retro_no_inter, y),
              functools.partial(self._retro_inter, y, x)
              ][self.retrospective*2 + self.interpolate]

        try:
            result = self.multiplier * fn()
        except NotImplementedError:
            result = self.multiplier * self._default_function(y, x)

        return result

    def _retro_inter(self, y: List[Any], x: List[Any]) -> float:
        raise NotImplementedError

    def _retro_no_inter(self, y: List[Any]) -> float:
        raise NotImplementedError

    def _no_retro_inter(self, y: List[Any], x: List[Any]) -> float:
        raise NotImplementedError

    def _no_retro_no_inter(self, y: List[Any]) -> float:
        raise NotImplementedError

    def _default_function(self, y: List[Any], x: Optional[List[Any]] = None) -> float:
        raise NotImplementedError


@dataclasses.dataclass
class NormalizedSquareDistance(RLFunction):
    center: float = 0.0
    band_width: float = 1.0
    exclude_first: bool = False

    def _default_function(self, y: List[Any], x: Optional[List[Any]] = None) -> float:
        _x = [1] * (len(y) - 1) if x is None else x

        if len(y) != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0] + y
        else:
            _y = y

        result = sum(square_dist(self.center, interpolate(_y[i], _y[i+1], _x[i]))
                     for i in range(len(_x)))

        # normalize
        result *= (2.0 / self.band_width) ** 2

        return result


@dataclasses.dataclass
class PercentInRange(RLFunction):
    acceptable_range: Tuple[float, float] = (0.0, 1.0)
    exclude_first: bool = False

    def _default_function(self, y: List[Any], x: Optional[List[Any]] = None) -> float:
        _x = [1] * (len(y) - 1) if x is None else x
        if len(y) != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0] + y
        else:
            _y = y

        result = sum(
            in_range(self.acceptable_range, interpolate(_y[i], _y[i+1], _x[i]))
            for i in range(len(_x)))

        total_intervals = sum(_x)

        return result / total_intervals

# TODO: not implemented yet!
@dataclasses.dataclass
class Delta(RLFunction):
    '''
    Get changes in the series.
    
    available `op`s:
        count: counts the number of change points in y.
        sum: sum of value changes
        average: average value change

    available `interpolation_method`s:
        linear
        post: y = y[i] at x[i]
        pre: y = y[i] at x[i-1]
    '''
    exclude_first: bool = False
    op: str = 'count'
    interpolation_method: str = 'linear'

    # def _default_function(self, y: List[Any], x: Optional[List[Any]] = None) -> float:
    #     if self.op == 'count':
    #         result = sum(yi != y[i+1]
    #                     for i, yi in enumerate(y[:-1]))

    #     return result


class Functions:
    @staticmethod
    def TTR(INRs: List[float],
            intervals: Optional[List[int]] = None,
            exclude_first: bool = False,
            INR_range: Tuple[float, float] = (2.0, 3.0)) -> float:
        if intervals is None:
            temp = INRs[1:] if exclude_first else INRs
            result = sum((1 if INR_range[0] <= INRi <= INR_range[1] else 0
                          for INRi in temp)
                         ) / len(temp)
        else:
            if len(INRs) != len(intervals) + 1:
                raise ValueError(
                    'INRs should have exactly one item more than intervals.')

            result = 0.0
            for i, current_interval in enumerate(intervals):
                result += sum(1 if INR_range[0] <= (INRs[i] + (INRs[i+1] - INRs[i])/current_interval*j) <= INR_range[1] else 0
                              for j in range(1, current_interval + 1))

            total_intervals = sum(intervals)
            if not exclude_first:
                result += Functions.TTR([INRs[0]])
                total_intervals += 1

            result /= total_intervals

        return result

    @staticmethod
    def dose_change_count(dose_list: List[float], intervals: Optional[List[int]] = None) -> int:
        # assuming dose is fixed during each interval
        return sum(x != dose_list[i+1]
                   for i, x in enumerate(dose_list[:-1]))

    @staticmethod
    def delta_dose(dose_list: List[float], intervals: Optional[List[int]] = None) -> float:
        # assuming dose is fixed during each interval
        return sum(abs(x-dose_list[i+1])
                   for i, x in enumerate(dose_list[:-1]))

    @staticmethod
    def total_dose(dose_list: List[float],
                   intervals: Optional[List[int]] = None) -> float:
        if intervals is None:
            result = sum(dose_list)
        else:
            if len(dose_list) != len(intervals):
                raise ValueError(
                    'dose_list and intervals should have the same number of items.')

            result = sum(dose*interval
                         for dose, interval in zip(dose_list, intervals))

        return result

    @staticmethod
    def average_dose(dose_list: List[float],
                     intervals: Optional[List[int]] = None) -> float:
        total_dose = Functions.total_dose(dose_list, intervals)
        total_interval = len(
            dose_list) if intervals is None else sum(intervals)

        return total_dose / total_interval

    @staticmethod
    def normalized_square_dist(INRs: List[float],
                               intervals: Optional[List[int]] = None,
                               exclude_first: bool = False,
                               INR_range: Tuple[float, float] = (2.0, 3.0)) -> float:

        def square_dist(x: float, y: Generator[float, None, None]) -> float:
            return sum((x - yi) ** 2.0 for yi in y)

        def interpolate(start: float, end: float, steps: int) -> Generator[float, None, None]:
            return (start + (end - start) / steps * j for j in range(1, steps + 1))

        INR_mid = sum(INR_range) / 2.0

        _intervals = [1] * (len(INRs) - 1) if intervals is None else intervals

        if len(INRs) != len(_intervals) + 1:
            raise ValueError(
                'INRs should have exactly one item more than intervals.')

        if not exclude_first:
            _intervals = [1] + _intervals
            _INRs = [0.0] + INRs
        else:
            _INRs = INRs

        result = sum(square_dist(INR_mid, interpolate(_INRs[i], _INRs[i+1], _intervals[i]))
                     for i in range(len(_intervals)))

        # normalize
        result *= (2.0 / (INR_range[1] - INR_range[0])) ** 2

        return result
