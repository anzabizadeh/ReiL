from typing import Generator, List, Optional, Tuple


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
