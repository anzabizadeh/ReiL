# -*- coding: utf-8 -*-
'''
Aurora class
============

Aurora Dosing Protocol, based on Ravvaz et al. (2017).
'''

from typing import Any, Dict, Tuple
from reil.healthcare.dosing_protocols import DosingProtocol


class Aurora(DosingProtocol):
    '''
    Aurora Dosing Protocol, based on Ravvaz et al. (2017)
    '''

    def prescribe(self,  # noqa: C901
                  patient: Dict[str, Any],
                  additional_info: Dict[str, Any]
                  ) -> Tuple[float, int, Dict[str, Any]]:
        today = patient['day']
        INRs = patient['INR_history']
        previous_INR = INRs[-1]
        previous_dose = patient['dose_history'][-1]
        previous_interval = patient['interval_history'][-1]

        red_flag = additional_info.get('red_flag', False)
        skip_dose = additional_info.get('skip_dose', 0)
        new_dose = additional_info.get('new_dose', 0)
        number_of_stable_days = additional_info.get('number_of_stable_days', 0)

        if red_flag:
            if previous_INR > 3.0:
                next_dose = 0.0
                next_interval = 2
            else:
                red_flag = False
                next_dose = new_dose
                next_interval = 7
        elif skip_dose:
            skip_dose = 0
            next_dose = new_dose
            next_interval = 7
        elif today <= 2:  # initial dosing
            next_dose = 10.0 if patient['age'] < 65.0 else 5.0
            next_interval = 3 - today
        elif today == 3:  # adjustment dosing
            if previous_INR >= 2.0:
                next_dose = 5.0
                next_interval = 2
                if previous_INR <= 3.0:
                    number_of_stable_days = self._stable_days(
                        INRs[-2], previous_INR, previous_interval)
            else:
                next_dose, next_interval, _, _ = self.aurora_dosing_table(
                    previous_INR, previous_dose)
        elif today == 4:
            raise ValueError(
                'Cannot use Aurora on day 4. '
                'Dose on day 4 equals dose on day 3.')
        else:  # maintenance dosing
            if 2 <= previous_INR <= 3:
                number_of_stable_days += self._stable_days(
                    INRs[-2], previous_INR, previous_interval)

                next_dose = previous_dose
                next_interval = self.aurora_retesting_table(
                    number_of_stable_days)
            else:
                number_of_stable_days = 0
                new_dose, next_interval, skip_dose, red_flag = \
                    self.aurora_dosing_table(previous_INR, previous_dose)
                if red_flag:
                    next_dose = 0.0
                    next_interval = 2
                elif skip_dose:
                    next_dose = 0.0
                    next_interval = skip_dose
                else:
                    next_dose = new_dose

        new_info = {
            'red_flag': red_flag,
            'skip_dose': skip_dose,
            'new_dose': new_dose,
            'number_of_stable_days': number_of_stable_days
        }

        return next_dose, next_interval, new_info

    @staticmethod
    def _stable_days(INR_start: float, INR_end: float, interval: int) -> int:
        '''
        Interpolate the INR values in the range and compute the number of days
        in therapeutic range of [2, 3].

        Arguments
        ---------
        INR_start:
            INR at the beginning of the period.

        INR_end:
            INR at the end of the period.

        interval:
            The number of days from start to end.

        Returns
        -------
        :
            The number of days in therapeuric range (TTR).

        Notes
        -----
        This method excludes `INR_start` and includes `INR_end` in
        computing TTR.

        '''
        return sum(2 <= INR_end + (INR_start - INR_end) * i / interval <= 3
                   for i in range(1, interval+1))

    @staticmethod
    def aurora_dosing_table(
            current_INR: float, dose: float) -> Tuple[float, int, int, bool]:
        '''
        Determine the dosing information, based on Aurora dosing table.

        Arguments
        ---------
        current_INR:
            The latest value of INR.

        dose:
            The latest dose prescribed.

        Returns
        -------
        :
            * The next dose
            * The time of the next test (in days).
            * The number of doses to skip.
            * Red flag for too high INR values.
        '''
        skip_dose = 0
        red_flag = False
        if current_INR < 1.50:
            dose = dose * 1.15
            next_test = 7
        elif current_INR < 1.80:
            dose = dose * 1.10
            next_test = 7
        elif current_INR < 2.00:
            dose = dose * 1.075
            next_test = 7
        elif current_INR <= 3.00:
            next_test = 28
        elif current_INR < 3.40:
            dose = dose * 0.925
            next_test = 7
        elif current_INR < 4.00:
            dose = dose * 0.9
            next_test = 7
        elif current_INR <= 5.00:
            skip_dose = 2
            dose = dose * 0.875
            next_test = 7
        else:
            red_flag = True
            next_test = 2
            dose = dose * 0.85

        return dose, next_test, skip_dose, red_flag

    @staticmethod
    def aurora_retesting_table(number_of_stable_days: int) -> int:
        '''
        Determine when the next test should be based on current number of
        stable days.

        Arguments
        ---------
        number_of_stable_days:
            Number of consecutive stable days.

        Returns
        -------
        :
            The time of the next test (in days).
        '''
        retesting_table = {1: 1, 2: 5, 7: 7, 14: 14, 28: 28}
        return retesting_table[max(
            map(lambda x: x if x <= number_of_stable_days else 1,
                retesting_table))]
