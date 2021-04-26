# -*- coding: utf-8 -*-
'''
Intermountain class
===================

Intermountain warfarin dosing protocol based on `Anderson et al. (2007)
supplements Appendix B
<https://www.ahajournals.org/doi/10.1161/circulationaha.107.737312>`_
'''

import functools
from typing import Any, Dict, List, Tuple

from reil.healthcare.dosing_protocols import DosingProtocol, DoseInterval


class Intermountain(DosingProtocol):
    '''
    Intermountain warfarin dosing protocol based on `Anderson et al. (2007)
    supplements Appendix B
    <https://www.ahajournals.org/doi/10.1161/circulationaha.107.737312>`_
    '''

    def __init__(self, enforce_day_ge_8: bool = True) -> None:
        super().__init__()
        self._enforce_day_ge_8 = enforce_day_ge_8

    def prescribe(self,
                  patient: Dict[str, Any],
                  additional_info: Dict[str, Any]
                  ) -> Tuple[float, int, Dict[str, Any]]:
        dose_interval_list = additional_info.get('dose_interval_list', [])
        last_zone = additional_info.get('last_zone', '')
        previous_INR = patient['INR_history'][-1]

        if not dose_interval_list:
            today = patient['day']

            if self._enforce_day_ge_8 and today < 8:
                raise ValueError('Intermountain is only valid for day>=8.')

            if self._enforce_day_ge_8 and today == 8:
                dose_list = patient['dose_history']
                interval_list = patient['interval_history']
                dose_list = functools.reduce(
                    lambda x, y: x+y,
                    ([dose_list[-i]]*interval_list[-i]
                     for i in range(len(interval_list), 0, -1)))

                if len(dose_list) < 3:
                    raise ValueError(
                        'Intermountain requires doses for days 5 to 7 '
                        'for dosing on day 8.')

                previous_dose = sum(dose_list[-3:])/3
            else:
                previous_dose = patient['dose_history'][-1]

            dose_interval_list, last_zone = self.intermountain_dosing_table(
                previous_INR, last_zone, previous_dose)

        else:
            if dose_interval_list[0].interval == -1:
                dose_interval_list, last_zone = \
                    self.intermountain_dosing_table(
                        previous_INR, last_zone, dose_interval_list[0].dose)

        additional_info['last_zone'] = last_zone
        additional_info['dose_interval_list'] = dose_interval_list[1:]

        return (dose_interval_list[0].dose,
                dose_interval_list[0].interval, additional_info)

    @staticmethod  # noqa: C901
    def intermountain_dosing_table(
            INR: float,
            last_zone: str,
            daily_dose: float) -> Tuple[List[DoseInterval], str]:
        '''
        Determine the dosing information, based on Intermountain dosing table.

        Arguments
        ---------
        current_INR:
            The latest value of INR.

        last_zone:
            The last zone the patient was in.
            * action point low
            * red zone low
            * yellow zone low
            * green zone
            * yellow zone high
            * red zone high
            * action point high

        daily_dose:
            The latest daily dose prescribed.

        Returns
        -------
        :
            * A list of `DoseInterval`s. It always includes the new daily dose
              and the new next test (in days). If an immediate dose is
              necessary, the first item will be the immediate dose and the next
              test day.
            * The new zone that patient's INR falls into.
        '''
        zone = Intermountain.zone(INR)

        weekly_dose = daily_dose * 7

        immediate_dose: float = -1.0
        immediate_interval: int = -1

        next_interval: int = {
            # -1 because immediate_interval = 1
            'action point low': (5-1, 14-1),
            'red zone low': (7-1, 14-1),  # -1 because immediate_interval = 1
            'yellow zone low': (14, 14),
            'green zone': (14, 28),
            'yellow zone high': (14, 14),
            'red zone high': (7-1, 14-1),  # -1 because immediate_interval = 1
            'action point high': (7, 14)
        }[zone][zone == last_zone]

        if last_zone == 'action point high':
            if zone in ['yellow zone low', 'green zone', 'yellow zone high']:
                weekly_dose *= 0.85
                next_interval = 7
            else:
                immediate_dose = 0.0
                immediate_interval = 2
                next_interval = -1

        elif zone == 'action point low':
            # (immediate extra dose) average 5-7 for day 8
            immediate_dose = daily_dose * 2
            immediate_interval = 1
            weekly_dose *= 1.10
        elif zone == 'red zone low':
            # (extra half dose) average 5-7 for day 8
            immediate_dose = daily_dose * 1.5
            immediate_interval = 1
            weekly_dose *= 1.05
        elif zone == 'yellow zone low':
            if zone == last_zone:
                weekly_dose *= 1.05
        # elif zone == 'green zone':
        #     pass
        elif zone == 'yellow zone high':
            if zone == last_zone:
                weekly_dose *= 0.95
        elif zone == 'red zone high':
            immediate_dose = daily_dose * 0.5 if INR < 4 else 0.0
            immediate_interval = 1
            weekly_dose *= 0.90
        elif zone == 'action point high':
            immediate_dose = 0.0
            immediate_interval = 2
            next_interval = -1

        dose_intervals = list(
            (DoseInterval(immediate_dose, immediate_interval),)
            if (immediate_dose >= 0.0 and immediate_interval > 0)
            else []
        ) + [DoseInterval(weekly_dose / 7, next_interval)]

        return dose_intervals, zone

    @staticmethod
    def zone(INR: float) -> str:
        '''
        Determine the zone based on patient's INR.

        Arguments
        ---------
        INR:
            the value of a patient's INR.

        Returns
        -------
        :
            Name of the dose, one of:
            * action point low
            * red zone low
            * yellow zone low
            * green zone
            * yellow zone high
            * red zone high
            * action point high
        '''
        if INR < 1.60:
            z = 'action point low'
        elif INR < 1.80:
            z = 'red zone low'
        elif INR < 2.00:
            z = 'yellow zone low'
        elif INR <= 3.00:
            z = 'green zone'
        elif INR < 3.40:
            z = 'yellow zone high'
        elif INR < 5.00:
            z = 'red zone high'
        else:
            z = 'action point high'

        return z
