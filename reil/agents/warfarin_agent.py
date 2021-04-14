# -*- coding: utf-8 -*-
'''
WarfarinAgent class
===================

An agent for warfarin modeling based on the doses define in Ravvaz et al (2017)
'''

import collections
import functools
from math import exp, log, sqrt
from reil.datatypes.feature import FeatureGenerator
from typing import Any, Dict, List, Optional, Tuple

from reil import agents
from reil.datatypes import FeatureArray
from typing_extensions import Literal

DoseInterval = collections.namedtuple('DoseInterval', 'dose, interval')


class DosingProtocol:
    '''
    Base class for all dosing protocol objects.
    '''

    def __init__(self) -> None:
        pass

    def prescribe(self,
                  patient: Dict[str, Any],
                  additional_info: Dict[str, Any]
                  ) -> Tuple[float, int, Dict[str, Any]]:
        '''
        Prescribe a dose for the given `patient` and `additional_info`.

        Arguments
        ---------
        patient:
            A dictionary of patient characteristics necessary to make dosing
            decisions.

        additional_info:
            A dictionary of information being communicated between protocols at
            each call to `prescribe`. These additional information are
            protocol-dependent.

        Returns
        -------
        :
            The prescribed dose along with updated `additional_info`.
        '''
        raise NotImplementedError

    def reset(self) -> None:
        '''Reset the dosing protocol'''
        pass


class Aurora(DosingProtocol):
    '''
    Aurora Dosing Protocol, based on Ravvaz et al. (2017)
    '''

    def prescribe(self,  # noqa: C901
                  patient: Dict[str, Any],
                  additional_info: Dict[str, Any]
                  ) -> Tuple[float, int, Dict[str, Any]]:
        today = patient['day']
        INRs = patient['INRs']
        previous_INR = INRs[-1]
        previous_dose = patient['Doses'][-1]
        previous_interval = patient['Intervals'][-1]

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


class IWPC(DosingProtocol):
    '''
    IWPC dosing protocol ('pharmacogenetic', 'clinical', 'modified').
    '''

    def __init__(self,
                 method: Literal['pharmacogenetic',
                                 'clinical',
                                 'modified'] = 'pharmacogenetic') -> None:
        '''
        Arguments
        ---------
        method:
            One of 'pharmacogenetic', 'clinical', 'modified'.
        '''
        if method.lower() == 'clinical':
            self._method = self.clinical
        elif method.lower() == 'modified':
            self._method = self.modified_pg
        elif method.lower() in ['pg', 'pharmacogenetic', 'default']:
            self._method = self.pg

        self.reset()

    def prescribe(self,
                  patient: Dict[str, Any],
                  additional_info: Dict[str, Any]
                  ) -> Tuple[float, int, Dict[str, Any]]:
        if self._doses:
            dose = self._doses.pop()
            interval = 1
        else:
            dose, interval = self._method(patient)
            if isinstance(dose, list):
                self._doses = dose
                self._doses.reverse()
                dose = self._doses.pop()
                interval = 1

        return dose, interval, {}

    @staticmethod
    def clinical(patient: Dict[str, Any]) -> Tuple[float, int]:
        '''
        Determine warfarin dose using clinical IWPC formula.

        Arguments
        ---------
        patient:
            A dictionary of patient characteristics including:
            - age
            - height (in)
            - weight (lb)
            - race ('Asian', 'Black', etc.)
            - amiodarone ('yes', 'no')

        Returns
        -------
        :
            The dose and time for the next test.

        Notes
        -----
        This method always returns 2 (days) for the next test.
        '''
        weekly_dose = (4.0376
                       - 0.2546 * (patient['age'] // 10)
                       # in to cm
                       + 0.0118 * patient['height'] * 2.54
                       # lb to kg
                       + 0.0134 * patient['weight'] * 0.454
                       - 0.6752 * (patient['race'] == 'Asian')
                       + 0.4060 * (patient['race'] == 'Black')
                       # # missing or mixed race
                       # + 0.0443 * (patient['race'] not in  [...])
                       # Enzyme inducer status
                       # (Fluvastatin is reductant not an inducer!)
                       + 1.2799 * 0
                       - 0.5695 * (patient['amiodarone'] == 'Yes')) ** 2

        return weekly_dose / 7.0, 2

    # only the initial dose (day <= 2)
    @staticmethod
    def pg(patient: Dict[str, Any]) -> Tuple[float, int]:
        '''
        Determine warfarin dose using pharmacogenetic IWPC formula.

        Arguments
        ---------
        patient:
            A dictionary of patient characteristics including:
            - age
            - height (in)
            - weight (lb)
            - VKORC1 ('G/G', 'G/A', 'A/A', etc.)
            - CYP2C9 ('*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3',...)
            - race ('Asian', 'Black', etc.)
            - amiodarone ('yes', 'no')

        Returns
        -------
        :
            The dose and time for the next test.

        Notes
        -----
        This method always returns 2 (days) for the next test.
        '''
        weekly_dose = (5.6044
                       # Based on Ravvaz (EU-PACT (page 18) uses year, Ravvaz
                       # (page 10 of annex) uses decades!
                       - 0.2614 * (patient['age'] // 10)
                       + 0.0087 * patient['height'] * 2.54  # in to cm
                       + 0.0128 * patient['weight'] * 0.454  # lb to kg
                       - 0.8677 * (patient['VKORC1'] == 'G/A')
                       - 1.6974 * (patient['VKORC1'] == 'A/A')
                       # Not in EU-PACT ?!!
                       - 0.4854 * \
                       int(patient['VKORC1'] not in [
                           'G/A', 'A/A', 'G/G'])
                       - 0.5211 * (patient['CYP2C9'] == '*1/*2')
                       - 0.9357 * (patient['CYP2C9'] == '*1/*3')
                       - 1.0616 * (patient['CYP2C9'] == '*2/*2')
                       - 1.9206 * (patient['CYP2C9'] == '*2/*3')
                       - 2.3312 * (patient['CYP2C9'] == '*3/*3')
                       # Not in EU-PACT
                       - 0.2188 * \
                       int(patient['CYP2C9'] not in [
                           '*1/*1', '*1/*2', '*1/*3',
                           '*2/*2', '*2/*3', '*3/*3'])
                       # Not in EU-PACT
                       - 0.1092 * (patient['race'] == 'Asian')
                       # Not in EU-PACT
                       - 0.2760 * (patient['race'] == 'Black')
                       # # missing or mixed race - Not in EU-PACT
                       # - 1.0320 * (patient['race'] not in  [...])
                       # Enzyme inducer status(Fluvastatin is reductant
                       # not an inducer!) (comment by Ravvaz)
                       + 1.1816 * 0
                       - 0.5503 * (patient['amiodarone'] == 'Yes')) ** 2

        return weekly_dose / 7.0, 2

    @staticmethod
    def modified_pg(patient: Dict[str, Any]) -> Tuple[List[float], int]:
        '''
        Determine warfarin dose using the modified pharmacogenetic IWPC
        formula.

        Arguments
        ---------
        patient:
            A dictionary of patient characteristics including:
            - age
            - height (in)
            - weight (lb)
            - VKORC1 ('G/G', 'G/A', 'A/A', etc.)
            - CYP2C9 ('*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3',...)
            - amiodarone ('yes', 'no')

        Returns
        -------
        :
            A list of dose and time for the next test.

        Notes
        -----
        This method always returns 1 (day) for the next test.
        '''
        weekly_dose = (5.6044
                       # Based on Ravvaz (EU-PACT (page 18) uses year,
                       # Ravvaz (page 10 of annex) uses decades!
                       - 0.2614 * (patient['age'] / 10)
                       # in to cm
                       + 0.0087 * patient['height'] * 2.54
                       # lb to kg
                       + 0.0128 * patient['weight'] * 0.454
                       - 0.8677 * (patient['VKORC1'] == 'G/A')
                       - 1.6974 * (patient['VKORC1'] == 'A/A')
                       - 0.5211 * (patient['CYP2C9'] == '*1/*2')
                       - 0.9357 * (patient['CYP2C9'] == '*1/*3')
                       - 1.0616 * (patient['CYP2C9'] == '*2/*2')
                       - 1.9206 * (patient['CYP2C9'] == '*2/*3')
                       - 2.3312 * (patient['CYP2C9'] == '*3/*3')
                       - 0.5503 * (patient['amiodarone'] == 'Yes')) ** 2

        k = {'*1/*1': 0.0189,
             '*1/*2': 0.0158,
             '*1/*3': 0.0132,
             '*2/*2': 0.0130,
             '*2/*3': 0.0090,
             '*3/*3': 0.0075
             }
        LD3 = weekly_dose / ((1 - exp(-24*k[patient['CYP2C9']])) * (
            1 + exp(-24*k[patient['CYP2C9']]) + exp(-48*k[patient['CYP2C9']])))
        # The following dose calculation is based on EU-PACT report page 19
        # Ravvaz uses the same formula, but uses weekly dose. However,
        # EU-PACT explicitly mentions "predicted daily dose (D)"
        doses = [(1.5 * LD3 - 0.5 * weekly_dose) / 7,
                 LD3 / 7,
                 (0.5 * LD3 + 0.5 * weekly_dose) / 7]

        return doses, 1

    def reset(self) -> None:
        '''Reset the dosing protocol'''
        self._doses = []


class Lenzini(DosingProtocol):
    '''Lenzini warfarin dosing protocol based on Lenzini (2010).'''

    def prescribe(self,
                  patient: Dict[str, Any],
                  additional_info: Dict[str, Any]
                  ) -> Tuple[float, int, Dict[str, Any]]:
        if patient['day'] != 4:
            raise ValueError('Lenzini can only be called on day 4.')

        dose = exp(
            3.10894
            - 0.00767 * patient['age']
            - 0.51611 * log(patient['INRs'][-1])  # Natural log
            - 0.23032 * (1 * (patient['VKORC1'] == 'G/A')  # Heterozygous
                         + 2 * (patient['VKORC1'] == 'A/A'))  # Homozygous
            - 0.14745 * (1 * (patient['CYP2C9'] in ['*1/*2', '*2/*3'])  # Het.
                         + 2 * (patient['CYP2C9'] == '*2/*2'))  # Homozygous
            - 0.30770 * (1 * (patient['CYP2C9'] in ['*1/*3', '*2/*3'])  # Het.
                         + 2 * (patient['CYP2C9'] == '*3/*3'))  # Homozygous
            + 0.24597 * \
            sqrt(patient['height'] * 2.54 * \
                 patient['weight'] * 0.454 / 3600)  # BSA
            + 0.26729 * 2.5  # target INR
            - 0.10350 * (patient['amiodarone'] == 'Yes')
            + 0.01690 * patient['Doses'][-2]
            + 0.02018 * patient['Doses'][-3]
            # available if INR is measured on day 5
            + 0.01065 * patient['Doses'][-4]
        ) / 7

        return dose, 2, {}


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
        previous_INR = patient['INRs'][-1]

        if not dose_interval_list:
            today = patient['day']

            if self._enforce_day_ge_8 and today < 8:
                raise ValueError('Intermountain is only valid for day>=8.')

            if self._enforce_day_ge_8 and today == 8:
                dose_list = patient['Doses']
                interval_list = patient['Intervals']
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
                previous_dose = patient['Doses'][-1]

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


class CompositeDosingProtocol:
    '''
    A dosing protocol class that can contain three dosing protocols for
    `initial`, `adjustment` and `maintenance` phases of dosing.
    '''

    def __init__(self,
                 initial_protocol: DosingProtocol,
                 adjustment_protocol: DosingProtocol,
                 maintenance_protocol: DosingProtocol) -> None:
        '''
        Arguments
        ---------
        initial_protocol:
            A dosing protocol for the initial phase of dosing.

        adjustment_protocol
            A dosing protocol for the adjustment phase of dosing.

        maintenance_protocol
            A dosing protocol for the maintenance phase of dosing.
        '''
        self._initial_protocol = initial_protocol
        self._adjustment_protocol = adjustment_protocol
        self._maintenance_protocol = maintenance_protocol
        self._additional_info = {}

    def prescribe(self, patient: Dict[str, Any]) -> Tuple[float, int]:
        '''
        Prescribe a dose and next test (in days) for the given `patient`.

        Arguments
        ---------
        patient:
            A dictionary of patient characteristics necessary to make dosing
            decisions.

        Returns
        -------
        :
            The prescribed dose and the time of the next test (in days).
        '''
        raise NotImplementedError

    def reset(self) -> None:
        '''Reset the dosing protocol.'''
        self._initial_protocol.reset()
        self._adjustment_protocol.reset()
        self._maintenance_protocol.reset()
        self._additional_info = {}


class AAA(CompositeDosingProtocol):
    '''
    A composite dosing protocol with `Aurora` in all phases.
    '''

    def __init__(self) -> None:
        aurora_instance = Aurora()
        super().__init__(aurora_instance, aurora_instance, aurora_instance)

    def prescribe(self, patient: Dict[str, Any]) -> Tuple[float, int]:
        dose, interval, self._additional_info = \
            self._initial_protocol.prescribe(patient, self._additional_info)

        return dose, interval


class CAA(CompositeDosingProtocol):
    '''
    A composite dosing protocol with clinical `IWPC` in initial phase, and
    `Aurora` in adjustment and maintenance phases.
    '''

    def __init__(self) -> None:
        iwpc_instance = IWPC('clinical')
        aurora_instance = Aurora()
        super().__init__(iwpc_instance, aurora_instance, aurora_instance)

    def prescribe(self, patient: Dict[str, Any]) -> Tuple[float, int]:
        fn = (self._initial_protocol if patient['day'] <= 2
              else self._adjustment_protocol)
        dose, interval, self._additional_info = fn.prescribe(
            patient, self._additional_info)

        return dose, interval


class PGAA(CompositeDosingProtocol):
    '''
    A composite dosing protocol with pharmacogenetic `IWPC` in initial phase,
    and `Aurora` in adjustment and maintenance phases.
    '''

    def __init__(self) -> None:
        iwpc_instance = IWPC('pharmacogenetic')
        aurora_instance = Aurora()
        super().__init__(iwpc_instance, aurora_instance, aurora_instance)

    def prescribe(self, patient: Dict[str, Any]) -> Tuple[float, int]:
        fn = (self._initial_protocol if patient['day'] <= 2
              else self._adjustment_protocol)
        dose, interval, self._additional_info = fn.prescribe(
            patient, self._additional_info)

        return dose, interval


class PGPGA(CompositeDosingProtocol):
    '''
    A composite dosing protocol with modified `IWPC` in initial phase,
    `Lenzini` in adjustment phase, and `Aurora` in maintenance phase.
    '''

    def __init__(self) -> None:
        iwpc_instance = IWPC('modified')
        lenzini_instance = Lenzini()
        aurora_instance = Aurora()
        super().__init__(iwpc_instance, lenzini_instance, aurora_instance)

    def prescribe(self, patient: Dict[str, Any]) -> Tuple[float, int]:
        if patient['day'] <= 3:
            fn = self._initial_protocol
        elif patient['day'] <= 5:
            fn = self._adjustment_protocol
        else:
            fn = self._maintenance_protocol

        dose, interval, self._additional_info = fn.prescribe(
            patient, self._additional_info)

        return dose, interval


class PGPGI(CompositeDosingProtocol):
    '''
    A composite dosing protocol with modified `IWPC` in initial phase,
    `Lenzini` in adjustment phase, and `Intermountain` in maintenance phase.
    '''

    def __init__(self) -> None:
        iwpc_instance = IWPC('modified')
        lenzini_instance = Lenzini()
        intermountain_instance = Intermountain(enforce_day_ge_8=False)
        super().__init__(
            iwpc_instance, lenzini_instance, intermountain_instance)

    def prescribe(self, patient: Dict[str, Any]) -> Tuple[float, int]:
        if patient['day'] <= 3:
            fn = self._initial_protocol
        elif patient['day'] <= 5:
            fn = self._adjustment_protocol
        else:
            fn = self._maintenance_protocol

        dose, interval, self._additional_info = fn.prescribe(
            patient, self._additional_info)

        return dose, interval


class WarfarinAgent(agents.NoLearnAgent):
    '''
    An `agent` that prescribes dose for a warfarin `subject`,
    based on the dosing protocols defined in Ravvaz et al (2017).
    '''

    def __init__(self,
                 study_arm: Literal['aaa', 'caa', 'pgaa',
                                    'pgpgi', 'pgpga'] = 'aaa',
                 **kwargs: Any):
        '''
        Arguments
        ---------
        study_arm:
            One of available study arms: aaa, caa, pgaa, pgpgi, pgpga
        '''
        super().__init__(**kwargs)

        if study_arm.lower() in ['aaa', 'ravvaz aaa', 'ravvaz_aaa']:
            self._protocol = AAA()
        elif study_arm.lower() in ['caa', 'ravvaz caa', 'ravvaz_caa']:
            self._protocol = CAA()
        elif study_arm.lower() in ['pgaa', 'ravvaz pgaa', 'ravvaz_pgaa']:
            self._protocol = PGAA()
        elif study_arm.lower() in ['pgpgi', 'ravvaz pgpgi', 'ravvaz_pgpgi']:
            self._protocol = PGPGI()
        elif study_arm.lower() in ['pgpga', 'ravvaz pgpga', 'ravvaz_pgpga']:
            self._protocol = PGPGA()

        self._dose_gen = FeatureGenerator.numerical(
                name='dose', lower=0.0, upper=15.0)
        self._interval_gen = FeatureGenerator(
                name='interval', lower=1, upper=28)

    def act(self,
            state: FeatureArray,
            actions: Optional[Tuple[FeatureArray, ...]] = None,
            epoch: Optional[int] = 0) -> FeatureArray:
        '''
        Generate the dosing `action` based on the `state` and current dosing
        protocol.

        Arguments
        ---------
        state:
            The state for which the action should be returned.

        actions:
            The set of possible actions to choose from.

        epoch:
            The epoch in which the agent is acting.

        Returns
        -------
        :
            The action
        '''
        patient = state.value
        patient['day'] += 1

        dose, interval = self._protocol.prescribe(patient)

        # Cuts out the dose if it is >15.0
        return FeatureArray([
            self._dose_gen(min(dose, 15.0)),
            self._interval_gen(min(interval, 28))
        ])

    def reset(self):
        '''Resets the agent at the end of a learning epoch.'''
        self._protocol.reset()

    def __repr__(self) -> str:
        try:
            return super().__repr__() + f' arm: {self._protocol}'
        except NameError:
            return super().__repr__()

# Implementation that matches RAVVAZ dataset.
# -------------------------------------------
#
# def aurora(self, patient: Dict[str, Any]) -> float:
#     if self._red_flag:
#         if self._retest_day > patient['day']:
#             return_value = 0.0
#         elif patient['INRs'][-1] > 3.0:
#             self._retest_day = patient['day'] + 2
#             return_value = 0.0
#         else:
#             self._red_flag = False
#             self._retest_day = patient['day'] + 7
#             return_value = self._dose
#         return return_value

#     next_test = 2
#     if patient['day'] <= 2:
#         self._dose = 10.0 if patient['age'] < 65.0 else 5.0
#     elif patient['day'] <= 4:
#         day_2_INR = (patient['INRs'][-1] if patient['day'] == 3
#                      else patient['INRs'][-2])
#         if day_2_INR >= 2.0:
#             self._dose = 5.0
#             if day_2_INR <= 3.0:
#                 self._early_therapeutic = True

#             self._number_of_stable_days, next_test = \
#                 self._aurora_retesting_table(patient['INRs'][-1],
#                                              self._number_of_stable_days,
#                                              self._early_therapeutic)
#         else:
#             self._dose, next_test, _, _ = self._aurora_dosing_table(
#                 day_2_INR, self._dose)
#     else:
#         self._number_of_stable_days, next_test = \
#             self._aurora_retesting_table(
#               patient['INRs'][-1],
#               self._number_of_stable_days,
#               self._early_therapeutic)

#     if next_test == -1:
# #         self._early_therapeutic = False
#         self._number_of_stable_days = 0
#         self._dose, next_test, self._skip_dose, self._red_flag = \
#           self._aurora_dosing_table(patient['INRs'][-1], self._dose)

#     self._retest_day = patient['day'] + next_test

#     return self._dose if self._skip_dose == 0 else 0.0

# def _aurora_dosing_table(self,
#     current_INR: float, dose: float) -> Tuple[float, int, int, bool]:
#     skip_dose = 0
#     red_flag = False
#     if current_INR < 1.50:
#         dose = dose * 1.15
#         next_test = 7
#     elif current_INR < 1.80:
#         dose = dose * 1.10
#         next_test = 7
#     elif current_INR < 2.00:
#         dose = dose * 1.075
#         next_test = 7
#     elif current_INR <= 3.00:
#         next_test = 28
#     elif current_INR < 3.40:
#         dose = dose * 0.925
#         next_test = 7
#     elif current_INR < 4.00:
#         dose = dose * 0.9
#         next_test = 7
#     elif current_INR <= 5.00:
#         skip_dose = 1  # 2
#         dose = dose * 0.875
#         next_test = 7
#     else:
#         red_flag = True
#         next_test = 2
#         dose = dose * 0.85

#     return dose, next_test, skip_dose, red_flag

# def _aurora_retesting_table(self,
#   current_INR: float, number_of_stable_days: int,
#   early_therapeutic: bool = False) -> Tuple[int, int]:
#     # next_test = {0: 1, 1: 1, 2: 5, 7: 7, 14: 14, 28: 28}
#     # NOTE: When patient gets into the range early on
#     #      (before the maintenance period), we have 1: 2,
#     # but when patient is in maintenance,\
#     # we have 1: 6. I track the former in the main aurora method.

#     if early_therapeutic:
#         next_test = {0: 1, 1: 2, 3: 4, 7: 6, 13: 6, 19: 13, 32: 27}
#         max_gap = 32
#     else:
#         next_test = {0: 1, 1: 6, 7: 6, 13: 13, 26: 27}
#         max_gap = 26
#     if 2.0 <= current_INR <= 3.0:
#         number_of_stable_days = min(
#             number_of_stable_days + next_test[number_of_stable_days],
#             max_gap)
#     else:
#         return -1, -1

#     return number_of_stable_days, next_test[number_of_stable_days]
