# -*- coding: utf-8 -*-
'''
IWPC class
==========

IWPC dosing protocol ('pharmacogenetic', 'clinical', 'modified').
'''

from math import exp
from typing import Any, Dict, List, Literal, Tuple

from reil.healthcare.dosing_protocols.dosing_protocol import DosingProtocol


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
