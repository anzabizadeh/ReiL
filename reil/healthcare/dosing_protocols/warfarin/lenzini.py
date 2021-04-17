# -*- coding: utf-8 -*-
'''
Lenzini class
=============

Lenzini warfarin dosing protocol based on Lenzini (2010).
'''

from math import exp, log, sqrt
from typing import Any, Dict, Tuple

from reil.healthcare.dosing_protocols import DosingProtocol


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
