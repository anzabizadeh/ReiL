# -*- coding: utf-8 -*-
'''
CustomProtocol and CustomFullProtocol classes
=============================================

Aurora Dosing Protocol, based on `Ravvaz et al. (2017)
<https://doi.org/10.1161/circgenetics.117.001804>`_
'''

from typing import Any, Callable, Literal

import reil.healthcare.dosing_protocols.three_phase_dosing_protocol as dp
from reil.healthcare.dosing_protocols.warfarin import IWPC


class CustomProtocol(dp.DosingProtocol):
    '''
    Custom protocols based on my research.
    '''
    def __init__(
            self,
            method: Literal['123'] = '123') -> None:
        '''
        Arguments
        ---------
        method:
            ID of the trained models.
        '''
        self._method: Callable[[dict[str, Any]], float]
        if method == '123':
            self._method = self.protocol_123
        else:
            raise ValueError(f'Unknown method {method}.\n')

        self.reset()

    def prescribe(
        self, patient: dict[str, Any], additional_info:  dp.AdditionalInfo
    ) -> tuple[dp.DosingDecision, dp.AdditionalInfo]:
        day = patient['day']

        if day == 1:
            raise ValueError('Protocol starts at day 4.')

        return dp.DosingDecision(self._method(patient), 7), additional_info

    @staticmethod
    def protocol_123(patient: dict[str, Any]) -> float:
        previous_dose: float = patient['dose_history'][-1]
        previous_INR: float = patient['INR_history'][-1]

        if previous_INR <= 2.27:
            next_dose = previous_dose * 1.6  # 60%
        elif previous_INR <= 2.66:
            next_dose = previous_dose
        else:
            next_dose = previous_dose * 0.5  # -50%

        return next_dose


class CustomFullProtocol(dp.ThreePhaseDosingProtocol):
    '''
    A composite dosing protocol with modified `IWPC` in the initial phase
    (days 1, 2, and 3), and custom protocol in the rest.
    '''
    def __init__(self, method: Literal['123']) -> None:
        self._method = str(method)
        iwpc_instance = IWPC('modified')
        custom_protocol_instance = CustomProtocol(method)
        super().__init__(
            iwpc_instance, custom_protocol_instance, custom_protocol_instance)

    def prescribe(
            self, patient: dict[str, Any]) -> dp.DosingDecision:
        if patient['day'] <= 3:
            temp, self._additional_info = \
                self._initial_protocol.prescribe(
                    patient, self._additional_info)
            dosing_decision = dp.DosingDecision(temp.dose, 4 - patient['day'])
        else:
            dosing_decision, self._additional_info = \
                self._maintenance_protocol.prescribe(
                    patient, self._additional_info)

        return dosing_decision

    def __repr__(self) -> str:
        return super().__repr__() + self._method
