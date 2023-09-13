# -*- coding: utf-8 -*-
'''
CustomProtocol and CustomFullProtocol classes
=============================================

Aurora Dosing Protocol, based on `Ravvaz et al. (2017)
<https://doi.org/10.1161/circgenetics.117.001804>`_
'''

from collections.abc import Callable
from typing import Any, Literal

import reil.healthcare.dosing_protocols.three_phase_dosing_protocol as dp
from reil.healthcare.dosing_protocols.warfarin import IWPC


class CustomProtocol(dp.DosingProtocol):
    '''
    Custom protocols based on my research.
    '''
    def __init__(
            self,
            method: Literal[
                '123', 'joint', 'joint_791',
                'tandem_812', 'separate_891'] = '123') -> None:
        '''
        Arguments
        ---------
        method:
            ID of the trained models.
        '''
        self._method: Callable[[dict[str, Any]], dp.DosingDecision]
        if method == '123':
            self._method = self.protocol_123
        elif method == 'Joint':
            self._method = self.protocol_joint
        elif method == 'joint_791':
            self._method = self.protocol_joint_791
        elif method == 'tandem_812':
            self._method = self.protocol_tandem_812
        elif method == 'separate_891':
            self._method = self.protocol_separate_891
        else:
            raise ValueError(f'Unknown method {method}.\n')

        self.reset()

    def prescribe(
        self, patient: dict[str, Any], additional_info:  dp.AdditionalInfo
    ) -> tuple[dp.DosingDecision, dp.AdditionalInfo]:
        day = patient['day']

        if day == 1:
            raise ValueError('Protocol starts at day 4.')

        return self._method(patient), additional_info

    @staticmethod
    def protocol_123(patient: dict[str, Any]) -> dp.DosingDecision:
        previous_dose: float = patient['dose_history'][-1]
        previous_INR: float = patient['INR_history'][-1]

        if previous_INR <= 2.27:
            next_dose = previous_dose * 1.6  # 60%
        elif previous_INR <= 2.66:
            next_dose = previous_dose
        else:
            next_dose = previous_dose * 0.5  # -50%

        return dp.DosingDecision(next_dose, 7)

    @staticmethod
    def protocol_joint(patient: dict[str, Any]) -> dp.DosingDecision:
        previous_dose: float = patient['dose_history'][-1]
        previous_INR: float = patient['INR_history'][-1]

        if previous_INR <= 1.72:
            next_dose = previous_dose * 1.6  # 60%
            duration = 3
        elif previous_INR <= 2.24:
            next_dose = previous_dose * 1.4  # 40%
            duration = 1
        elif previous_INR <= 3.28:
            next_dose = previous_dose * 0.7  # -30%
            duration = 1
        else:
            next_dose = previous_dose * 0.5  # -50%
            duration = 28

        return dp.DosingDecision(next_dose, duration)

    @staticmethod
    def protocol_joint_791(patient: dict[str, Any]) -> dp.DosingDecision:
        previous_dose: float = patient['dose_history'][-1]
        previous_INR: float = patient['INR_history'][-1]

        if previous_INR <= 1.38:
            next_dose = previous_dose * 1.9  # 90%
            duration = 2
        elif previous_INR <= 1.70:
            next_dose = previous_dose  # 0%
            duration = 1
        elif previous_INR <= 2.50:
            next_dose = previous_dose * 1.9  # 90%
            duration = 1
        elif previous_INR <= 2.93:
            next_dose = previous_dose * 0.6  # -40%
            duration = 1
        else:
            next_dose = previous_dose * 0.2  # -80%
            duration = 3

        return dp.DosingDecision(next_dose, duration)

    @staticmethod
    def protocol_tandem_812(patient: dict[str, Any]) -> dp.DosingDecision:
        previous_dose: float = patient['dose_history'][-1]
        previous_INR: float = patient['INR_history'][-1]

        if previous_INR <= 2.56:
            next_dose = previous_dose * 1.6  # 60%
        elif previous_INR <= 3.15:
            next_dose = previous_dose * 0.5  # -50%
        else:
            next_dose = previous_dose * 0.3  # -70%

        return dp.DosingDecision(next_dose, 3)

    @staticmethod
    def protocol_separate_891(patient: dict[str, Any]) -> dp.DosingDecision:
        previous_dose: float = patient['dose_history'][-1]
        previous_INR: float = patient['INR_history'][-1]

        if previous_INR <= 1.49:
            next_dose = previous_dose * 1.8  # 80%
            duration = 2
        elif previous_INR <= 2.11:
            next_dose = previous_dose * 0.9  # -10%
            duration = 1
        elif previous_INR <= 2.21:
            next_dose = previous_dose * 0.8  # -20%
            duration = 2
        elif previous_INR <= 2.72:
            next_dose = previous_dose * 0.4  # -60%
            duration = 28
        else:
            next_dose = previous_dose * 0.2  # -80%
            duration = 28

        return dp.DosingDecision(next_dose, duration)


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
