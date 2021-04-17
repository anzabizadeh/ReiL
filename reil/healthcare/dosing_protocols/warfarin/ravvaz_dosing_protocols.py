# -*- coding: utf-8 -*-
'''
AAA, CAA, PGAA, PGPGA, PGPGI classes
====================================

Study arms in Ravvaz et al. (2017).
'''

from typing import Any, Dict, Tuple

from reil.healthcare.dosing_protocols import ThreePhaseDosingProtocol
from reil.healthcare.dosing_protocols.warfarin import (IWPC, Aurora,
                                                       Intermountain, Lenzini)


class AAA(ThreePhaseDosingProtocol):
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


class CAA(ThreePhaseDosingProtocol):
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


class PGAA(ThreePhaseDosingProtocol):
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


class PGPGA(ThreePhaseDosingProtocol):
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


class PGPGI(ThreePhaseDosingProtocol):
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
