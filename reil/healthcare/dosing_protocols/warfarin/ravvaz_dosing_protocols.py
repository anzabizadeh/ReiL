# -*- coding: utf-8 -*-
'''
AAA, CAA, PGAA, PGPGA, PGPGI classes
====================================

Study arms in `Ravvaz et al. (2017)
<https://doi.org/10.1161/circgenetics.117.001804>`_
'''

from typing import Any, Dict

from reil.healthcare.dosing_protocols import (DosingDecision,
                                              ThreePhaseDosingProtocol)
from reil.healthcare.dosing_protocols.warfarin import (IWPC, Aurora,
                                                       Intermountain, Lenzini)


class AAA(ThreePhaseDosingProtocol):
    '''
    A composite dosing protocol with `Aurora` in all phases.
    '''

    def __init__(self) -> None:
        aurora_instance = Aurora()
        super().__init__(aurora_instance, aurora_instance, aurora_instance)

    def prescribe(
            self, patient: Dict[str, Any]) -> DosingDecision:
        dosing_decision, self._additional_info = \
            self._initial_protocol.prescribe(patient, self._additional_info)

        return dosing_decision

    def __repr__(self) -> str:
        return super().__repr__() + '[AAA]'


class CAA(ThreePhaseDosingProtocol):
    '''
    A composite dosing protocol with clinical `IWPC` in the initial phase
    (days 1 and 2), and `Aurora` in the adjustment and maintenance phases.
    '''

    def __init__(self) -> None:
        iwpc_instance = IWPC('clinical')
        aurora_instance = Aurora()
        super().__init__(iwpc_instance, aurora_instance, aurora_instance)

    def prescribe(
            self, patient: Dict[str, Any]) -> DosingDecision:
        fn = (self._initial_protocol if patient['day'] <= 2
              else self._adjustment_protocol)
        dosing_decision, self._additional_info = fn.prescribe(
            patient, self._additional_info)

        return dosing_decision

    def __repr__(self) -> str:
        return super().__repr__() + '[CAA]'


class PGAA(ThreePhaseDosingProtocol):
    '''
    A composite dosing protocol with pharmacogenetic `IWPC` in the initial
    phase (days 1 and 2), and `Aurora` in the adjustment and maintenance
    phases.
    '''

    def __init__(self) -> None:
        iwpc_instance = IWPC('pharmacogenetic')
        aurora_instance = Aurora()
        super().__init__(iwpc_instance, aurora_instance, aurora_instance)

    def prescribe(
            self, patient: Dict[str, Any]) -> DosingDecision:
        fn = (self._initial_protocol if patient['day'] <= 2
              else self._adjustment_protocol)
        dosing_decision, self._additional_info = fn.prescribe(
            patient, self._additional_info)

        return dosing_decision

    def __repr__(self) -> str:
        return super().__repr__() + '[PGAA]'


class PGPGA(ThreePhaseDosingProtocol):
    '''
    A composite dosing protocol with modified `IWPC` in the initial phase
    (days 1, 2, and 3), `Lenzini` in the adjustment phase (days 4 and 5),
    and `Aurora` in the maintenance phase.
    '''

    def __init__(self) -> None:
        iwpc_instance = IWPC('modified')
        lenzini_instance = Lenzini()
        aurora_instance = Aurora()
        super().__init__(iwpc_instance, lenzini_instance, aurora_instance)

    def prescribe(
            self, patient: Dict[str, Any]) -> DosingDecision:
        if patient['day'] <= 3:
            fn = self._initial_protocol
        elif patient['day'] <= 5:
            fn = self._adjustment_protocol
        else:
            fn = self._maintenance_protocol

        dosing_decision, self._additional_info = fn.prescribe(
            patient, self._additional_info)

        return dosing_decision

    def __repr__(self) -> str:
        return super().__repr__() + '[PGPGA]'


class PGPGI(ThreePhaseDosingProtocol):
    '''
    A composite dosing protocol with modified `IWPC` in the initial phase
    (days 1, 2, and 3), `Lenzini` in the adjustment phase (days 4 and 5),
    and `Intermountain` in the maintenance phase.
    '''

    def __init__(self) -> None:
        iwpc_instance = IWPC('modified')
        lenzini_instance = Lenzini()
        intermountain_instance = Intermountain(enforce_day_ge_8=False)
        super().__init__(
            iwpc_instance, lenzini_instance, intermountain_instance)

    def prescribe(
            self, patient: Dict[str, Any]) -> DosingDecision:
        if patient['day'] <= 3:
            fn = self._initial_protocol
        elif patient['day'] <= 5:
            fn = self._adjustment_protocol
        else:
            fn = self._maintenance_protocol

        dosing_decision, self._additional_info = fn.prescribe(
            patient, self._additional_info)

        return dosing_decision

    def __repr__(self) -> str:
        return super().__repr__() + '[PGPGI]'
