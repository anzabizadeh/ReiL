# -*- coding: utf-8 -*-
'''
ThreePhaseDosingProtocol class
==============================

A dosing protocol class that can contain three dosing protocols for
`initial`, `adjustment` and `maintenance` phases of dosing.
'''

from typing import Any, Dict, Tuple

from reil.healthcare.dosing_protocols import DosingProtocol


class ThreePhaseDosingProtocol:
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
