# -*- coding: utf-8 -*-
'''
DosingProtocol class
====================

The base class for all basic dosing protocols.
'''

import collections
from typing import Any, Dict, Tuple

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
