# -*- coding: utf-8 -*-
'''
DistilledTree class
===================

A three-phase dosing protocol mirroring `PGPGA` (modified `IWPC` on days 1-3,
`Lenzini` on days 4-5) with the maintenance phase replaced by a decision tree
distilled from a trained PPO policy. Used for `[EXP-C2-005]` per
`50_paper1_chapter2_canonical.md`.
'''

from typing import Any, Protocol

import numpy as np

from reil.healthcare.dosing_protocols.dosing_protocol import (AdditionalInfo,
                                                              DosingDecision,
                                                              DosingProtocol)
from reil.healthcare.dosing_protocols.three_phase_dosing_protocol import \
    ThreePhaseDosingProtocol
from reil.healthcare.dosing_protocols.warfarin.iwpc import IWPC
from reil.healthcare.dosing_protocols.warfarin.lenzini import Lenzini


class _TreeLike(Protocol):
    def predict(self, X: Any) -> Any: ...


class IntervalTree:
    '''Piecewise-constant classifier over a single feature (INR).

    Drop-in replacement for sklearn's `predict([[x]]) -> [class_label]`
    interface, used as a compact equivalent to a decision tree fit on a
    single feature. `thresholds` are ascending and split the input axis
    into `len(thresholds) + 1` intervals; `classes[i]` is the label for
    the i-th interval. The right boundary of each interval is closed,
    matching sklearn's `x <= threshold` semantics — input `x` at a
    threshold maps to the interval to its LEFT.

    Construction is left to callers (typically a redundancy-merge pass
    over a fitted `DecisionTreeClassifier`); this class only consumes
    the merged representation at inference time.
    '''

    def __init__(self, thresholds: Any, classes: Any) -> None:
        self._thresholds = np.asarray(thresholds, dtype=float)
        self._classes = np.asarray(classes)
        if self._classes.shape[0] != self._thresholds.shape[0] + 1:
            raise ValueError(
                f'classes (len={self._classes.shape[0]}) must have exactly '
                f'one more entry than thresholds '
                f'(len={self._thresholds.shape[0]}).')

    def predict(self, X: Any) -> np.ndarray:
        # sklearn's DecisionTreeClassifier casts X to float32 in
        # `_check_input` before tree traversal. We mirror that cast so the
        # comparison `x <= threshold` produces the same answer as the
        # original tree at boundary values that round across a threshold
        # when stored as float32.
        x = np.asarray(X, dtype=np.float32).reshape(-1)
        idx = np.searchsorted(self._thresholds, x, side='left')
        return self._classes[idx]

    @property
    def thresholds(self) -> np.ndarray:
        return self._thresholds

    @property
    def classes(self) -> np.ndarray:
        return self._classes

    @property
    def n_intervals(self) -> int:
        return int(self._classes.shape[0])

    def __repr__(self) -> str:
        return f'IntervalTree(K={self.n_intervals})'


class _TreeMaintenance(DosingProtocol):
    '''Maintenance protocol backed by a fitted classifier mapping INR to
    `int(percent_action * 10)` (consistent with `bin/distill_tier3.py`).

    The classifier is duck-typed: any object exposing `predict([[INR]]) -> [int]`
    works (sklearn `DecisionTreeClassifier` is the canonical case).
    '''

    def __init__(self, tree: _TreeLike) -> None:
        super().__init__()
        self._tree = tree

    def prescribe(
            self,
            patient: dict[str, Any],
            additional_info: AdditionalInfo
    ) -> tuple[DosingDecision, AdditionalInfo]:
        inr: float = patient['INR_history'][-1]
        previous_dose: float = patient['dose_history'][-1]
        label = int(self._tree.predict([[inr]])[0])
        action_pct = label / 10.0
        next_dose = previous_dose * (1.0 + action_pct)
        return DosingDecision(next_dose, 7), additional_info


class _TreeMaintenance2D(DosingProtocol):
    '''Direction-aware maintenance protocol: a classifier mapping the pair
    (current INR, ΔINR) -> `int(percent_action * 10)`, where ΔINR = INR(now) −
    INR(previous maintenance decision). Lets an interpretable tree condition on
    the INR *trend*, which a single-INR tree cannot represent (see the T2.3
    velocity finding).

    `prev_inr` is per episode, and the episode boundary is detected from the
    patient's own clock rather than from `reset()`. This is deliberate:
    `WarfarinAgent.reset()` is called once per learning *iteration*, not once
    per patient, so a `reset()`-cleared field leaks across patients. Before this
    was fixed, only the first patient of an evaluation pass saw
    :math:`\Delta INR = 0` on its first maintenance decision; every subsequent
    patient's first :math:`\Delta INR` was measured against the *previous*
    patient's final INR.

    The classifier is duck-typed: any object exposing
    `predict([[INR, dINR]]) -> [int]` works (sklearn `DecisionTreeClassifier`
    fit on two features is the canonical case).
    '''

    def __init__(self, tree: _TreeLike) -> None:
        super().__init__()
        self._tree = tree
        self._prev_inr: float | None = None
        self._prev_day: int | None = None

    def reset(self) -> None:
        super().reset()
        self._prev_inr = None
        self._prev_day = None

    def prescribe(
            self,
            patient: dict[str, Any],
            additional_info: AdditionalInfo
    ) -> tuple[DosingDecision, AdditionalInfo]:
        inr: float = patient['INR_history'][-1]
        previous_dose: float = patient['dose_history'][-1]
        day = int(patient['day'])

        # A new episode restarts the patient clock, so a day that does not
        # advance means we are looking at a different patient. Do NOT rely on
        # `reset()` for this: it fires per learning iteration, not per patient.
        if self._prev_day is None or day <= self._prev_day:
            self._prev_inr = None
        self._prev_day = day

        # First maintenance decision has no prior maintenance reading -> ΔINR = 0.
        d_inr: float = 0.0 if self._prev_inr is None else inr - self._prev_inr
        self._prev_inr = inr
        label = int(self._tree.predict([[inr, d_inr]])[0])
        action_pct = label / 10.0
        next_dose = previous_dose * (1.0 + action_pct)
        return DosingDecision(next_dose, 7), additional_info


class DistilledTree(ThreePhaseDosingProtocol):
    '''Three-phase protocol with modified `IWPC` (days 1-3), `Lenzini` (days
    4-5), and a distilled decision tree (day 6+) in maintenance.

    The day-1..5 scaffolding is identical to `PGPGA` so PTTR is directly
    comparable to the Aurora-maintenance baseline.
    '''

    def __init__(self, tree: _TreeLike) -> None:
        super().__init__(IWPC('modified'), Lenzini(), _TreeMaintenance(tree))

    def prescribe(self, patient: dict[str, Any]) -> DosingDecision:
        day: int = patient['day']
        if day <= 3:
            temp, self._additional_info = self._initial_protocol.prescribe(
                patient, self._additional_info)
            return DosingDecision(temp.dose, 4 - day)
        if day <= 5:
            temp, self._additional_info = self._adjustment_protocol.prescribe(
                patient, self._additional_info)
            return DosingDecision(temp.dose, 6 - day)
        dosing_decision, self._additional_info = \
            self._maintenance_protocol.prescribe(
                patient, self._additional_info)
        return dosing_decision

    def __repr__(self) -> str:
        return super().__repr__() + '[DistilledTree]'


class DistilledTree2D(ThreePhaseDosingProtocol):
    '''Like `DistilledTree`, but the maintenance tree is direction-aware: it
    consumes (current INR, ΔINR-since-last-decision). Day-1..5 scaffolding is
    identical to `PGPGA`/`DistilledTree` so PTTR is directly comparable.
    '''

    def __init__(self, tree: _TreeLike) -> None:
        super().__init__(IWPC('modified'), Lenzini(), _TreeMaintenance2D(tree))

    def prescribe(self, patient: dict[str, Any]) -> DosingDecision:
        day: int = patient['day']
        if day <= 3:
            temp, self._additional_info = self._initial_protocol.prescribe(
                patient, self._additional_info)
            return DosingDecision(temp.dose, 4 - day)
        if day <= 5:
            temp, self._additional_info = self._adjustment_protocol.prescribe(
                patient, self._additional_info)
            return DosingDecision(temp.dose, 6 - day)
        dosing_decision, self._additional_info = \
            self._maintenance_protocol.prescribe(
                patient, self._additional_info)
        return dosing_decision

    def __repr__(self) -> str:
        return super().__repr__() + '[DistilledTree2D]'
