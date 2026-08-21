# -*- coding: utf-8 -*-
"""DeltaINR in the two-feature distilled student must not leak across patients.

Regression test for a defect found on 2026-08-17 while re-checking the Paper-2
supplementary two-feature result. `_TreeMaintenance2D` kept `_prev_inr` and
cleared it in `reset()`, but `WarfarinAgent.reset()` runs once per learning
*iteration*, not once per patient. Across an evaluation pass only the first
patient saw DeltaINR = 0 on its first maintenance decision; every later
patient's first DeltaINR was measured against the previous patient's final INR.

These tests drive the protocol directly, so they hold regardless of whether any
caller happens to invoke `reset()`.
"""
from __future__ import annotations

import pytest

from reil.healthcare.dosing_protocols.warfarin.distilled_tree import \
    DistilledTree2D


class _RecordingTree:
    """Duck-typed classifier that records the (INR, dINR) pairs it is asked about."""

    def __init__(self) -> None:
        self.seen: list[tuple[float, float]] = []

    def predict(self, X):
        inr, d_inr = float(X[0][0]), float(X[0][1])
        self.seen.append((inr, d_inr))
        return [0]                      # always "hold"; the action is irrelevant here


def _patient(day: int, inr: float) -> dict:
    return {
        'day': day,
        'INR_history': [inr],
        'dose_history': [5.0],
        'duration_history': [7],
    }


def _maintenance_days(n: int) -> list[int]:
    """Maintenance decisions land on day 6 and every 7 days thereafter."""
    return [6 + 7 * k for k in range(n)]


def test_delta_inr_is_zero_on_each_patients_first_decision():
    """Every patient's first maintenance decision must see dINR = 0."""
    tree = _RecordingTree()
    protocol = DistilledTree2D(tree)

    # Three patients in a row, with deliberately different INR levels so that a
    # leak across the boundary would produce a large, obvious non-zero dINR.
    for first_inr in (1.4, 3.9, 2.2):
        for day, inr in zip(_maintenance_days(3),
                            (first_inr, first_inr + 0.3, first_inr + 0.6)):
            protocol.prescribe(_patient(day, inr))

    firsts = tree.seen[0::3]
    assert [d for _, d in firsts] == [0.0, 0.0, 0.0], (
        f"first decision of each patient must have dINR = 0, got {firsts}")


def test_delta_inr_is_the_within_patient_difference():
    """Within a patient, dINR is the change since that patient's last decision."""
    tree = _RecordingTree()
    protocol = DistilledTree2D(tree)

    for day, inr in zip(_maintenance_days(3), (2.0, 2.5, 2.3)):
        protocol.prescribe(_patient(day, inr))

    assert tree.seen[0] == (2.0, 0.0)
    assert tree.seen[1] == pytest.approx((2.5, 0.5))
    assert tree.seen[2] == pytest.approx((2.3, -0.2))


def test_no_leak_without_any_call_to_reset():
    """The boundary is detected from the patient clock, not from reset().

    This is the property that actually failed in production: nothing in the
    evaluation path called `reset()` between patients.
    """
    tree = _RecordingTree()
    protocol = DistilledTree2D(tree)

    for day, inr in zip(_maintenance_days(2), (1.5, 1.8)):
        protocol.prescribe(_patient(day, inr))
    # Next patient — clock restarts, and reset() is deliberately NOT called.
    for day, inr in zip(_maintenance_days(2), (4.0, 3.6)):
        protocol.prescribe(_patient(day, inr))

    _, d_first_of_second_patient = tree.seen[2]
    assert d_first_of_second_patient == 0.0, (
        "dINR leaked across the patient boundary: got "
        f"{d_first_of_second_patient}, which is 4.0 - 1.8 from the previous "
        "patient rather than 0")


def test_reset_still_clears_state():
    """`reset()` remains a valid way to clear state; it is just not the only one."""
    tree = _RecordingTree()
    protocol = DistilledTree2D(tree)

    protocol.prescribe(_patient(6, 2.0))
    protocol.prescribe(_patient(13, 2.6))
    protocol.reset()
    protocol.prescribe(_patient(6, 3.1))

    assert tree.seen[-1] == (3.1, 0.0)
