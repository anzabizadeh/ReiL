"""Regression tests for the Paper-3 additive dose-duration reward.

Ground truth (plan 200 §3): r = lambda * tau - c * sum_t (mu_t - 2.5)^2, where
the square-distance cost is exactly `sq_dist` (NormalizedSquareDistance,
multiplier=-1, band_width=1 => c=4) and `+ lambda * tau` is a flat interval
bonus with tau = length of the per-decision daily-INR window.
"""
import types

import pytest

from reil.utils.reil_functions import (
    DurationIncentiveSquareDistance,
    NormalizedSquareDistance,
)

BASE_KW = dict(
    y_var_name="daily_INR_history", length=-1, multiplier=-1.0,
    interpolate=False, center=2.5, band_width=1.0, exclude_first=False,
)


def _args(inr):
    # Reward fns only touch args.value[y_var_name]; a stub is enough.
    return types.SimpleNamespace(value={"daily_INR_history": list(inr)})


def _sq_dist():
    return NormalizedSquareDistance(name="sq_dist", **BASE_KW)


def _dur(lam):
    return DurationIncentiveSquareDistance(
        name=f"sq_dist_dur_{lam}", duration_coef=lam, **BASE_KW)


WINDOWS = [[2.5], [1.5, 2.0, 2.5], [3.0, 4.0, 1.0, 2.5], [2.0] * 7]


@pytest.mark.parametrize("inr", WINDOWS)
def test_lambda_zero_equals_sq_dist(inr):
    a = _args(inr)
    assert _dur(0.0)(a) == pytest.approx(_sq_dist()(a))


@pytest.mark.parametrize("inr", WINDOWS)
@pytest.mark.parametrize("lam", [0.15, 0.25, 0.5, 1.0, 1.5])
def test_additive_identity(inr, lam):
    a = _args(inr)
    expected = _sq_dist()(a) + lam * len(inr)
    assert _dur(lam)(a) == pytest.approx(expected)


def test_hand_computed_case():
    # One day at INR 1.5: cost = -4 * (2.5 - 1.5)^2 = -4.0; +lambda*tau (tau=1).
    a = _args([1.5])
    assert _sq_dist()(a) == pytest.approx(-4.0)
    assert _dur(0.25)(a) == pytest.approx(-4.0 + 0.25 * 1)


def test_registered_in_warfarin_reward_definitions():
    from reil.healthcare.subjects.warfarin import reward_definitions
    for tag in ("0p00", "0p15", "0p25", "0p50", "1p00", "1p50"):
        assert f"sq_dist_dur_l{tag}" in reward_definitions
    fn, state_comp = reward_definitions["sq_dist_dur_l0p25"]
    assert isinstance(fn, DurationIncentiveSquareDistance)
    assert fn.duration_coef == 0.25
    assert state_comp == "recent_daily_INR"
