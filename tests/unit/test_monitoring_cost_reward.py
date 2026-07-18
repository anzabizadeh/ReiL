"""Parity + behaviour tests for the per-visit monitoring cost (kappa).

Paper-3 EB (doc 220 §9): a flat per-DECISION monitoring cost `monitoring_coef`
(kappa) is subtracted once per reward call on the base `NormalizedSquareDistance`
and is therefore inherited by `DurationIncentiveSquareDistance` and
`DoseDurationSafetyReward` (both include_control paths). Requirements:

  * PARITY: default kappa=0.0 is byte-identical to the pre-kappa reward.
  * BEHAVIOUR: kappa>0 subtracts EXACTLY one kappa per decision, through the
    whole inheritance chain, once per visit (not per day), for every variant.
"""
import types

import pytest

from reil.utils.reil_functions import (
    DoseDurationSafetyReward,
    DurationIncentiveSquareDistance,
    NormalizedSquareDistance,
)

BASE_KW = dict(
    y_var_name="daily_INR_history", length=-1, multiplier=-1.0,
    interpolate=False, center=2.5, band_width=1.0, exclude_first=False,
)

WINDOWS = [[2.5], [1.5, 2.0, 2.5], [3.0, 4.0, 1.0, 2.5], [2.0] * 7]
KAPPAS = [0.1, 0.5, 1.0, 2.5]


def _args(inr):
    # Reward fns only touch args.value[y_var_name]; a stub is enough.
    return types.SimpleNamespace(value={"daily_INR_history": list(inr)})


def _dds(**over):
    kw = dict(name="dds", duration_coef=1.5, safety_coef=0.5,
              penalty_shape="huber", **BASE_KW)
    kw.update(over)
    return DoseDurationSafetyReward(**kw)


# --------------------------------------------------------------------------
# PARITY: default kappa=0.0 must not change anything.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("inr", WINDOWS)
def test_parity_sq_dist(inr):
    a = _args(inr)
    implicit = NormalizedSquareDistance(name="sq", **BASE_KW)(a)
    explicit0 = NormalizedSquareDistance(
        name="sq", monitoring_coef=0.0, **BASE_KW)(a)
    assert implicit == pytest.approx(explicit0)


def test_parity_hand_computed():
    # One day at INR 1.5: cost = -4 * (2.5 - 1.5)^2 = -4.0, unchanged by kappa=0.
    a = _args([1.5])
    assert NormalizedSquareDistance(name="sq", **BASE_KW)(a) == pytest.approx(-4.0)


@pytest.mark.parametrize("inr", WINDOWS)
def test_parity_duration(inr):
    a = _args(inr)
    implicit = DurationIncentiveSquareDistance(
        name="dur", duration_coef=1.5, **BASE_KW)(a)
    explicit0 = DurationIncentiveSquareDistance(
        name="dur", duration_coef=1.5, monitoring_coef=0.0, **BASE_KW)(a)
    assert implicit == pytest.approx(explicit0)


@pytest.mark.parametrize("inr", WINDOWS)
@pytest.mark.parametrize("include_control", [True, False])
def test_parity_safety_reward(inr, include_control):
    a = _args(inr)
    implicit = _dds(include_control=include_control)(a)
    explicit0 = _dds(include_control=include_control, monitoring_coef=0.0)(a)
    assert implicit == pytest.approx(explicit0)


# --------------------------------------------------------------------------
# BEHAVIOUR: kappa > 0 subtracts exactly one kappa per decision.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("inr", WINDOWS)
@pytest.mark.parametrize("kappa", KAPPAS)
def test_sq_dist_subtracts_one_kappa(inr, kappa):
    a = _args(inr)
    old = NormalizedSquareDistance(name="sq", **BASE_KW)(a)
    new = NormalizedSquareDistance(
        name="sq", monitoring_coef=kappa, **BASE_KW)(a)
    assert new == pytest.approx(old - kappa)


@pytest.mark.parametrize("inr", WINDOWS)
@pytest.mark.parametrize("kappa", KAPPAS)
def test_duration_subtracts_one_kappa(inr, kappa):
    a = _args(inr)
    old = DurationIncentiveSquareDistance(
        name="dur", duration_coef=1.5, **BASE_KW)(a)
    new = DurationIncentiveSquareDistance(
        name="dur", duration_coef=1.5, monitoring_coef=kappa, **BASE_KW)(a)
    assert new == pytest.approx(old - kappa)


@pytest.mark.parametrize("inr", WINDOWS)
@pytest.mark.parametrize("kappa", KAPPAS)
@pytest.mark.parametrize("include_control", [True, False])
def test_safety_reward_subtracts_one_kappa(inr, kappa, include_control):
    # include_control=False bypasses super().__call__; kappa must still apply
    # exactly once via the patched branch.
    a = _args(inr)
    old = _dds(include_control=include_control)(a)
    new = _dds(include_control=include_control, monitoring_coef=kappa)(a)
    assert new == pytest.approx(old - kappa)


def test_kappa_is_per_visit_not_per_day():
    # Same kappa for tau=1 and tau=7 (cost is per blood draw, not per day).
    k = 0.5
    base = NormalizedSquareDistance(name="sq", **BASE_KW)
    withk = NormalizedSquareDistance(name="sq", monitoring_coef=k, **BASE_KW)
    for inr in ([2.5], [2.5] * 7):
        a = _args(inr)
        assert base(a) - withk(a) == pytest.approx(k)


@pytest.mark.parametrize("inr", WINDOWS)
def test_kappa_with_average_dose_reward(inr):
    # sq_dist_avg (dose head, average=True): kappa still subtracts once.
    k = 0.5
    a = _args(inr)
    old = NormalizedSquareDistance(name="avg", average=True, **BASE_KW)(a)
    new = NormalizedSquareDistance(
        name="avg", average=True, monitoring_coef=k, **BASE_KW)(a)
    assert new == pytest.approx(old - k)


# --------------------------------------------------------------------------
# REGISTRATION: the sq_dist_kap* Pareto family is wired into warfarin.py.
# --------------------------------------------------------------------------

EXPECTED_KAP = {
    "0p00": 0.0, "0p50": 0.5, "1p00": 1.0,
    "2p00": 2.0, "4p00": 4.0, "8p00": 8.0,
}


@pytest.mark.parametrize("tag,kap", list(EXPECTED_KAP.items()))
def test_sq_dist_kap_registered(tag, kap):
    from reil.healthcare.subjects.warfarin import reward_definitions
    name = f"sq_dist_kap{tag}"
    assert name in reward_definitions
    fn, state_comp = reward_definitions[name]
    assert isinstance(fn, NormalizedSquareDistance)
    assert fn.monitoring_coef == kap
    assert fn.average is False          # EB uses SUMMED control (tau matters)
    assert state_comp == "recent_daily_INR"


def test_sq_dist_kap_zero_matches_sq_dist():
    # kappa=0 anchor must equal the plain sq_dist reward on the same window.
    from reil.healthcare.subjects.warfarin import reward_definitions
    a = _args([1.5, 3.0, 2.0])
    kap0 = reward_definitions["sq_dist_kap0p00"][0]
    sq = reward_definitions["sq_dist"][0]
    assert kap0(a) == pytest.approx(sq(a))
