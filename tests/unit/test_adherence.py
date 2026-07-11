"""Unit + integration tests for the adherence-event model.

Covers the `AdherenceModel` transform in isolation (marginal miss rate,
overuse, holiday clustering, reproducibility, phenotype spread, validation)
and its integration into `DosingSubject._take_effect` (default-off parity,
prescribed-dose invariance / hidden-from-agent, and that drift actually
perturbs the INR trajectory).

See `Dissertation papers/210_paper3_adherence_model_proposal.md`.
"""
from __future__ import annotations

import unittest

import numpy as np

from reil.healthcare.adherence import AdherenceModel
from reil.healthcare.mathematical_models import HambergPKPD
from reil.healthcare.patient_warfarin_ravvaz import PatientWarfarinRavvaz
from reil.healthcare.subjects import Warfarin


def _miss_frac(administered: dict[int, float]) -> float:
    return sum(1 for v in administered.values() if v == 0.0) / len(administered)


def _holiday_runs(administered: dict[int, float]) -> list[int]:
    seq = [1 if administered[d] == 0.0 else 0 for d in sorted(administered)]
    runs: list[int] = []
    c = 0
    for x in seq:
        if x:
            c += 1
        elif c:
            runs.append(c)
            c = 0
    if c:
        runs.append(c)
    return runs


class TestAdherenceModel(unittest.TestCase):
    """The transform in isolation."""

    def test_off_is_passthrough(self) -> None:
        self.assertIsNone(AdherenceModel.from_args({'adherence_mode': 'off'}))
        off = AdherenceModel(mode='off')
        self.assertFalse(off.enabled)
        presc = {d: 5.0 for d in range(100)}
        self.assertEqual(off.administer(presc), presc)

    def test_iid_marginal_miss_rate(self) -> None:
        m = AdherenceModel(mode='iid', adherence_mean=0.8, adherence_sd=0.0)
        m.new_patient(42)
        self.assertAlmostEqual(m.adherence, 0.8, places=6)
        adm = m.administer({d: 5.0 for d in range(30000)})
        self.assertAlmostEqual(_miss_frac(adm), 0.20, delta=0.02)

    def test_mean_administered_ratio_matches_adherence(self) -> None:
        # With no overuse, mean administered/prescribed ≈ a_i.
        m = AdherenceModel(mode='iid', adherence_mean=0.7, adherence_sd=0.0,
                           overuse_prob=0.0)
        m.new_patient(1)
        presc = {d: 5.0 for d in range(30000)}
        adm = m.administer(presc)
        ratio = sum(adm.values()) / sum(presc.values())
        self.assertAlmostEqual(ratio, 0.7, delta=0.02)

    def test_overuse_only_on_taken_days(self) -> None:
        m = AdherenceModel(mode='iid', adherence_mean=1.0, adherence_sd=0.0,
                           overuse_prob=0.1, overuse_factor=2.0)
        m.new_patient(5)
        adm = m.administer({d: 5.0 for d in range(30000)})
        # a_i = 1.0 -> never missed; ~10% are double doses, rest single.
        self.assertEqual(_miss_frac(adm), 0.0)
        over = sum(1 for v in adm.values() if v == 10.0) / len(adm)
        self.assertAlmostEqual(over, 0.10, delta=0.02)

    def test_markov_marginal_and_clustering(self) -> None:
        m = AdherenceModel(mode='markov', adherence_mean=0.8, adherence_sd=0.0,
                           holiday_persistence=4.0, overuse_prob=0.0)
        m.new_patient(7)
        adm = m.administer({d: 5.0 for d in range(100000)})
        # Marginal miss rate preserved ...
        self.assertAlmostEqual(_miss_frac(adm), 0.20, delta=0.02)
        # ... but misses cluster into holidays of mean length ≈ persistence.
        self.assertAlmostEqual(float(np.mean(_holiday_runs(adm))), 4.0, delta=0.5)

    def test_markov_clusters_more_than_iid(self) -> None:
        presc = {d: 5.0 for d in range(60000)}
        iid = AdherenceModel(mode='iid', adherence_mean=0.8, adherence_sd=0.0,
                             overuse_prob=0.0)
        iid.new_patient(3)
        mk = AdherenceModel(mode='markov', adherence_mean=0.8, adherence_sd=0.0,
                            holiday_persistence=5.0, overuse_prob=0.0)
        mk.new_patient(3)
        self.assertGreater(np.mean(_holiday_runs(mk.administer(presc))),
                           np.mean(_holiday_runs(iid.administer(presc))))

    def test_reproducible_by_seed(self) -> None:
        presc = {d: 5.0 for d in range(500)}
        a = AdherenceModel(mode='markov')
        a.new_patient(123)
        b = AdherenceModel(mode='markov')
        b.new_patient(123)
        self.assertEqual(a.administer(presc), b.administer(presc))
        self.assertAlmostEqual(a.adherence, b.adherence, places=9)
        # Different seed -> (almost surely) different phenotype.
        c = AdherenceModel(mode='markov')
        c.new_patient(999)
        self.assertNotAlmostEqual(a.adherence, c.adherence, places=6)

    def test_phenotype_spread_below_pdc_threshold(self) -> None:
        # Default (mean 0.80, sd 0.21) should put a large minority/half of
        # patients below the 80% PDC threshold (IN-RANGE / Salmasi).
        m = AdherenceModel(mode='iid')
        below = 0
        n = 4000
        for seed in range(n):
            m.new_patient(seed)
            below += m.adherence < 0.80
        frac = below / n
        self.assertGreater(frac, 0.30)
        self.assertLess(frac, 0.70)

    def test_administer_before_new_patient_raises(self) -> None:
        m = AdherenceModel(mode='iid')
        with self.assertRaises(RuntimeError):
            m.administer({0: 5.0})

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            AdherenceModel(mode='sometimes')
        with self.assertRaises(ValueError):
            AdherenceModel(mode='iid', adherence_mean=1.5)
        with self.assertRaises(ValueError):
            # sd too large for a valid Beta at this mean
            AdherenceModel(mode='iid', adherence_mean=0.8, adherence_sd=0.5)
        with self.assertRaises(ValueError):
            AdherenceModel(mode='markov', holiday_persistence=0.5)

    def test_from_args_roundtrip(self) -> None:
        m = AdherenceModel.from_args({
            'adherence_mode': 'markov', 'adherence_mean': 0.75,
            'adherence_sd': 0.15, 'adherence_overuse_prob': 0.05,
            'adherence_holiday_persistence': 6.0})
        assert m is not None
        self.assertEqual(m.mode, 'markov')
        self.assertEqual(m._adherence_mean, 0.75)
        self.assertEqual(m._holiday_persistence, 6.0)


def _make_subject(seed: int, adherence: AdherenceModel | None) -> Warfarin:
    # Absolute-dose mode so a fixed schedule actually administers drug (a
    # percent-change policy starting from 0 mg would stay at 0 forever).
    return Warfarin(
        patient=PatientWarfarinRavvaz(model=HambergPKPD(cache_size=100),
                                      random_seed=seed),
        adherence=adherence,
        decision_mode='dose',
        decision_values=(0.0, 15.0),
        decision_range=(0.0, 15.0),
        dose_range=(0.0, 15.0),
        dose_step=0.5,
        round_to_step=False,
        default_duration=7,
        max_day=90,
    )


def _constant_dose_action(subject: Warfarin, dose: float):
    """Fetch the FeatureSet action whose value is {'dose': dose}."""
    gen = subject.possible_actions('daily_15')
    next(gen)
    for action in gen.send('return feature exclusive'):
        if abs(float(action.value['dose']) - dose) < 1e-9:
            return action
    raise AssertionError(f'no action for dose={dose}')


def _run_constant(subject: Warfarin, dose: float = 5.0
                  ) -> tuple[list[float], list[float]]:
    """Drive a constant `dose` mg/day schedule to termination; return
    (prescribed dose history, INR history)."""
    subject.reset()
    action = _constant_dose_action(subject, dose)
    while subject._day < subject._max_day:
        subject._take_effect(action)
    return list(subject._full_dose_history), list(subject._full_measurement_history)


class TestAdherenceIntegration(unittest.TestCase):
    """The transform inside DosingSubject._take_effect."""

    def test_off_model_identical_to_none(self) -> None:
        # An explicit off-model must be byte-identical to no adherence at all.
        dose_none, inr_none = _run_constant(_make_subject(0, None))
        dose_off, inr_off = _run_constant(_make_subject(0, AdherenceModel(mode='off')))
        self.assertEqual(sum(dose_none), 5.0 * 90)     # drug really administered
        self.assertEqual(dose_none, dose_off)
        self.assertEqual(inr_none, inr_off)

    def test_prescribed_dose_hidden_but_inr_drifts(self) -> None:
        # Both runs prescribe the SAME constant schedule; adherence must leave
        # the prescribed history untouched (hidden from the agent) yet change
        # the realised INR trajectory (autocorrelated drift from missed doses).
        dose_off, inr_off = _run_constant(_make_subject(0, None))
        heavy = AdherenceModel(mode='markov', adherence_mean=0.5,
                               adherence_sd=0.0, holiday_persistence=5.0,
                               overuse_prob=0.0)
        dose_on, inr_on = _run_constant(_make_subject(0, heavy))
        # Prescribed dose history the agent sees is unchanged.
        self.assertEqual(dose_off, dose_on)
        # But the INR trajectory differs (missed doses -> lower exposure).
        self.assertTrue(any(abs(a - b) > 1e-6 for a, b in zip(inr_off, inr_on)))
        # Under heavy non-adherence the patient spends less time anticoagulated,
        # so mean INR is lower than the fully-adherent case.
        self.assertLess(float(np.mean(inr_on)), float(np.mean(inr_off)))


if __name__ == '__main__':
    unittest.main()
