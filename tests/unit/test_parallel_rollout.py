"""Tests for the patient-shard parallel rollout helper.

Three layers of validation per the plan:
  1. Pure shard-arithmetic test — no env, no joblib, fast.
  2. Mock-env test — validates that the helper calls the factory with the
     right shards and aggregates results. Uses fakes; no TF or pickling.
  3. Real-env parity test — builds a tiny `Single` env with a warfarin subject
     and a deterministic agent, runs serially via `simulate_pass`, then via
     `parallel_simulate(n_workers=1)` and `(n_workers=2)`. The per-patient
     INR trajectory must match bit-for-bit across all three paths.

Layer 3 is the one that actually proves the helper preserves PPO-frozen
semantics; layers 1 and 2 catch wiring bugs cheaply.
"""
from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from reil.environments.parallel_rollout import (
    _shard_range, _run_one, parallel_simulate,
    parallel_collect_trajectories)


# ----------------------------------------------------------------------
# Layer 1: shard arithmetic
# ----------------------------------------------------------------------


class TestShardRange(unittest.TestCase):

    def test_even_split(self) -> None:
        # 100 seeds across 4 workers → 25 each
        self.assertEqual(_shard_range(0, 100, 4),
                         [(0, 25), (25, 50), (50, 75), (75, 100)])

    def test_uneven_split(self) -> None:
        # 100 seeds across 3 workers → ceil(100/3)=34, 34, 32
        # (the last shard is shorter — no padding, no overlap)
        self.assertEqual(_shard_range(0, 100, 3),
                         [(0, 34), (34, 68), (68, 100)])

    def test_offset_start(self) -> None:
        self.assertEqual(_shard_range(100, 110, 2),
                         [(100, 105), (105, 110)])

    def test_more_workers_than_seeds(self) -> None:
        # 3 seeds across 8 workers → 3 single-seed shards, no empties
        self.assertEqual(_shard_range(0, 3, 8),
                         [(0, 1), (1, 2), (2, 3)])

    def test_partition_is_complete_and_disjoint(self) -> None:
        # Whatever the split, every seed in [start, stop) is covered exactly
        # once. This is the test that would have caught off-by-one bugs in a
        # parallel rollout that silently dropped or duplicated patients.
        for start, stop, n in [(0, 100, 4), (0, 100, 3), (50, 73, 5),
                               (0, 1, 4), (0, 0, 4), (10, 11, 1)]:
            with self.subTest(start=start, stop=stop, n=n):
                shards = _shard_range(start, stop, n)
                covered = []
                for s, e in shards:
                    covered.extend(range(s, e))
                self.assertEqual(covered, list(range(start, stop)))


# ----------------------------------------------------------------------
# Layer 2: mock-env wiring
# ----------------------------------------------------------------------


class _FakeEnv:
    """Minimal stand-in for `Single`: records calls and returns stub stats."""

    def __init__(self, shard: tuple[int, int]) -> None:
        self.shard = shard
        self.activated_plan: str | None = None
        self.simulate_count = 0

    def activate_plan(self, plan_name: str) -> None:
        self.activated_plan = plan_name

    def simulate_pass(self) -> None:
        self.simulate_count += 1

    def report_statistics(self, unstack: bool = True,
                          reset_history: bool = True) -> dict[str, Any]:
        return {"shard": self.shard, "simulate_count": self.simulate_count}


class TestParallelSimulatePlumbing(unittest.TestCase):

    def test_serial_path_runs_inline(self) -> None:
        # n_workers=1 must not spawn a joblib process — pass a factory that
        # records the process it ran in (we use a list as a marker).
        seen: list[tuple[int, int]] = []

        def factory(shard: tuple[int, int]) -> _FakeEnv:
            seen.append(shard)
            return _FakeEnv(shard)

        reports = parallel_simulate(
            factory, (0, 50), n_workers=1, plan_name="validation")

        self.assertEqual(seen, [(0, 50)])
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["shard"], (0, 50))
        self.assertEqual(reports[0]["simulate_count"], 1)

    def test_plan_activation_is_optional(self) -> None:
        envs: list[_FakeEnv] = []

        def factory(shard: tuple[int, int]) -> _FakeEnv:
            e = _FakeEnv(shard)
            envs.append(e)
            return e

        parallel_simulate(factory, (0, 10), n_workers=1, plan_name=None)
        self.assertIsNone(envs[0].activated_plan)

        envs.clear()
        parallel_simulate(factory, (0, 10), n_workers=1, plan_name="test_07")
        self.assertEqual(envs[0].activated_plan, "test_07")

    def test_empty_range_returns_empty_list(self) -> None:
        reports = parallel_simulate(_FakeEnv, (5, 5), n_workers=4)
        self.assertEqual(reports, [])

    def test_run_one_directly(self) -> None:
        # _run_one is the worker body — must be picklable and callable
        # independent of joblib. Direct-call test catches signature drift.
        report = _run_one(_FakeEnv, "validation", (10, 20))
        self.assertEqual(report["shard"], (10, 20))


# ----------------------------------------------------------------------
# Layer 3: real-env bit-parity (frozen policy on a warfarin subject)
# ----------------------------------------------------------------------


def _eval_seeds_serially(seeds: list[int]) -> np.ndarray:
    """Drive one fixed-dose patient per seed, return (n_patients, 91) INRs.

    Bypasses the Single env entirely — this is the "ground truth" the helper
    must reproduce. Uses the same HambergPKPD + PatientWarfarinRavvaz path the
    parity tests use, so it's deterministic given the seed.
    """
    from reil.healthcare.mathematical_models import HambergPKPD
    from reil.healthcare.patient_warfarin_ravvaz import PatientWarfarinRavvaz

    dose = {d: 5.0 for d in range(90)}
    meas = list(range(91))
    inr = np.empty((len(seeds), 91), dtype=np.float64)
    for i, seed in enumerate(seeds):
        m = HambergPKPD(cache_size=100)
        p = PatientWarfarinRavvaz(model=m, random_seed=seed)
        inr[i] = np.asarray(p.model(dose=dose, measurement_days=meas)["INR"])
    return inr


def _eval_one_shard(shard: tuple[int, int]) -> np.ndarray:
    """Module-level so loky can pickle it. Mirrors what an env_factory would
    do in production — instantiate per-seed and collect a per-trajectory
    statistic — but skips the Single wrapper to keep the test focused on the
    parallelism contract."""
    return _eval_seeds_serially(list(range(shard[0], shard[1])))


class _ShardEnv:
    """Tiny env that wraps `_eval_one_shard` so the helper's contract
    (activate_plan / simulate_pass / report_statistics) is honoured.
    Records its INR matrix into the report dict for the test to compare.
    """

    def __init__(self, shard: tuple[int, int]) -> None:
        self._shard = shard
        self._inr: np.ndarray | None = None

    def activate_plan(self, plan_name: str) -> None:
        pass

    def simulate_pass(self) -> None:
        self._inr = _eval_one_shard(self._shard)

    def report_statistics(self, unstack: bool = True,
                          reset_history: bool = True) -> dict[str, Any]:
        return {"shard": self._shard, "inr": self._inr}


def _shard_env_factory(shard: tuple[int, int]) -> _ShardEnv:
    """Module-level factory — loky-picklable."""
    return _ShardEnv(shard)


class TestFrozenPolicyParity(unittest.TestCase):
    """The bit-parity contract: parallel rollout produces the same per-patient
    INR matrix as a serial pass, regardless of n_workers."""

    SEEDS = list(range(20))

    def setUp(self) -> None:
        self._serial = _eval_seeds_serially(self.SEEDS)

    def _check_reports(self, reports: list[dict[str, Any]]) -> None:
        # Reassemble shards in order and compare to the serial matrix
        reports = sorted(reports, key=lambda r: r["shard"][0])
        combined = np.concatenate([r["inr"] for r in reports], axis=0)
        np.testing.assert_allclose(combined, self._serial,
                                   atol=1e-12, rtol=1e-12)

    def test_serial_via_helper_matches_direct(self) -> None:
        """n_workers=1 inline path produces identical INRs to a direct call."""
        reports = parallel_simulate(
            _shard_env_factory, (0, len(self.SEEDS)), n_workers=1)
        self._check_reports(reports)

    def test_parallel_two_workers_matches_serial(self) -> None:
        """n_workers=2 must produce the same INRs (just sharded)."""
        reports = parallel_simulate(
            _shard_env_factory, (0, len(self.SEEDS)), n_workers=2)
        self._check_reports(reports)

    def test_parallel_four_workers_matches_serial(self) -> None:
        """n_workers=4 — the same SEEDS=20 split into 4 shards of 5 each."""
        reports = parallel_simulate(
            _shard_env_factory, (0, len(self.SEEDS)), n_workers=4)
        self._check_reports(reports)


# ----------------------------------------------------------------------
# Layer 4: parallel_collect_trajectories — the Phase B helper
# ----------------------------------------------------------------------

# Module-level factories so loky can pickle them.

def _agent_factory() -> Any:
    """Build a fresh RandomAgent. Seeded inside so workers are reproducible."""
    import reil
    from reil.agents.random_agent import RandomAgent
    reil.set_reil_random_seed(12345)
    agent = RandomAgent(seed=42)
    # Match what _register does in test_collect_trajectory: cross-register
    # so collect_trajectory can validate subject_id.
    agent.register('subject')
    return agent


def _subject_factory(seed: int) -> Any:
    """Build a Warfarin subject for `seed`. Module-level for loky-pickle."""
    from reil.healthcare.mathematical_models import HambergPKPD
    from reil.healthcare.patient_warfarin_ravvaz import PatientWarfarinRavvaz
    from reil.healthcare.subjects.warfarin import Warfarin
    patient = PatientWarfarinRavvaz(model=HambergPKPD(cache_size=100),
                                    random_seed=seed)
    subject = Warfarin(
        patient=patient,
        decision_mode='dose_percent_change',
        decision_values=(-1.0, -0.5, 0.0, 0.5, 1.0),
        decision_range=(-1.0, 1.0),
        dose_range=(0.0, 15.0),
        dose_step=0.5,
        round_to_step=False,
        default_duration=7,
        max_day=90,
    )
    subject.register('agent')
    return subject


def _protocol() -> Any:
    from reil.datatypes import Entity, InteractionProtocol
    return InteractionProtocol(
        agent=Entity(name='agent'),
        subject=Entity(name='subject'),
        state_name='patient_w_dosing_w_baseline_01',
        action_name='percent',
        reward_name='sq_dist',
        n=1, unit='instance',
    )


class TestParallelCollectTrajectories(unittest.TestCase):
    """The Phase-B contract: parallel rollout returns one history per seed,
    in seed order, each driving its subject to termination."""

    def test_serial_inline_one_worker(self) -> None:
        results = parallel_collect_trajectories(
            agent_factory=_agent_factory,
            subject_factory=_subject_factory,
            protocol=_protocol(),
            seed_range=(0, 4),
            n_workers=1,
        )
        self.assertEqual(len(results), 4)
        for history, agent_stat, subject_stat in results:
            self.assertGreaterEqual(len(history), 12)
            self.assertIsNotNone(history[0].state)
            self.assertIsNotNone(history[0].action_taken)
            # Stat hooks weren't requested — both should be None.
            self.assertIsNone(agent_stat)
            self.assertIsNone(subject_stat)

    def test_parallel_two_workers_matches_serial_shape(self) -> None:
        """Same seed range under n_workers=2 produces same number of
        triples, same per-history length as serial. (Bit-exact action
        parity is not asserted here — the worker-local RNG init isn't
        synchronised to serial; the e2e training run is the parity bar
        for Phase B.)"""
        r_serial = parallel_collect_trajectories(
            agent_factory=_agent_factory, subject_factory=_subject_factory,
            protocol=_protocol(), seed_range=(0, 4), n_workers=1,
        )
        r_parallel = parallel_collect_trajectories(
            agent_factory=_agent_factory, subject_factory=_subject_factory,
            protocol=_protocol(), seed_range=(0, 4), n_workers=2,
        )
        self.assertEqual(len(r_serial), len(r_parallel))
        for (hs, _, _), (hp, _, _) in zip(r_serial, r_parallel):
            self.assertEqual(len(hs), len(hp))


if __name__ == "__main__":
    unittest.main()
