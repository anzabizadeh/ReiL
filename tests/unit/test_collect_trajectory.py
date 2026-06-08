"""Tests for `BaseAgent.collect_trajectory` — the flat per-patient rollout.

These cover the building block for Phase B (parallel training rollouts in
synchronous PPO). The contract:

  - Drives the subject to termination using current weights via `act()`.
  - Returns a `History` of `Observation`s mirroring what the generator-based
    `observe()` would have accumulated, for the simple (no-lookahead,
    single-agent) case.
  - Does NOT call `learn()` — the caller orchestrates that.

End-to-end parity vs. the `observe()`+`Single.interact` path is covered
implicitly: a downstream trainer test will roll out one chunk both ways and
verify equivalent agent updates. Here we focus on the unit-level invariants
that catch silent shape/order bugs.
"""
from __future__ import annotations

import unittest

import reil
from reil.agents.random_agent import RandomAgent
from reil.datatypes import Entity, InteractionProtocol
from reil.healthcare.mathematical_models import HambergPKPD
from reil.healthcare.patient_warfarin_ravvaz import PatientWarfarinRavvaz
from reil.healthcare.subjects.warfarin import Warfarin


def _make_warfarin_subject(seed: int) -> Warfarin:
    """Construct a Warfarin subject identical to what WarfarinGenerator
    produces in warfarin_dosing/wd_runner/trainer/setup.py for one seed."""
    patient = PatientWarfarinRavvaz(model=HambergPKPD(cache_size=100),
                                    random_seed=seed)
    return Warfarin(
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


def _make_protocol() -> InteractionProtocol:
    return InteractionProtocol(
        agent=Entity(name='agent'),
        subject=Entity(name='subject'),
        state_name='patient_w_dosing_w_baseline_01',
        action_name='percent',
        reward_name='sq_dist',
        n=1,
        unit='instance',
    )


def _register(subject: Warfarin, agent: RandomAgent) -> tuple[int, int]:
    """Mirror what Environment does: cross-register agent and subject so they
    each know the other's id. Returns (agent_id, subject_id)."""
    subject_id = agent.register('subject')
    agent_id = subject.register('agent')
    return agent_id, subject_id


class TestCollectTrajectory(unittest.TestCase):

    def setUp(self) -> None:
        # RandomAgent's `actions.send('choose feature exclusive')` consults
        # the global reil RNG; seed it so tests are reproducible regardless
        # of order.
        reil.set_reil_random_seed(12345)

    def test_drives_to_termination_with_full_observations(self) -> None:
        subject = _make_warfarin_subject(seed=0)
        agent = RandomAgent(seed=42)
        agent_id, subject_id = _register(subject, agent)

        history = agent.collect_trajectory(
            subject, agent_id, subject_id, _make_protocol(), iteration=0)

        # max_day=90, default_duration=7 → 13 decision points, give or take 1
        self.assertGreaterEqual(len(history), 12)
        self.assertLessEqual(len(history), 14)

        # Subject must end terminated
        self.assertTrue(subject.is_terminated(None))

        # Every observation: state + action + action_taken populated. Reward
        # may be None only on the last observation (terminal step).
        for i, obs in enumerate(history):
            with self.subTest(i=i):
                self.assertIsNotNone(obs.state)
                self.assertIsNotNone(obs.action)
                self.assertIsNotNone(obs.action_taken)
                if i < len(history) - 1:
                    self.assertIsNotNone(obs.reward)

    def test_repeated_calls_have_consistent_shape(self) -> None:
        """Two runs with same seeds produce same-length histories with the
        same per-step shape. (Bit-exact action equality requires seeding
        more RNG sources than this test controls — covered in the e2e
        validation against the EXP-C2-001 reference run.)"""
        def run_once() -> list:
            reil.set_reil_random_seed(12345)
            subject = _make_warfarin_subject(seed=7)
            agent = RandomAgent(seed=42)
            agent_id, subject_id = _register(subject, agent)
            return agent.collect_trajectory(
                subject, agent_id, subject_id, _make_protocol())

        h1 = run_once()
        h2 = run_once()

        self.assertEqual(len(h1), len(h2))
        for i, (a, b) in enumerate(zip(h1, h2)):
            with self.subTest(step=i):
                self.assertIsNotNone(a.state)
                self.assertIsNotNone(b.state)
                self.assertIsNotNone(a.action_taken)
                self.assertIsNotNone(b.action_taken)

    def test_different_seeds_produce_different_trajectories(self) -> None:
        """Sanity: seed actually matters (catches a stubbed-out RNG)."""
        def run(seed: int) -> list:
            reil.set_reil_random_seed(12345)
            subject = _make_warfarin_subject(seed=seed)
            agent = RandomAgent(seed=42)
            agent_id, subject_id = _register(subject, agent)
            return agent.collect_trajectory(
                subject, agent_id, subject_id, _make_protocol())

        h1 = run(seed=0)
        h2 = run(seed=1)

        # At least one action_taken should differ between different patient
        # seeds (same agent RNG, but different patient PKPD response leads
        # the agent to different doses via the state-dependent action gen).
        actions1 = [tuple(o.action_taken.index.values()) for o in h1]
        actions2 = [tuple(o.action_taken.index.values()) for o in h2]
        self.assertNotEqual(actions1, actions2,
                            'expected per-seed trajectories to diverge')


if __name__ == '__main__':
    unittest.main()
