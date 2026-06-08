# -*- coding: utf-8 -*-
"""Patient-shard parallel rollout helper (joblib MVP for plan #2).

Wraps `joblib.Parallel` around `Single.simulate_pass()` so that a frozen-policy
evaluation cohort (validation, final_test) can be rolled out across N worker
processes instead of serially in one. The caller provides an `env_factory` that
builds a fresh `Single` env scoped to a given seed shard; each worker calls it,
runs `simulate_pass()`, and returns the resulting statistics report.

# Scope: frozen-policy only

This helper is bit-correct **only** when the agent's `_training_trigger` is
`'none'` (i.e. weights are frozen for the duration of the pass). That matches:

  - validation passes that core.train() runs after every chunk
  - final_test passes against Hamberg 07 + 10

It does NOT cover the on-policy training rollout. PPO updates fire on every
patient termination (see `Agent.observe` at agent.py:381-385), so each patient's
`act()` depends on the gradient steps from all preceding patients in the chunk.
Parallelising that path requires accepting an on-policy semantics change
("collect K patients with frozen weights, then do all K updates serially"),
which is a math change vs. the current code — out of scope for this MVP and
documented as a follow-up in the perf overhaul plan.

# Worker contract

The factory must return a fully-built `Single` env whose subjects, agents,
demons, and plans are all initialised. Specifically the agent must already
hold the right weights (workers do NOT broadcast weights; the caller is
responsible for loading the trained checkpoint inside the factory).

The factory IS called inside the worker process, so any heavy state (TF
weights, generators, observers) is built per-worker. This costs ~5-10s of TF
import on the first call per worker but amortises across the shard.

# When this is worth using

Loky-backend startup overhead is ~5s of TF import per worker per call. For a
4-worker pool that's a ~20s tax. Rule of thumb: only parallelise if the
serial cost of the pass is comfortably above the spawn tax.

  - Validation pass (2,000 patients, frozen policy, post-#1 ~6ms/patient =
    ~12s serial): parallel break-even is right around the spawn tax;
    real-world win is small unless you keep the worker pool warm across
    multiple chunks (open one `joblib.parallel_backend('loky', n_jobs=N)`
    context around the whole chunk loop).
  - Final test (2,000 patients × 2 PKPDs = 4,000 patients, ~20s serial
    post-#1): parallel cleanly wins.
  - Long training runs where the same validation cohort is re-evaluated
    every chunk: open the worker pool once at the top of `train()` and
    reuse it for every validation + final_test call.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import joblib

from reil.environments.single import Single

EnvFactory = Callable[[tuple[int, int]], Single]
StatsReport = dict[tuple[str, str], Any]


def _shard_range(start: int, stop: int, n_workers: int) -> list[tuple[int, int]]:
    """Split [start, stop) into up to `n_workers` contiguous shards.

    Returns fewer than n_workers shards if the range is smaller than the
    worker count (no empty shards).
    """
    total = stop - start
    if total <= 0 or n_workers <= 0:
        return []
    shard_size = (total + n_workers - 1) // n_workers
    shards: list[tuple[int, int]] = []
    for i in range(n_workers):
        s = start + i * shard_size
        if s >= stop:
            break
        e = min(stop, s + shard_size)
        shards.append((s, e))
    return shards


def _run_one(env_factory: EnvFactory, plan_name: str | None,
             shard: tuple[int, int]) -> StatsReport:
    """Worker body: build a fresh env for this shard, simulate, return stats.

    Kept module-level so joblib's loky backend can pickle it cleanly.
    """
    env = env_factory(shard)
    if plan_name is not None:
        env.activate_plan(plan_name)
    env.simulate_pass()
    return env.report_statistics(unstack=True, reset_history=True)


def _collect_one_shard(
    agent_factory: Callable[[], Any],
    subject_factory: Callable[[int], Any],
    protocol: Any,
    seed_shard: tuple[int, int],
    agent_id: int,
    subject_id: int,
    agent_stat_name: str | None = None,
    subject_stat_name: str | None = None,
) -> list[tuple[Any, Any, Any]]:
    """Worker body for `parallel_collect_trajectories`. Module-level for loky.

    Builds one agent in this worker (loads frozen weights), iterates over the
    seeds in `seed_shard` building a fresh subject per seed, calls
    `agent.collect_trajectory(...)`, then replays the observer-side stat
    bookkeeping locally (mirroring what `Agent.observe`'s GeneratorExit and
    `Single.reset_subject` would have done in the serial path).

    Returns a list of `(history, agent_stat, subject_stat)` triples — one
    per seed. `agent_stat` / `subject_stat` are precomputed
    `(FeatureSet, value)` tuples ready for `Statistic.raw_append` on the
    main-process live entities; either is `None` if the corresponding
    `statistic_name` was unset in the protocol.
    """
    agent = agent_factory()
    results: list[tuple[Any, Any, Any]] = []
    for seed in range(seed_shard[0], seed_shard[1]):
        subject = subject_factory(seed)
        history = agent.collect_trajectory(
            subject, agent_id=agent_id, subject_id=subject_id,
            protocol=protocol, iteration=0)
        # `possible_actions` holds a Python generator that joblib's loky
        # backend can't pickle. Downstream `Agent._prepare_training` reads
        # state / action_taken / action / reward only, so dropping it is
        # safe and necessary for cross-process transport.
        for obs in history:
            obs.possible_actions = None

        agent_stat = None
        if agent_stat_name:
            agent.statistic.append(agent_stat_name, subject_id)
            agent_stat = agent.statistic.latest(subject_id)

        subject_stat = None
        if subject_stat_name:
            subject.statistic.append(subject_stat_name, agent_id)
            subject_stat = subject.statistic.latest(agent_id)

        results.append((history, agent_stat, subject_stat))
    return results


def parallel_collect_trajectories(
    agent_factory: Callable[[], Any],
    subject_factory: Callable[[int], Any],
    protocol: Any,
    seed_range: tuple[int, int],
    n_workers: int,
    agent_id: int = 0,
    subject_id: int = 0,
    agent_stat_name: str | None = None,
    subject_stat_name: str | None = None,
) -> list[tuple[Any, Any, Any]]:
    """Phase-B helper: collect per-patient histories in parallel.

    Synchronous parallel PPO: every worker uses a frozen weight snapshot
    (loaded inside `agent_factory()`) to roll out its shard. Main process
    then iterates the returned histories and calls `live_agent.learn(h)`
    on each one — this preserves the per-patient PPO update granularity
    while parallelising the (expensive) rollout work.

    Math change vs. fully-serial training: each chunk's rollouts use the
    chunk-start policy, not the after-each-patient updated policy. The
    PPO clip ratio + KL early-stop keep this drift small per chunk, but
    end-to-end results are NOT bit-equivalent to serial. Validate by
    comparing TensorBoard PTTR curves on a real run.

    Arguments
    ---------
    agent_factory:
        Callable that, when invoked in a worker, returns a fresh agent
        with the FROZEN policy weights loaded. Picklable (loky backend).
        Typically a closure that loads a weight snapshot file written by
        the main process at chunk start.
    subject_factory:
        Callable taking one integer seed and returning a fully-initialised
        Subject (Warfarin in the production path). Picklable.
    protocol:
        InteractionProtocol — supplies state_name, action_name, reward_name
        to `collect_trajectory`. Same protocol the main env uses for the
        training plan.
    seed_range:
        `(start, stop)` half-open range of patient seeds — typically the
        chunk's slice of the WarfarinGenerator's `stops` table.
    n_workers:
        Worker count. `n_workers <= 1` runs inline (no joblib spawn).
    agent_id, subject_id:
        Cross-registration ids. Default 0/0 works for the single-agent /
        single-subject case which is what warfarin uses.

    Returns
    -------
    Flat list of `History` objects, one per seed in `seed_range`, in seed
    order across shards (joblib preserves submission order in its result).
    """
    shards = _shard_range(seed_range[0], seed_range[1], max(1, n_workers))
    if not shards:
        return []

    if n_workers <= 1 or len(shards) == 1:
        return _collect_one_shard(
            agent_factory, subject_factory, protocol, shards[0],
            agent_id, subject_id, agent_stat_name, subject_stat_name)

    nested = joblib.Parallel(n_jobs=n_workers, backend='loky')(
        joblib.delayed(_collect_one_shard)(
            agent_factory, subject_factory, protocol, shard,
            agent_id, subject_id, agent_stat_name, subject_stat_name)
        for shard in shards
    )
    # Flatten in shard order (joblib preserves order of delayed() submissions)
    return [item for shard_results in nested for item in shard_results]


def parallel_simulate(
    env_factory: EnvFactory,
    seed_range: tuple[int, int],
    n_workers: int,
    plan_name: str | None = None,
) -> list[StatsReport]:
    """Run `env.simulate_pass()` for `seed_range` across `n_workers` processes.

    Returns a list of per-shard statistics reports. When `n_workers <= 1` the
    call is executed inline in the current process (no joblib overhead, no
    process spawn) so callers can use this as a drop-in for the serial path
    by passing `n_workers=1`.

    Arguments
    ---------
    env_factory:
        Callable taking `(shard_start, shard_stop)` and returning a fully
        initialised `Single` env. See module docstring for the contract.
    seed_range:
        `(start, stop)` — half-open range of patient seeds to cover.
    n_workers:
        Worker process count. Use 1 (or 0) for serial execution.
    plan_name:
        If set, each worker calls `env.activate_plan(plan_name)` before
        `simulate_pass`. Omit if the factory already activates the right plan.

    Notes
    -----
    Bit-correctness vs. a single serial `simulate_pass` over the same range
    holds only when:
      - the agent's `_training_trigger == 'none'`
      - the underlying patient/subject construction is seed-deterministic
        (PatientWarfarinRavvaz at `random_seed=seed` qualifies)
      - the statistics aggregator is associative across shards (mean of
        per-patient PTTRs is — it's a sum-of-sums divided by count).
    """
    shards = _shard_range(seed_range[0], seed_range[1], max(1, n_workers))
    if not shards:
        return []

    if n_workers <= 1 or len(shards) == 1:
        return [_run_one(env_factory, plan_name, shards[0])]

    return joblib.Parallel(n_jobs=n_workers, backend='loky')(
        joblib.delayed(_run_one)(env_factory, plan_name, shard)
        for shard in shards
    )
