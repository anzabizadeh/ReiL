"""Microbenchmark: HambergPKPD INR throughput.

Measures the median wall-clock time over 5 runs for two scenarios:
  - 1 patient × 90 days @ 5 mg/day
  - 500 patients × 90 days @ 5 mg/day  (proxy for a quarter of a Paper-2 chunk)

Run before AND after the perf overhaul (use `git stash` for the baseline) and
record the ratio in the commit message. This script is opt-in — not wired into
the unittest discovery — to keep CI fast.

Usage:
    poetry run python -m tests.data._bench_inr
"""
from __future__ import annotations

import statistics
import time

from reil.healthcare.mathematical_models import HambergPKPD
from reil.healthcare.mathematical_models.hamberg_pkpd_2010 import HambergPKPD2010
from reil.healthcare.patient_warfarin_ravvaz import PatientWarfarinRavvaz

CACHE_SIZE = 100
DAYS = 90
DOSE_MG = 5.0
DOSE = {d: DOSE_MG for d in range(DAYS)}
MEAS = list(range(DAYS + 1))


def _one_pass(model_cls, n_patients: int) -> float:
    t0 = time.perf_counter()
    for seed in range(n_patients):
        m = model_cls(cache_size=CACHE_SIZE)
        p = PatientWarfarinRavvaz(model=m, random_seed=seed)
        p.model(dose=DOSE, measurement_days=MEAS)
    return time.perf_counter() - t0


def _bench(label: str, model_cls, n_patients: int, n_runs: int = 5) -> None:
    # warmup
    _one_pass(model_cls, min(n_patients, 5))
    times = [_one_pass(model_cls, n_patients) for _ in range(n_runs)]
    med = statistics.median(times)
    best = min(times)
    per_patient_ms = 1000.0 * med / n_patients
    print(f"  {label:40s}  median={med:7.3f}s  best={best:7.3f}s  "
          f"per_patient={per_patient_ms:6.2f}ms")


def main() -> None:
    print(f"HambergPKPD INR microbench (cache_size={CACHE_SIZE}, "
          f"{DAYS} days @ {DOSE_MG} mg/day):")
    _bench("HambergPKPD 1 patient", HambergPKPD, 1)
    _bench("HambergPKPD 500 patients", HambergPKPD, 500, n_runs=3)
    _bench("HambergPKPD2010 1 patient", HambergPKPD2010, 1)
    _bench("HambergPKPD2010 500 patients", HambergPKPD2010, 500, n_runs=3)


if __name__ == "__main__":
    main()
