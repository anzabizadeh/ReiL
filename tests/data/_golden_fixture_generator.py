"""Golden-fixture generator for the HambergPKPD performance overhaul.

Produces three .npz files in this directory that serve as the parity baseline
for the perf overhaul described in the plan
`C:/Users/sj_an/.claude/plans/synthetic-nibbling-wolf.md`:

    hamberg_pkpd_golden.npz             constant-dose, Hamberg 2007 PKPD
    hamberg_pkpd_2010_golden.npz        constant-dose, Hamberg 2010 PKPD
    hamberg_pkpd_multidose_golden.npz   varying schedule, Hamberg 2007 PKPD

Each file stores a numpy array `inr` of shape `(n_patients, n_days+1)` plus the
metadata that pins how it was produced (`seeds`, `doses`, `measurement_days`).

Why .npz and not .parquet (the plan said parquet):
    pyarrow is not pinned in either Poetry env. .npz is numpy-native, zero new
    deps, and equivalent for our needs (numeric arrays only, no schema).

Run this ONCE under the unmodified ReiL code to lock in the baseline numbers,
then commit the .npz files. After any HambergPKPD math change, do not re-run
this script unless the change is intentional and the divergence is documented.

Usage:
    poetry run python -m tests.data._golden_fixture_generator
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from reil.healthcare.mathematical_models import HambergPKPD
from reil.healthcare.mathematical_models.hamberg_pkpd_2010 import HambergPKPD2010
from reil.healthcare.patient_warfarin_ravvaz import PatientWarfarinRavvaz

HERE = Path(__file__).parent

CONSTANT_DOSE_N_PATIENTS = 200
CONSTANT_DOSE_DAYS = 90
CONSTANT_DOSE_MG = 5.0

MULTIDOSE_N_PATIENTS = 50
MULTIDOSE_DAYS = 90
MULTIDOSE_DECISION_INTERVAL = 7
MULTIDOSE_SCHEDULE_SEED = 999
MULTIDOSE_DOSE_RANGE = (0.0, 10.0)
MULTIDOSE_DOSE_STEP = 0.5

CACHE_SIZE = 100  # > CONSTANT_DOSE_DAYS so _expand_caches never fires


def _build_multidose_schedule() -> dict[int, float]:
    """Build a fixed dose schedule shared by every multidose patient.

    13 decision points (days 0, 7, ..., 84), each a dose drawn from a seeded
    numpy RNG. Same schedule for all 50 patients keeps the test focused on
    `prescribe()`'s additive path rather than per-patient schedule variance.
    """
    rng = np.random.default_rng(MULTIDOSE_SCHEDULE_SEED)
    lo, hi = MULTIDOSE_DOSE_RANGE
    n_steps = int((hi - lo) / MULTIDOSE_DOSE_STEP) + 1
    decision_days = range(0, MULTIDOSE_DAYS, MULTIDOSE_DECISION_INTERVAL)
    schedule: dict[int, float] = {}
    for day in decision_days:
        step = int(rng.integers(0, n_steps))
        dose = lo + step * MULTIDOSE_DOSE_STEP
        # apply same dose for each day in the interval (matches DosingSubject's
        # `dose={i: current_dose for i in range(day, day+duration)}` pattern in
        # ReiL/reil/healthcare/subjects/dosing_subject.py:351-354)
        for d in range(day, min(day + MULTIDOSE_DECISION_INTERVAL, MULTIDOSE_DAYS)):
            schedule[d] = dose
    return schedule


def _run_one_patient(model_cls, seed: int, dose: dict[int, float],
                     measurement_days: list[int]) -> np.ndarray:
    """Instantiate one Ravvaz patient at `seed`, prescribe, return INR array."""
    model = model_cls(cache_size=CACHE_SIZE)
    patient = PatientWarfarinRavvaz(model=model, random_seed=seed)
    out = patient.model(dose=dose, measurement_days=measurement_days)
    return np.asarray(out["INR"], dtype=np.float64)


def _build_constant_dose(model_cls, n_patients: int, days: int,
                         dose_mg: float) -> tuple[np.ndarray, np.ndarray, dict, list[int]]:
    seeds = np.arange(n_patients, dtype=np.int64)
    dose = {d: dose_mg for d in range(days)}
    measurement_days = list(range(days + 1))
    inr = np.empty((n_patients, days + 1), dtype=np.float64)
    for i, seed in enumerate(seeds):
        inr[i] = _run_one_patient(model_cls, int(seed), dose, measurement_days)
    return inr, seeds, dose, measurement_days


def _build_multidose(model_cls, n_patients: int, days: int,
                     schedule: dict[int, float]) -> tuple[np.ndarray, np.ndarray, dict, list[int]]:
    seeds = np.arange(n_patients, dtype=np.int64)
    measurement_days = list(range(days + 1))
    inr = np.empty((n_patients, days + 1), dtype=np.float64)
    for i, seed in enumerate(seeds):
        inr[i] = _run_one_patient(model_cls, int(seed), schedule, measurement_days)
    return inr, seeds, schedule, measurement_days


def _save_npz(path: Path, inr: np.ndarray, seeds: np.ndarray,
              dose: dict[int, float], measurement_days: list[int]) -> None:
    dose_days = np.fromiter(dose.keys(), dtype=np.int64)
    dose_vals = np.fromiter(dose.values(), dtype=np.float64)
    np.savez_compressed(
        path,
        inr=inr,
        seeds=seeds,
        dose_days=dose_days,
        dose_vals=dose_vals,
        measurement_days=np.asarray(measurement_days, dtype=np.int64),
    )
    print(f"wrote {path}  inr={inr.shape}  finite={np.isfinite(inr).all()}")


def main() -> None:
    print("[1/3] HambergPKPD constant-dose (200 patients x 90 days @ 5 mg) ...")
    inr, seeds, dose, meas = _build_constant_dose(
        HambergPKPD, CONSTANT_DOSE_N_PATIENTS, CONSTANT_DOSE_DAYS, CONSTANT_DOSE_MG)
    _save_npz(HERE / "hamberg_pkpd_golden.npz", inr, seeds, dose, meas)

    print("[2/3] HambergPKPD2010 constant-dose (200 patients x 90 days @ 5 mg) ...")
    inr, seeds, dose, meas = _build_constant_dose(
        HambergPKPD2010, CONSTANT_DOSE_N_PATIENTS, CONSTANT_DOSE_DAYS, CONSTANT_DOSE_MG)
    _save_npz(HERE / "hamberg_pkpd_2010_golden.npz", inr, seeds, dose, meas)

    print("[3/3] HambergPKPD multidose (50 patients x 90 days, varying schedule) ...")
    schedule = _build_multidose_schedule()
    inr, seeds, sched, meas = _build_multidose(
        HambergPKPD, MULTIDOSE_N_PATIENTS, MULTIDOSE_DAYS, schedule)
    _save_npz(HERE / "hamberg_pkpd_multidose_golden.npz", inr, seeds, sched, meas)


if __name__ == "__main__":
    main()
