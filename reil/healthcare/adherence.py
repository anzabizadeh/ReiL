# -*- coding: utf-8 -*-
'''
Adherence-event model for the warfarin simulation
==================================================

`AdherenceModel` transforms a *prescribed* daily-dose schedule into the
*administered* schedule that actually reaches the PK model, modeling
medication non-adherence: per-patient missed doses, clustered drug
"holidays", and occasional overuse (double dosing).

Where it sits
-------------
Between the agent's prescribed dose and the PK model's dose history. The
transform is applied in ``DosingSubject._take_effect`` to the ``dose`` dict
handed to ``patient.model(...)``; the subject's *prescribed*-dose history
(``_full_dose_history`` / ``_decision_points_dose_history``) is left
untouched. Consequences:

* It **does not touch the PK/PD math** — the Hamberg golden-fixture parity
  tests (``tests/unit/test_hamberg_pkpd_parity.py``) remain valid.
* Adherence is **hidden from the agent** — the observation stays INR history
  (and the dose the agent *believes* it gave), so the policy must *infer*
  drift, which is the core Chapter-3 monitoring decision.
* With ``mode='off'`` the model is a pass-through and every existing Paper-2
  and Paper-3 run is byte-identical.

Why it exists
-------------
In the plain simulation, control (TTR) is ~flat across the monitoring
interval because the Hamberg randomness (between-patient IIV + i.i.d.
residual INR error) has **no within-patient temporal drift**, so a longer
retest interval never loses control. Non-adherence is the clinically
dominant *autocorrelated, unobserved* within-patient process and the primary
reason INR monitoring exists: a run of missed doses drifts a patient
sub-therapeutic over days, which only more frequent testing can catch. This
creates a genuine PTTR-vs-monitoring-burden tradeoff for the duration policy
to optimize.

Grounded parameters and full rationale:
``Dissertation papers/210_paper3_adherence_model_proposal.md``. Anchors
(IN-RANGE MEMS study, Metlay et al. 2008; Salmasi et al. 2020 meta-analysis):
per-dose miss ~0.19, overuse ~0.03 (miss:overuse ~6:1), ~50% of patients
below the 80% PDC threshold. The run-length / holiday clustering for warfarin
is **not** pinned by the literature and is a modeling choice to sweep
(``holiday_persistence``).
'''
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Decouples the adherence RNG stream from the patient's PK residual-error
# stream: seeding from ``patient_seed ^ _SEED_SALT`` means turning adherence
# on does not perturb the PK noise draws (and off changes nothing at all),
# while keeping each patient's adherence reproducible from its seed.
_SEED_SALT = 0x5A17ADE
_SEED_MASK = 0xFFFFFFFF


class AdherenceModel:
    '''A per-patient medication-adherence transform.

    Modes
    -----
    ``'off'``    : pass-through (default). ``administer`` returns its input
                   unchanged; nothing is drawn.
    ``'iid'``    : each scheduled dose is taken independently with the
                   patient's adherence probability ``a_i`` (Bernoulli).
    ``'markov'`` : a sticky two-state (adherent <-> holiday) chain so misses
                   cluster into drug holidays of mean length
                   ``holiday_persistence`` while preserving the marginal
                   miss rate ``1 - a_i``.

    The per-patient adherence phenotype ``a_i`` is drawn once per patient from
    ``Beta(alpha, beta)`` (method-of-moments from ``adherence_mean`` /
    ``adherence_sd``), mirroring how the IIV etas are drawn once per patient.
    On a taken day, an independent ``overuse_prob`` chance replaces the dose
    with ``overuse_factor x`` (a double dose) to drive high-INR excursions.

    Lifecycle: construct once (from config), then call :meth:`new_patient` at
    every episode reset to (re)seed the stream and redraw the phenotype, then
    :meth:`administer` on each prescribed daily-dose dict.
    '''

    MODES = ('off', 'iid', 'markov')

    def __init__(
            self,
            mode: str = 'off',
            *,
            adherence_mean: float = 0.80,
            adherence_sd: float = 0.21,
            overuse_prob: float = 0.03,
            overuse_factor: float = 2.0,
            holiday_persistence: float = 3.0,
            adherence_bounds: tuple[float, float] = (0.05, 1.0),
            missing_dose: float = 0.0,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(
                f'Unknown adherence mode {mode!r}; expected one of {self.MODES}.')
        if not 0.0 < adherence_mean <= 1.0:
            raise ValueError('adherence_mean must be in (0, 1].')
        if adherence_sd < 0.0:
            raise ValueError('adherence_sd must be >= 0.')
        max_sd = (adherence_mean * (1.0 - adherence_mean)) ** 0.5
        if adherence_sd >= max_sd and adherence_mean < 1.0:
            raise ValueError(
                f'adherence_sd={adherence_sd} too large for mean='
                f'{adherence_mean}; must be < sqrt(m(1-m))={max_sd:.4f} for a '
                'valid Beta distribution.')
        if not 0.0 <= overuse_prob <= 1.0:
            raise ValueError('overuse_prob must be in [0, 1].')
        if overuse_factor < 0.0:
            raise ValueError('overuse_factor must be >= 0.')
        if holiday_persistence < 1.0:
            raise ValueError('holiday_persistence (mean holiday length) must be >= 1.')

        self._mode = mode
        self._adherence_mean = float(adherence_mean)
        self._adherence_sd = float(adherence_sd)
        self._overuse_prob = float(overuse_prob)
        self._overuse_factor = float(overuse_factor)
        self._holiday_persistence = float(holiday_persistence)
        self._adherence_bounds = adherence_bounds
        self._missing_dose = float(missing_dose)

        # Per-patient runtime state (set by new_patient).
        self._rng: np.random.Generator | None = None
        self._adherence: float = adherence_mean   # a_i
        self._p_miss: float = 1.0 - adherence_mean
        self._in_holiday: bool = False            # current markov state

    # -- introspection --------------------------------------------------
    @property
    def enabled(self) -> bool:
        '''True unless mode is ``'off'``.'''
        return self._mode != 'off'

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def adherence(self) -> float:
        '''The current patient's drawn adherence probability ``a_i``.'''
        return self._adherence

    def _beta_ab(self) -> tuple[float, float]:
        '''Method-of-moments (mean, sd) -> Beta(alpha, beta).'''
        m, v = self._adherence_mean, self._adherence_sd ** 2
        if v <= 0.0:
            # Degenerate: a point mass at the mean (large concentration).
            kappa = 1e6
        else:
            kappa = m * (1.0 - m) / v - 1.0
        kappa = max(kappa, 1e-6)
        return m * kappa, (1.0 - m) * kappa

    # -- lifecycle ------------------------------------------------------
    def new_patient(self, seed: int | None = None) -> None:
        '''(Re)seed the stream and draw a fresh per-patient phenotype.

        Call at every episode reset, *after* the patient has been generated.
        Seeding from the patient's own seed makes each patient's adherence
        reproducible and independent of the PK residual-error stream. A
        pass-through (``off``) model does nothing.
        '''
        if not self.enabled:
            return

        if seed is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = np.random.default_rng((int(seed) ^ _SEED_SALT) & _SEED_MASK)

        if self._adherence_sd == 0.0:
            self._adherence = self._adherence_mean
        else:
            alpha, beta = self._beta_ab()
            lo, hi = self._adherence_bounds
            self._adherence = float(np.clip(self._rng.beta(alpha, beta), lo, hi))
        self._p_miss = 1.0 - self._adherence

        # Initialise the markov chain from its stationary distribution so the
        # first days are not biased toward adherence.
        self._in_holiday = bool(self._rng.random() < self._p_miss)

    # -- transform ------------------------------------------------------
    def administer(
            self, prescribed: Mapping[int, float]) -> dict[int, float]:
        '''Map a ``{day: prescribed_dose}`` dict to administered doses.

        Missed days become ``missing_dose`` (0 by default); overuse days are
        multiplied by ``overuse_factor``. Days are processed in ascending
        order so the markov holiday state advances along the calendar. A
        pass-through model returns the input unchanged.
        '''
        if not self.enabled or not prescribed:
            return dict(prescribed)
        if self._rng is None:
            raise RuntimeError(
                'AdherenceModel.administer called before new_patient(); no '
                'phenotype/stream has been drawn.')

        out: dict[int, float] = {}
        for day in sorted(prescribed):
            dose = prescribed[day]
            if self._dose_taken():
                if self._overuse_prob > 0.0 and self._rng.random() < self._overuse_prob:
                    out[day] = dose * self._overuse_factor
                else:
                    out[day] = dose
            else:
                out[day] = self._missing_dose
        return out

    def _dose_taken(self) -> bool:
        '''Whether today's dose is taken; advances the markov state.'''
        if self._mode == 'iid':
            return bool(self._rng.random() >= self._p_miss)  # type: ignore[union-attr]

        # markov: today's intake reflects the current state, then transition.
        taken = not self._in_holiday
        p_h = self._p_miss
        p_a = 1.0 - p_h
        # Transitions chosen so the stationary marginal is P(holiday)=p_miss
        # and the mean holiday run-length is holiday_persistence (=1/p_HA).
        p_ha = 1.0 / self._holiday_persistence
        p_ah = 0.0 if p_a <= 0.0 else min(1.0, (p_h / p_a) * p_ha)
        r = self._rng.random()  # type: ignore[union-attr]
        if self._in_holiday:
            self._in_holiday = not (r < p_ha)   # leave holiday w.p. p_ha
        else:
            self._in_holiday = r < p_ah          # enter holiday w.p. p_ah
        return taken

    # -- construction from config --------------------------------------
    @classmethod
    def from_args(cls, args: Mapping[str, Any]) -> 'AdherenceModel | None':
        '''Build from a runner ``args`` mapping, or ``None`` when off.

        Reads ``adherence_mode`` and the ``adherence_*`` keys. Returns
        ``None`` for the off case so the object graph is identical to the
        pre-adherence code path (maximally parity-safe).
        '''
        mode = str(args.get('adherence_mode', 'off') or 'off')
        if mode == 'off':
            return None
        return cls(
            mode=mode,
            adherence_mean=float(args.get('adherence_mean', 0.80)),
            adherence_sd=float(args.get('adherence_sd', 0.21)),
            overuse_prob=float(args.get('adherence_overuse_prob', 0.03)),
            overuse_factor=float(args.get('adherence_overuse_factor', 2.0)),
            holiday_persistence=float(args.get('adherence_holiday_persistence', 3.0)),
        )

    def __repr__(self) -> str:
        return (
            f'AdherenceModel(mode={self._mode!r}, mean={self._adherence_mean}, '
            f'sd={self._adherence_sd}, overuse_prob={self._overuse_prob}, '
            f'holiday_persistence={self._holiday_persistence})')
