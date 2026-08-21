# -*- coding: utf-8 -*-  pylint: disable=undefined-variable
'''
warfarin class
==============

This `warfarin` class implements a two compartment PK/PD model for warfarin.
'''
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Literal, cast

from reil.datatypes.feature import Feature, FeatureGeneratorType, FeatureSet
from reil.healthcare.subjects.dosing_subject import DosingSubject
from reil.utils import reil_functions

DefComponents = tuple[tuple[str, dict[str, Any]], ...]

patient_basic: DefComponents = (
    ('age', {}), ('CYP2C9', {}),
    ('VKORC1', {})
)
patient_extra: DefComponents = (
    ('weight', {}), ('height', {}),
    ('gender', {}), ('race', {}), ('tobaco', {}),
    ('amiodarone', {}), ('fluvastatin', {})
)

sensitivity: DefComponents = (('sensitivity', {}),)
patient_w_sensitivity: DefComponents = (
    *patient_basic, *sensitivity, *patient_extra)

state_definitions: dict[str, DefComponents] = {
    'age': (('age', {}),),
    'patient_basic': patient_basic,
    'patient_w_sensitivity_basic': (*patient_basic, *sensitivity),
    'patient_w_sensitivity': patient_w_sensitivity,
    'patient': (*patient_basic, *patient_extra),
    'patient_w_dosing': (
        *patient_basic, *patient_extra,
        # ('day', {}),
        ('dose_history', {'length': -1}),
        ('INR_history', {'length': -1}),
        ('duration_history', {'length': -1})),
    'patient_for_baseline': (
        *patient_basic, *patient_extra,
        ('day', {}),
        ('dose_history', {'length': 4}),
        ('INR_history', {'length': 4}),
        ('duration_history', {'length': 4})),
    **{
        f'no_patient_w_dosing_{i:02}': (
            ('dose_history', {'length': i}),
            ('INR_history', {'length': i + 1}),
            ('duration_history', {'length': i}))
        for i in range(1, 4)},

    # Paper-3 stability-augmented policy state: no_patient_w_dosing_01 PLUS the
    # decision-time consecutive-in-range-days counter `s` -- the exact signal the
    # safety-ramp reward's tau_safe depends on. Without it the duration head is
    # blind to the stability that sets the reward-optimal interval and defaults
    # to tau=1 (see 230_paper3_duration_reward_findings.md). `at_decision=True`
    # gives the correct decision-time `s` (offset 0, not the reward offset).
    **{
        f'no_patient_w_dosing_stab_{i:02}': (
            ('dose_history', {'length': i}),
            ('INR_history', {'length': i + 1}),
            ('duration_history', {'length': i}),
            ('consecutive_in_range', {'at_decision': True}))
        for i in range(1, 4)},
    **{
        f'patient_w_dosing_{i:02}': (
            *patient_basic,
            ('dose_history', {'length': i}),
            ('INR_history', {'length': i + 1}),
            ('duration_history', {'length': i}))
        for i in range(1, 4)},

    **{
        f'patient_w_dosing_w_baseline_{i:02}': (
            *patient_basic, *patient_extra,
            ('day', {}),
            ('dose_history', {'length': i}),
            ('INR_history', {'length': i + 1}),
            ('duration_history', {'length': i}))
        for i in range(1, 4)},

    # Paper-3 stability-augmented FULL state (env `state_name`): the 2-phase
    # agent builds this superset then pops it down to `main_state_def`'s
    # features, so `consecutive_in_range` must live here too (else it is never
    # present to keep). Mirror of patient_w_dosing_w_baseline_* + `s`; pairs with
    # main_state_def=no_patient_w_dosing_stab_*. Keeps `day` for the 2-phase
    # init/main switch. `at_decision=True` -> decision-time `s`.
    **{
        f'patient_w_dosing_w_baseline_stab_{i:02}': (
            *patient_basic, *patient_extra,
            ('day', {}),
            ('dose_history', {'length': i}),
            ('INR_history', {'length': i + 1}),
            ('duration_history', {'length': i}),
            ('consecutive_in_range', {'at_decision': True}))
        for i in range(1, 4)},

    # Paper-3 EXTRAP-augmented stability states (doc 220 §9.5, 2026-07-18): the
    # stab state + the linear-extrapolation exit-day feature `extrap_exit` (a
    # model-free tau* estimate). main_state_def=no_patient_w_dosing_stab_extrap_*
    # (6-dim: dose,INR,INR,duration,s,extrap); env state_name pairs the FULL
    # superset so the 2-phase pop-down keeps extrap_exit.
    **{
        f'no_patient_w_dosing_stab_extrap_{i:02}': (
            ('dose_history', {'length': i}),
            ('INR_history', {'length': i + 1}),
            ('duration_history', {'length': i}),
            ('consecutive_in_range', {'at_decision': True}),
            ('extrap_exit', {}))
        for i in range(1, 4)},
    **{
        f'patient_w_dosing_w_baseline_stab_extrap_{i:02}': (
            *patient_basic, *patient_extra,
            ('day', {}),
            ('dose_history', {'length': i}),
            ('INR_history', {'length': i + 1}),
            ('duration_history', {'length': i}),
            ('consecutive_in_range', {'at_decision': True}),
            ('extrap_exit', {}))
        for i in range(1, 4)},

    'patient_w_full_dosing': (
        *patient_w_sensitivity,
        ('day', {}),
        ('daily_dose_history', {'length': -1}),
        ('daily_INR_history', {'length': -1}),
        ('duration_history', {'length': -1})),

    'daily_INR': (('daily_INR_history', {'length': -1}),),

    'recent_daily_INR': (('INR_within', {'length': 1}),),
    # Reward state for the dose-duration SAFETY reward: the window's daily INR
    # (via INR_within) PLUS the last two per-decision doses, so the reward can
    # read the dose change Delta = |dose[-1]/dose[-2]-1| for its safety ceiling.
    'recent_daily_INR_w_dose': (
        ('INR_within', {'length': 1}),
        ('dose_history', {'length': 2})),
    # + the consecutive-in-range-days stability counter `s`, for the safety-RAMP
    # reward (tau_safe climbs a retesting ladder with s, Aurora-style).
    'recent_daily_INR_w_dose_stab': (
        ('INR_within', {'length': 1}),
        ('dose_history', {'length': 2}),
        ('consecutive_in_range', {})),

    'Measured_INR_2': (
        ('INR_history', {'length': 2}),
        ('duration_history', {'length': 1})),
    'measured_dose_2': (('daily_dose_history', {'length': 2}),),
    'day_and_last_dose': (('day', {}), ('daily_dose_history', {'length': 1})),
    'day_and_last_dose_INR': (
        ('day', {}), ('daily_dose_history', {'length': 1}),
        ('daily_INR_history', {'length': 1}))
}

action_definition_names = [
    '237_15', 'daily_15', 'free_15', 'semi_15', 'weekly_15', 'delta',
    'percent', 'percent_semi', 'percent_semi_joint', 'semi']

reward_definitions: dict[str, tuple[reil_functions.ReilFunction[float, int], str]] = dict(
    sq_dist=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False),
        'recent_daily_INR'
    ),
    # Paper-3 per-head DOSE reward (2026-07-09): AVERAGE square distance
    # (average=True -> -c*mean_t dev^2), so control quality is length-invariant
    # and a longer interval is not structurally penalised for spanning more days.
    sq_dist_avg=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist_avg', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False, average=True),
        'recent_daily_INR'
    ),

    # ------------------------------------------------------------------
    # Paper-3 EB monitoring-cost reward (doc 220 §9, 2026-07-12):
    #   r = -c * sum_t (mu_t - 2.5)^2  -  kappa        (c = 4, band_width=1)
    # The SUMMED square-distance control (average=False) makes a longer interval
    # accrue more out-of-range days under the adherence sim, and `kappa`
    # (monitoring_coef) is a flat per-VISIT cost. Together they give the interval
    # a genuine control-vs-visits optimum with NO tau_safe / lambda scaffolding
    # (the scaffolding that caused the tau=1 collapse + s>=14 inversion, doc 230).
    # Sweeping kappa traces the PTTR-vs-visits Pareto frontier (EC); kappa=0
    # reproduces `sq_dist` exactly (the zero-burden anchor). A single shared
    # reward: drives a flat (dose,tau) joint head, or the duration head of a
    # per-head tandem (pair with `sq_dist_avg` on the dose head). Reads only
    # daily_INR_history, so it uses the plain `recent_daily_INR` state def.
    # ------------------------------------------------------------------
    **{
        f'sq_dist_kap{tag}': (
            reil_functions.NormalizedSquareDistance(
                name=f'sq_dist_kap{tag}', y_var_name='daily_INR_history',
                length=-1, multiplier=-1.0, interpolate=False,
                center=2.5, band_width=1.0, exclude_first=False,
                monitoring_coef=kap),
            'recent_daily_INR'
        )
        for tag, kap in (
            ('0p00', 0.0), ('0p25', 0.25), ('0p50', 0.5), ('0p75', 0.75),
            ('1p00', 1.0), ('2p00', 2.0), ('4p00', 4.0), ('8p00', 8.0),
        )
    },

    # ------------------------------------------------------------------
    # Paper-3 LOOKAHEAD duration reward (doc 220 §9.5, 2026-07-14): opportunity-cost
    # time-in-range. r = sum_t [+1 if INR in [2,3] else -pen] over the window; its
    # optimum interval is exactly tau* (the longest interval that stays in range),
    # so it realises the lookahead objective from the OBSERVED window (no separate
    # forward roll). Pair on the DURATION head (duration_reward_name=tir_p{pen}) with
    # sq_dist_avg on the dose head; `pen` = overshoot severity (alpha, beta=1).
    # ------------------------------------------------------------------
    **{
        f'tir_p{tag}': (
            reil_functions.TimeInRangeReward(
                name=f'tir_p{tag}', y_var_name='daily_INR_history',
                length=-1, interpolate=False, lo=2.0, hi=3.0,
                overshoot_penalty=pen),
            'recent_daily_INR'
        )
        for tag, pen in (('2', 2.0), ('4', 4.0), ('8', 8.0), ('16', 16.0))
    },

    # ------------------------------------------------------------------
    # Paper-3 additive dose-duration reward (plan 200 §3):
    #   r = lambda * tau  -  c * sum_t (mu_t - 2.5)^2 ,  c = 4 (band_width=1)
    # Same square-distance cost as `sq_dist`, plus a flat +lambda*tau interval
    # bonus. `lambda` (duration_coef) is the dose-duration trade-off dial; the
    # indifference deviation is delta* = sqrt(lambda / c). lambda=0 reproduces
    # `sq_dist` exactly. Grid = the D2 lambda-sweep {0, .15, .25, .5, 1, 1.5, 2,
    # 3, 5} (delta* ~ 0..1.12). Supersedes `custom_distance_4` for Paper-3 runs.
    # (2/3/5 added 2026-07-06 to explore higher lambda: NEWDOSE lengthens tau at
    # lambda>=1.5; EXPDOSE does not — see plan 200 §4 / the lambda-sweep memory.)
    # ------------------------------------------------------------------
    **{
        f'sq_dist_dur_l{tag}': (
            reil_functions.DurationIncentiveSquareDistance(
                name=f'sq_dist_dur_l{tag}', y_var_name='daily_INR_history',
                length=-1, multiplier=-1.0, interpolate=False,
                center=2.5, band_width=1.0, exclude_first=False,
                duration_coef=lam),
            'recent_daily_INR'
        )
        for tag, lam in (
            ('0p00', 0.0), ('0p15', 0.15), ('0p25', 0.25),
            ('0p50', 0.5), ('1p00', 1.0), ('1p50', 1.5),
            ('1p75', 1.75), ('2p00', 2.0), ('2p25', 2.25),
            ('2p50', 2.5), ('2p75', 2.75), ('3p00', 3.0), ('5p00', 5.0),
        )
    },

    # ------------------------------------------------------------------
    # Dose-duration SAFETY reward (Paper 3, 2026-07-07): lambda*tau - c*Sigma
    # dev^2 - rho*max(0, tau - tau_safe(Delta, mu0))^2. tau_safe = long (28) if
    # stable (in range, no dose change), medium (7) if small change in range,
    # short (2) if big change or out-of-range INR. Reads dose_history for Delta
    # via the `recent_daily_INR_w_dose` state def. Sweep over (lambda, rho).
    # ------------------------------------------------------------------
    **{
        f'dd_safe_l{lt}_r{rt}': (
            reil_functions.DoseDurationSafetyReward(
                name=f'dd_safe_l{lt}_r{rt}', y_var_name='daily_INR_history',
                length=-1, multiplier=-1.0, interpolate=False,
                center=2.5, band_width=1.0, exclude_first=False,
                duration_coef=lam, safety_coef=rho,
                dose_var_name='dose_history',
                tau_short=2.0, tau_med=7.0, tau_long=28.0,
                delta_zero=0.05, delta_big=0.30,
                range_lo=2.0, range_hi=3.0),
            'recent_daily_INR_w_dose'
        )
        for lt, lam in (('1p00', 1.0), ('1p50', 1.5), ('2p00', 2.0))
        for rt, rho in (('1p00', 1.0), ('2p00', 2.0), ('4p00', 4.0))
    },

    # Form B (symmetric target): r = -c*Sigma dev^2 - rho*(tau - tau_safe)^2,
    # duration_coef=0 (tau_safe encodes the interval preference). Penalises BOTH
    # over- and under-shooting tau_safe -> removes the tau=1 safe-harbor that
    # collapsed Form A (dd_safe_*). Small rho keeps it on the control-term scale.
    **{
        f'dd_targ_r{rt}': (
            reil_functions.DoseDurationSafetyReward(
                name=f'dd_targ_r{rt}', y_var_name='daily_INR_history',
                length=-1, multiplier=-1.0, interpolate=False,
                center=2.5, band_width=1.0, exclude_first=False,
                duration_coef=0.0, safety_coef=rho, symmetric=True,
                dose_var_name='dose_history',
                tau_short=2.0, tau_med=7.0, tau_long=28.0,
                delta_zero=0.05, delta_big=0.30,
                range_lo=2.0, range_hi=3.0),
            'recent_daily_INR_w_dose'
        )
        for rt, rho in (('0p02', 0.02), ('0p05', 0.05), ('0p10', 0.1), ('0p20', 0.2))
    },

    # SAFETY-RAMP reward (Paper 3, 2026-07-07, Aurora/Intermountain-informed):
    # r = lambda*tau - c*Sigma dev^2 - rho*(tau - tau_safe(zone, s))^2. Ramp
    # tau_safe: big-change/far-off -> 2, mildly off -> 5, in range -> largest
    # ladder rung <= s (consecutive in-range days). Reads `consecutive_in_range`
    # via `recent_daily_INR_w_dose_stab`. Small lambda offsets the control term's
    # short-tau pull; the symmetric penalty centres tau on the current rung.
    **{
        f'dd_ramp_l{lt}_r{rt}': (
            reil_functions.DoseDurationSafetyReward(
                name=f'dd_ramp_l{lt}_r{rt}', y_var_name='daily_INR_history',
                length=-1, multiplier=-1.0, interpolate=False,
                center=2.5, band_width=1.0, exclude_first=False,
                duration_coef=lam, safety_coef=rho,
                symmetric=True, ramp=True,
                dose_var_name='dose_history',
                stab_var_name='consecutive_in_range',
                delta_big=0.30, tau_short=2.0, tau_mild=5.0,
                far_lo=1.5, far_hi=4.0, range_lo=2.0, range_hi=3.0,
                ladder=(1, 3, 7, 14, 28)),
            'recent_daily_INR_w_dose_stab'
        )
        for lt, lam in (('0p30', 0.3), ('0p50', 0.5), ('1p50', 1.5))
        for rt, rho in (('0p10', 0.1), ('0p30', 0.3), ('0p50', 0.5), ('1p00', 1.0))
    },

    # Paper-3 per-head DURATION reward (2026-07-09): ramp safety reward with the
    # control term DROPPED (include_control=False) -> r = lambda*tau - rho*(tau -
    # tau_safe(s))^2. Pure burden vs safety; control is the dose head's job.
    **{
        f'dd_ramp_nc_l{lt}_r{rt}': (
            reil_functions.DoseDurationSafetyReward(
                name=f'dd_ramp_nc_l{lt}_r{rt}', y_var_name='daily_INR_history',
                length=-1, multiplier=-1.0, interpolate=False,
                center=2.5, band_width=1.0, exclude_first=False,
                duration_coef=lam, safety_coef=rho,
                symmetric=True, ramp=True, include_control=False,
                dose_var_name='dose_history',
                stab_var_name='consecutive_in_range',
                delta_big=0.30, tau_short=2.0, tau_mild=5.0,
                far_lo=1.5, far_hi=4.0, range_lo=2.0, range_hi=3.0,
                ladder=(1, 3, 7, 14, 28)),
            'recent_daily_INR_w_dose_stab'
        )
        for lt, lam in (('0p50', 0.5), ('1p50', 1.5))
        for rt, rho in (('0p50', 0.5), ('1p00', 1.0))
    },

    # Paper-3 per-head DURATION reward with HUBER safety penalty (2026-07-09):
    #   r = lambda*tau - rho*Huber_delta(tau - tau_safe(s))   (no control term)
    # Quadratic near tau_safe, linear beyond delta -> bounded gradient removes
    # the catastrophic-guess variance that pinned tau=1 under the quadratic
    # penalty. lambda(lt) x rho(rt) x delta(dt) grid.
    **{
        f'dd_huber_l{lt}_r{rt}_d{dt}': (
            reil_functions.DoseDurationSafetyReward(
                name=f'dd_huber_l{lt}_r{rt}_d{dt}', y_var_name='daily_INR_history',
                length=-1, multiplier=-1.0, interpolate=False,
                center=2.5, band_width=1.0, exclude_first=False,
                duration_coef=lam, safety_coef=rho,
                symmetric=True, ramp=True, include_control=False,
                penalty_shape='huber', huber_delta=float(dd),
                dose_var_name='dose_history',
                stab_var_name='consecutive_in_range',
                delta_big=0.30, tau_short=2.0, tau_mild=5.0,
                far_lo=1.5, far_hi=4.0, range_lo=2.0, range_hi=3.0,
                ladder=(1, 3, 7, 14, 28)),
            'recent_daily_INR_w_dose_stab'
        )
        for lt, lam in (('1p50', 1.5), ('2p50', 2.5), ('3p50', 3.5))
        for rt, rho in (('0p50', 0.5), ('1p00', 1.0))
        for dt, dd in (('2', 2.0), ('3', 3.0))
    },

    # Paper-3 per-head DURATION reward, Huber + IN-RANGE FLOOR of 7 (2026-07-10).
    # ladder=(7,14,28): in-range tau_safe = 7 (Paper-2's constant-7 default),
    # climbing to 14 (s>=14) / 28 (s>=28); mildly-off -> tau_mild(5), far-off /
    # big-change -> tau_short(2). So durations default to WEEKLY when in range and
    # drop <7 only for genuine instability -- the "mostly >=7 with flexibility"
    # target. lambda(lt) x rho(rt) x delta(dt) grid.
    **{
        f'dd_hub7_l{lt}_r{rt}_d{dt}': (
            reil_functions.DoseDurationSafetyReward(
                name=f'dd_hub7_l{lt}_r{rt}_d{dt}', y_var_name='daily_INR_history',
                length=-1, multiplier=-1.0, interpolate=False,
                center=2.5, band_width=1.0, exclude_first=False,
                duration_coef=lam, safety_coef=rho,
                symmetric=True, ramp=True, include_control=False,
                penalty_shape='huber', huber_delta=float(dd),
                dose_var_name='dose_history',
                stab_var_name='consecutive_in_range',
                delta_big=0.30, tau_short=2.0, tau_mild=5.0,
                far_lo=1.5, far_hi=4.0, range_lo=2.0, range_hi=3.0,
                ladder=(7, 14, 28)),
            'recent_daily_INR_w_dose_stab'
        )
        for lt, lam in (('1p00', 1.0), ('1p50', 1.5), ('2p50', 2.5))
        for rt, rho in (('0p50', 0.5), ('1p00', 1.0))
        for dt, dd in (('3', 3.0),)
    },

    # Paper-3 reward-form sweep (2026-07-11) — floor-7 variants for a clean
    # joint-vs-separate x huber-vs-quadratic comparison on the dense 1..28 grid.
    #   dd_q7   : separate/per-head, QUADRATIC penalty, no control term
    #             (the non-huber analogue of dd_hub7 at the same floor-7 ladder).
    **{
        f'dd_q7_l{lt}_r{rt}': (
            reil_functions.DoseDurationSafetyReward(
                name=f'dd_q7_l{lt}_r{rt}', y_var_name='daily_INR_history',
                length=-1, multiplier=-1.0, interpolate=False,
                center=2.5, band_width=1.0, exclude_first=False,
                duration_coef=lam, safety_coef=rho,
                symmetric=True, ramp=True, include_control=False,
                penalty_shape='quadratic',
                dose_var_name='dose_history',
                stab_var_name='consecutive_in_range',
                delta_big=0.30, tau_short=2.0, tau_mild=5.0,
                far_lo=1.5, far_hi=4.0, range_lo=2.0, range_hi=3.0,
                ladder=(7, 14, 28)),
            'recent_daily_INR_w_dose_stab'
        )
        for lt, lam in (('1p50', 1.5), ('2p50', 2.5))
        for rt, rho in (('0p50', 0.5), ('1p00', 1.0))
    },
    #   dd_hub7c / dd_q7c : JOINT (shared) — keep the control term (average=True
    #             for length-invariance) so a single shared reward drives BOTH
    #             heads (dose via avg control, duration via lambda*tau - safety).
    #             c=huber, q=quadratic penalty. For reward_name= (no per-head).
    **{
        f'dd_hub7c_l{lt}_r{rt}_d{dt}': (
            reil_functions.DoseDurationSafetyReward(
                name=f'dd_hub7c_l{lt}_r{rt}_d{dt}', y_var_name='daily_INR_history',
                length=-1, multiplier=-1.0, interpolate=False,
                center=2.5, band_width=1.0, exclude_first=False,
                duration_coef=lam, safety_coef=rho,
                symmetric=True, ramp=True, include_control=True, average=True,
                penalty_shape='huber', huber_delta=float(dd),
                dose_var_name='dose_history',
                stab_var_name='consecutive_in_range',
                delta_big=0.30, tau_short=2.0, tau_mild=5.0,
                far_lo=1.5, far_hi=4.0, range_lo=2.0, range_hi=3.0,
                ladder=(7, 14, 28)),
            'recent_daily_INR_w_dose_stab'
        )
        for lt, lam in (('2p50', 2.5),)
        for rt, rho in (('1p00', 1.0),)
        for dt, dd in (('3', 3.0),)
    },
    **{
        f'dd_q7c_l{lt}_r{rt}': (
            reil_functions.DoseDurationSafetyReward(
                name=f'dd_q7c_l{lt}_r{rt}', y_var_name='daily_INR_history',
                length=-1, multiplier=-1.0, interpolate=False,
                center=2.5, band_width=1.0, exclude_first=False,
                duration_coef=lam, safety_coef=rho,
                symmetric=True, ramp=True, include_control=True, average=True,
                penalty_shape='quadratic',
                dose_var_name='dose_history',
                stab_var_name='consecutive_in_range',
                delta_big=0.30, tau_short=2.0, tau_mild=5.0,
                far_lo=1.5, far_hi=4.0, range_lo=2.0, range_hi=3.0,
                ladder=(7, 14, 28)),
            'recent_daily_INR_w_dose_stab'
        )
        for lt, lam in (('2p50', 2.5),)
        for rt, rho in (('1p00', 1.0),)
    },

    sq_dist_modified=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist_modified', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05),
        'recent_daily_INR'
    ),
    average_sq_dist_modified_w_constant=(
        reil_functions.NormalizedSquareDistance(
            name='average_sq_dist_modified_w_constant',
            y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.0, average=True, constant=0.),
        'recent_daily_INR'
    ),
    custom_distance=(
        reil_functions.CustomDistance(
            name='custom_distance',
            y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False),
        'recent_daily_INR'
    ),
    custom_distance_2=(
        reil_functions.CustomDistance2(
            name='custom_distance_2',
            y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False),
        'recent_daily_INR'
    ),
    custom_distance_3=(
        reil_functions.CustomDistance3(
            name='custom_distance_3',
            y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False),
        'recent_daily_INR'
    ),
    custom_distance_4=(
        reil_functions.CustomDistance4(
            name='custom_distance_4',
            y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False),
        'recent_daily_INR'
    ),
    custom_distance_4b=(
        reil_functions.CustomDistance4b(
            name='custom_distance_4b',
            y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False),
        'recent_daily_INR'
    ),
    dist=(
        reil_functions.NormalizedDistance(
            name='dist', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False),
        'recent_daily_INR'
    ),
    sq_dist_interpolation=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist_interpolation',
            y_var_name='INR_history', x_var_name='duration_history',
            length=2, multiplier=-1.0,  interpolate=True,
            center=2.5, band_width=1.0, exclude_first=True),
        'Measured_INR_2'
    ),
    PTTR_exact=(
        reil_functions.PercentInRange(
            name='PTTR_exact', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0,  interpolate=False,
            acceptable_range=(2, 3), exclude_first=True),
        'recent_daily_INR'
    ),
    dose_change=(
        reil_functions.NotEqual(
            name='dose_change', y_var_name='daily_dose_history',
            length=2, multiplier=-1.0),
        'measured_dose_2'
    ),
    PTTR_interpolation=(
        reil_functions.PercentInRange(
            name='PTTR_interpolation',
            y_var_name='INR_history', x_var_name='duration_history',
            length=2, multiplier=-1.0,  interpolate=True,
            acceptable_range=(2, 3), exclude_first=True),
        'Measured_INR_2'
    ),

    # ------------------------------------------------------------------
    # Paper-2 reward-shape study (EXP-C2-RW1) — variants of the canonical
    # `sq_dist_modified` (center=2.5, band_width=1.0, eta=1.05) used to test
    # whether the quadratic-from-midpoint reward is what places the distilled
    # cut-offs inside the [2, 3] therapeutic range. Each entry changes exactly
    # one factor vs. `sq_dist_modified` so the contrast is clean.
    # See 50_paper1_chapter2_canonical.md#exp-c2-rw1.
    # ------------------------------------------------------------------

    # #5 center shift (eta held at 1.05): does the target point set the cut-offs?
    sq_dist_modified_c2p3=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist_modified_c2p3', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.3, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05),
        'recent_daily_INR'
    ),
    sq_dist_modified_c2p7=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist_modified_c2p7', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.7, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05),
        'recent_daily_INR'
    ),
    # #5 eta sweep (center held at 2.5). eta=1.0 is `sq_dist`; eta=1.05 is
    # `sq_dist_modified`; only eta=1.1 is new.
    sq_dist_modified_eta1p1=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist_modified_eta1p1', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.1),
        'recent_daily_INR'
    ),
    # #1 deadband sweep (center=2.5, eta=1.05): flat zone of half-width eps.
    # eps=0 is `sq_dist_modified`; eps=0.5 makes the reward flat across [2, 3].
    deadband_eps0p1=(
        reil_functions.DeadbandSquareDistance(
            name='deadband_eps0p1', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05, tolerance=0.1),
        'recent_daily_INR'
    ),
    deadband_eps0p25=(
        reil_functions.DeadbandSquareDistance(
            name='deadband_eps0p25', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05, tolerance=0.25),
        'recent_daily_INR'
    ),
    deadband_eps0p5=(
        reil_functions.DeadbandSquareDistance(
            name='deadband_eps0p5', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05, tolerance=0.5),
        'recent_daily_INR'
    ),
    # #3 linear control (center=2.5, eta=1.05): abs instead of square — isolates
    # curvature from centering. (`dist` is the eta=1.0 version.)
    dist_modified=(
        reil_functions.NormalizedDistance(
            name='dist_modified', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05),
        'recent_daily_INR'
    ),
    # #4 asymmetric (center=2.5, eta=1.05): heavier penalty above center
    # (supratherapeutic / bleeding side).
    asym_over2=(
        reil_functions.AsymmetricSquareDistance(
            name='asym_over2', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05, under_weight=1.0, over_weight=2.0),
        'recent_daily_INR'
    ),
    asym_over4=(
        reil_functions.AsymmetricSquareDistance(
            name='asym_over4', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05, under_weight=1.0, over_weight=4.0),
        'recent_daily_INR'
    ),
    # #2 in-range indicator: reward = fraction of days in [2, 3] (multiplier
    # +1.0, MAXIMISED). The existing `PTTR_exact` reward has multiplier=-1.0
    # (a cost, never selected as a training reward); this is the correctly
    # signed maximisation reward for the train/eval-aligned arm.
    pttr_in_range=(
        reil_functions.PercentInRange(
            name='pttr_in_range', y_var_name='daily_INR_history',
            length=-1, multiplier=1.0, interpolate=False,
            acceptable_range=(2, 3), exclude_first=True),
        'recent_daily_INR'
    ),

    # --- EXP-C2-RW1 extension (2026-06-22): higher asymmetry + high eta ---
    asym_over8=(
        reil_functions.AsymmetricSquareDistance(
            name='asym_over8', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05, under_weight=1.0, over_weight=8.0),
        'recent_daily_INR'
    ),
    asym_over16=(
        reil_functions.AsymmetricSquareDistance(
            name='asym_over16', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.05, under_weight=1.0, over_weight=16.0),
        'recent_daily_INR'
    ),
    # EXP-C2-IJ9 (Paper 2): severe-excursion hinge on top of the eta=1.0 anchor
    # `sq_dist`. Zero inside the therapeutic range, so unlike asym_over* it does
    # not move the effective target; it only prices the bleed-risk tail that the
    # -30 reward_clip floor otherwise renders invisible. Pair with a relaxed
    # reward_clip -- see SevereExcursionSquareDistance's docstring.
    hipen_hi4_w4=(
        reil_functions.SevereExcursionSquareDistance(
            name='hipen_hi4_w4', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.0, hi=4.0, hi_weight=4.0),
        'recent_daily_INR'
    ),
    hipen_hi3p5_w4=(
        reil_functions.SevereExcursionSquareDistance(
            name='hipen_hi3p5_w4', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.0, hi=3.5, hi_weight=4.0),
        'recent_daily_INR'
    ),
    # high eta (direction amplifier); compare to sq_dist (1.0), sq_dist_modified
    # (1.05), sq_dist_modified_eta1p1 (1.1).
    sq_dist_modified_eta1p2=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist_modified_eta1p2', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.2),
        'recent_daily_INR'
    ),
    sq_dist_modified_eta1p5=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist_modified_eta1p5', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=1.5),
        'recent_daily_INR'
    ),
    sq_dist_modified_eta2p0=(
        reil_functions.NormalizedSquareDistance(
            name='sq_dist_modified_eta2p0', y_var_name='daily_INR_history',
            length=-1, multiplier=-1.0, interpolate=False,
            center=2.5, band_width=1.0, exclude_first=False,
            amplifying_factor=2.0),
        'recent_daily_INR'
    ),
)

statistic_definition_names = ['PTTR_exact_basic', 'PTTR_exact']

statistic_PTTR = reil_functions.PercentInRange(
    name='PTTR', y_var_name='daily_INR_history',
    length=-1, multiplier=1.0,  interpolate=False,
    acceptable_range=(2, 3), exclude_first=True)


class Warfarin(DosingSubject):
    '''
    A warfarin subject based on Hamberg's two compartment PK/PD model.
    '''

    def __init__(
            self,
            INR_range: tuple[float, float] = (0.0, 15.0),
            dose_range: tuple[float, float] = (0.0, 15.0),
            dose_step: float = 0.5,
            duration_range: tuple[int, int] = (1, 28),
            duration_step: int | None = None,
            duration_values: tuple[int, ...] | None = None,
            max_day: int = 90,
            decision_mode: Literal[
                'dose', 'dose_duration', 'dose_change', 'dose_change_duration',
                'dose_percent_change', 'dose_percent_change_duration'
            ] = 'dose_duration',
            decision_values: tuple[float, ...] | None = None,
            decision_range: tuple[float, float] | None = None,
            round_to_step: bool = True,
            backfill: bool = True,
            **kwargs: Any):
        '''
        Arguments
        ---------
        patient:
            A patient object that generates new patients and models
            interaction between dose and INR.

        INR_range:
            A tuple that specifies min and max INR.

        dose_range:
            A tuple that specifies min and max dose.

        duration_range:
            A tuple that specifies min and max number of days between two
            measurements.

        max_day:
            Maximum duration of each trial.

        '''
        if duration_step is None and duration_values is None:
            duration_step = 1

        super().__init__(
            measurement_name='INR',
            measurement_range=INR_range,
            max_day=max_day,
            backfill=backfill,
            dose_range=dose_range,
            dose_step=dose_step,
            duration_range=duration_range,
            duration_step=duration_step,
            duration_values=duration_values,
            decision_mode=decision_mode,
            decision_values=decision_values,
            decision_range=decision_range,
            round_to_step=round_to_step,
            **kwargs)

        self.state.definition_reference_function(
            f=self._state_def_reference,
            available_definitions=list(state_definitions))
        self.possible_actions.definition_reference_function(
            f=self._action_def_reference,
            available_definitions=action_definition_names)
        self.reward.definition_reference_function(
            f=self._reward_def_reference,
            available_definitions=list(reward_definitions))
        self.statistic.definition_reference_function(
            f=self._statistic_def_reference,
            available_definitions=list(statistic_definition_names))

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        del config['measurement_name']
        config['INR_range'] = config.pop('measurement_range')

        return config

    def copy(
        self, perturb: bool = False, n: int | None = None
    ) -> 'Warfarin' | list['Warfarin']:
        copied_subjects_temp = super().copy(perturb=False, n=n)

        if perturb:
            if n is None:
                copied_subjects = cast(Warfarin, copied_subjects_temp)
                if copied_subjects._patient is not None:
                    copied_subjects._patient._model.perturb(  # type: ignore
                        day=self._day)
            else:
                copied_subjects = cast(list[Warfarin], copied_subjects_temp)
                for c in copied_subjects:
                    if c._patient is not None:
                        c._patient._model.perturb(day=self._day)  # type: ignore
        else:
            copied_subjects = cast(
                Warfarin | list[Warfarin], copied_subjects_temp)

        return copied_subjects

    def _generate_state_defs(self):
        current_defs = self.state.definitions
        for name, args in state_definitions.items():
            if name not in current_defs:
                self.state.add_definition(name, *args)

    def _generate_reward_defs(self):
        current_defs = self.reward.definitions

        for name, args in reward_definitions.items():
            if name not in current_defs:
                self.reward.add_definition(
                    name, *args)

    def _generate_statistic_defs(self):
        if 'PTTR_exact_basic' not in self.statistic.definitions:
            self.statistic.add_definition(
                'PTTR_exact_basic', statistic_PTTR,
                'daily_INR', 'patient_w_sensitivity_basic')

        if 'PTTR_exact' not in self.statistic.definitions:
            self.statistic.add_definition(
                'PTTR_exact', statistic_PTTR,
                'daily_INR', 'patient_w_sensitivity')

    def _generate_action_defs(self):  # noqa: C901
        current_action_definitions = self.possible_actions.definitions

        def _generate(
            feature: FeatureSet,
            ops: tuple[Callable[[FeatureSet], bool], ...],
            dose_masks: tuple[dict[float, float], ...],
            duration_masks: tuple[dict[int, int], ...]
        ) -> FeatureGeneratorType:
            self.action_gen_set.unmask('dose')
            if not self._duration_mode:
                for op, d_mask in zip(ops, dose_masks):
                    if op(feature):
                        self.action_gen_set.mask('dose', d_mask)

                        return self.action_gen_set.make_generator()

            else:
                self.action_gen_set.unmask('duration')
                for op, d_mask, i_mask in zip(ops, dose_masks, duration_masks):
                    if op(feature):
                        self.action_gen_set.mask('dose', d_mask)
                        self.action_gen_set.mask('duration', i_mask)

                        return self.action_gen_set.make_generator()

                self.action_gen_set.mask('duration', duration_masks[-1])

            self.action_gen_set.mask('dose', dose_masks[-1])

            return self.action_gen_set.make_generator()

        caps = tuple(
            i for i in (5.0, 10.0, 15.0)
            if self._dose_range[0] <= i <= self._dose_range[1])
        max_cap = min(caps[-1], self._dose_range[1])

        dose = {
            cap: {
                d: cap
                for d in self.generate_dose_values(cap, max_cap, 0.5)
                if d > cap}
            for cap in caps}

        min_duration, max_duration = self._duration_range
        int_fixed = {
            d: {
                i: d
                for i in (self._duration_values or range(
                    min_duration, max_duration + 1,
                    self._duration_step))  # type: ignore
                if i != d}
            for d in (1, 2, 3, 7)}
        int_semi_free = {
            i: min_duration
            for i in (self._duration_values or range(
                min_duration, max_duration + 1,
                self._duration_step))  # type: ignore
            if i not in (1, 2, 3, 7, 14, 28)}
        int_weekly = {
            i: min_duration
            for i in (self._duration_values or range(
                min_duration, max_duration + 1,
                self._duration_step))  # type: ignore
            if i not in (7, 14, 21, 28)}

        name: str
        for cap in caps[:-1]:
            name = f'237_{int(cap):02}'
            if name not in current_action_definitions:
                self.possible_actions.add_definition(
                    name, functools.partial(
                        _generate,
                        ops=(
                            lambda f: f['day'].value >= 5,  # type: ignore
                            lambda f: f['day'].value == 2,
                            lambda f: f['day'].value == 0),
                        dose_masks=(
                            dose[max_cap], dose[max_cap], dose[cap]
                        ),
                        duration_masks=(
                            int_fixed[7], int_fixed[3], int_fixed[2])
                    ),
                    'day')

            name = f'daily_{int(cap):02}'
            if name not in current_action_definitions:
                self.possible_actions.add_definition(
                    name, functools.partial(
                        _generate,
                        ops=(lambda f: f['day'].value > 0,),  # type: ignore
                        dose_masks=(dose[max_cap], dose[cap]),
                        duration_masks=(int_fixed[1], int_fixed[1])
                    ),
                    'day')

            name = f'free_{int(cap):02}'
            if name not in current_action_definitions:
                self.possible_actions.add_definition(
                    name, functools.partial(
                        _generate,
                        ops=(lambda f: f['day'].value > 0,),  # type: ignore
                        dose_masks=(dose[max_cap], dose[cap]),
                        duration_masks=({}, {})),
                    'day')

            name = f'semi_{int(cap):02}'
            if name not in current_action_definitions:
                self.possible_actions.add_definition(
                    name, functools.partial(
                        _generate,
                        ops=(lambda f: f['day'].value > 0,),  # type: ignore
                        dose_masks=(dose[max_cap], dose[cap]),
                        duration_masks=(int_semi_free, int_semi_free)),
                    'day')

            name = f'weekly_{int(cap):02}'
            if name not in current_action_definitions:
                self.possible_actions.add_definition(
                    name, functools.partial(
                        _generate,
                        ops=(lambda f: f['day'].value > 0,),  # type: ignore
                        dose_masks=(dose[max_cap], dose[cap]),
                        duration_masks=(int_weekly, int_weekly)),
                    'day')

        name = '237_15'
        if name not in current_action_definitions:
            self.possible_actions.add_definition(
                name, functools.partial(
                    _generate,
                    ops=(
                        lambda f: f['day'].value >= 5,  # type: ignore
                        lambda f: f['day'].value == 2,
                        lambda f: f['day'].value == 0),
                    dose_masks=(
                        dose[max_cap], dose[max_cap], dose[max_cap]
                    ),
                    duration_masks=(int_fixed[7], int_fixed[3], int_fixed[2])),
                'day')

        name = 'daily_15'
        if name not in current_action_definitions:
            self.possible_actions.add_definition(
                name, functools.partial(
                    _generate,
                    ops=(),
                    dose_masks=(dose[max_cap],),
                    duration_masks=(int_fixed[1],)),
                'day')

        name = 'free_15'
        if name not in current_action_definitions:
            self.possible_actions.add_definition(
                name, functools.partial(
                    _generate,
                    ops=(),
                    dose_masks=(dose[max_cap],),
                    duration_masks=({},)),
                'day')

        name = 'semi_15'
        if name not in current_action_definitions:
            self.possible_actions.add_definition(
                name, functools.partial(
                    _generate,
                    ops=(),
                    dose_masks=(dose[max_cap],),
                    duration_masks=(int_semi_free,)),
                'day')

        name = 'weekly_15'
        if name not in current_action_definitions:
            self.possible_actions.add_definition(
                name, functools.partial(
                    _generate,
                    ops=(lambda f: f['day'].value > 0,),  # type: ignore
                    dose_masks=(dose[max_cap],),
                    duration_masks=(int_weekly,)),
                'day')

        name = 'delta'
        if name not in current_action_definitions:
            def delta_dose(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_change')
                self.action_gen_set.unmask('duration')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                day: int = feature['day'].value  # type: ignore
                min_dose, max_dose = self._dose_range
                min_delta = min_dose - last_dose
                max_delta = max_dose - last_dose
                d_list = self.generate_dose_values(min_dose, max_dose)
                d_list = set((*d_list, *(-x for x in d_list)))
                d_mask = {
                    d: min_delta if d < min_delta else max_delta
                    for d in d_list
                    if not (min_delta <= d <= max_delta)
                }
                self.action_gen_set.mask('dose_change', d_mask)

                if day >= 5:
                    duration_mask = int_fixed[7]
                elif day == 2:
                    duration_mask = int_fixed[3]
                elif day == 0:
                    duration_mask = int_fixed[2]
                else:
                    raise ValueError(f'wrong day: {day}.')

                self.action_gen_set.mask('duration', duration_mask)

                return self.action_gen_set.make_generator()

            self.possible_actions.add_definition(
                name, delta_dose, 'day_and_last_dose')

        # name = 'percent'
        # if name not in current_action_definitions:
        #     def percent_dose(feature: FeatureSet) -> FeatureGeneratorType:
        #         self.action_gen_set.unmask('dose_percent_change')
        #         self.action_gen_set.unmask('duration')
        #         last_dose: float = \
        #             feature['daily_dose_history'].value[-1]  # type: ignore
        #         day: int = feature['day'].value  # type: ignore
        #         min_dose, max_dose = self._dose_range
        #         min_delta = min_dose - last_dose
        #         max_delta = max_dose - last_dose
        #         d_list = self.generate_dose_values(min_dose, max_dose)
        #         d_list = set((*d_list, *(-x for x in d_list)))
        #         d_mask = {
        #             d: min_delta if d < min_delta else max_delta
        #             for d in d_list
        #             if not (min_delta <= d <= max_delta)
        #         }
        #         self.action_gen_set.mask('dose_percent_change', d_mask)

        #         if day >= 5:
        #             duration_mask = int_fixed[7]
        #         elif day == 2:
        #             duration_mask = int_fixed[3]
        #         elif day == 0:
        #             duration_mask = int_fixed[2]
        #         else:
        #             raise ValueError(f'wrong day: {day}.')

        #         self.action_gen_set.mask('duration', duration_mask)

        #         return self.action_gen_set.make_generator()

        #     self.possible_actions.add_definition(
        #         name, percent_dose, 'day_and_last_dose')

    def _state_def_reference(
            self, name: str) -> DefComponents | None:
        try:
            return state_definitions[name]
        except KeyError:
            return super()._state_def_reference(name)

    def _action_def_reference(  # noqa: C901
        self, name: str
    ) -> tuple[Callable[..., FeatureGeneratorType], str] | None:
        def _generate(
                feature: FeatureSet,
                ops: tuple[Callable[[FeatureSet], bool], ...],
                dose_masks: tuple[dict[float, float], ...],
                duration_masks: tuple[dict[int, int], ...]
        ) -> FeatureGeneratorType:
            self.action_gen_set.unmask('dose')
            if not self._duration_mode:
                for op, d_mask in zip(ops, dose_masks):
                    if op(feature):
                        self.action_gen_set.mask('dose', d_mask)

                        return self.action_gen_set.make_generator()

            else:
                self.action_gen_set.unmask('duration')
                for op, d_mask, i_mask in zip(ops, dose_masks, duration_masks):
                    if op(feature):
                        self.action_gen_set.mask('dose', d_mask)
                        self.action_gen_set.mask('duration', i_mask)

                        return self.action_gen_set.make_generator()

                self.action_gen_set.mask('duration', duration_masks[-1])

            self.action_gen_set.mask('dose', dose_masks[-1])

            return self.action_gen_set.make_generator()

        caps = tuple(
            i for i in (5.0, 10.0, 15.0)
            if self._dose_range[0] <= i <= self._dose_range[1])
        max_cap = min(caps[-1], self._dose_range[1])

        dose = {
            cap: {
                d: cap
                for d in self.generate_dose_values(cap, max_cap, 0.5)
                if d > cap}
            for cap in caps}

        min_duration, max_duration = self._duration_range
        int_fixed = {
            d: {
                i: d
                for i in (self._duration_values or range(
                    min_duration, max_duration + 1,
                    self._duration_step))  # type: ignore
                if i != d}
            for d in (1, 2, 3, 7)}
        int_semi_free = {
            i: min_duration
            for i in (self._duration_values or range(
                min_duration, max_duration + 1,
                self._duration_step))  # type: ignore
            if i not in (1, 2, 3, 7, 14, 28)}
        int_weekly = {
            i: min_duration
            for i in (self._duration_values or range(
                min_duration, max_duration + 1,
                self._duration_step))  # type: ignore
            if i not in (7, 14, 21, 28)}

        # for cap in caps[:-1]:
        #     name = f'237_{int(cap):02}'
        #     if name not in current_action_definitions:
        #         self.possible_actions.add_definition(
        #             name, functools.partial(
        #                 _generate,
        #                 ops=(
        #                     lambda f: f['day'].value >= 5,
        #                     lambda f: f['day'].value == 2,
        #                     lambda f: f['day'].value == 0),
        #                 dose_masks=(
        #                     dose[max_cap], dose[max_cap], dose[cap]
        #                 ),
        #                 duration_masks=(
        #                     int_fixed[7], int_fixed[3], int_fixed[2])),
        #             'day')

        #     name = f'daily_{int(cap):02}'
        #     if name not in current_action_definitions:
        #         self.possible_actions.add_definition(
        #             name, functools.partial(
        #                 _generate,
        #                 ops=(lambda f: f['day'].value > 0,),
        #                 dose_masks=(dose[max_cap], dose[cap]),
        #                 duration_masks=(int_fixed[1], int_fixed[1])),
        #             'day')

        #     name = f'free_{int(cap):02}'
        #     if name not in current_action_definitions:
        #         self.possible_actions.add_definition(
        #             name, functools.partial(
        #                 _generate,
        #                 ops=(lambda f: f['day'].value > 0,),
        #                 dose_masks=(dose[max_cap], dose[cap]),
        #                 duration_masks=({}, {})),
        #             'day')

        #     name = f'semi_{int(cap):02}'
        #     if name not in current_action_definitions:
        #         self.possible_actions.add_definition(
        #             name, functools.partial(
        #                 _generate,
        #                 ops=(lambda f: f['day'].value > 0,),
        #                 dose_masks=(dose[max_cap], dose[cap]),
        #                 duration_masks=(int_semi_free, int_semi_free)),
        #             'day')

        #     name = f'weekly_{int(cap):02}'
        #     if name not in current_action_definitions:
        #         self.possible_actions.add_definition(
        #             name, functools.partial(
        #                 _generate,
        #                 ops=(lambda f: f['day'].value > 0,),
        #                 dose_masks=(dose[max_cap], dose[cap]),
        #                 duration_masks=(int_weekly, int_weekly)),
        #             'day')

        if name == '237_15':
            return (
                functools.partial(
                    _generate,
                    ops=(
                        lambda f: f['day'].value >= 5,  # type: ignore
                        lambda f: f['day'].value == 2,
                        lambda f: f['day'].value == 0),
                    dose_masks=(
                        dose[max_cap], dose[max_cap], dose[max_cap]
                    ),
                    duration_masks=(
                        int_fixed[7], int_fixed[3], int_fixed[2])),
                'day')

        if name == 'daily_15':
            return (
                functools.partial(
                    _generate, ops=(),
                    dose_masks=(dose[max_cap],),
                    duration_masks=(int_fixed[1],)),
                'day')

        if name == 'free_15':
            return (
                functools.partial(
                    _generate, ops=(),
                    dose_masks=(dose[max_cap],), duration_masks=({},)),
                'day')

        if name == 'semi_15':
            return (
                functools.partial(
                    _generate, ops=(),
                    dose_masks=(dose[max_cap],),
                    duration_masks=(int_semi_free,)),
                'day')

        if name == 'weekly_15':
            return (
                functools.partial(
                    _generate, ops=(lambda f: f['day'].value > 0,),  # type: ignore
                    dose_masks=(dose[max_cap],),
                    duration_masks=(int_weekly,)),
                'day')

        if name == 'delta':
            def delta_dose(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_change')
                self.action_gen_set.unmask('duration')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                day: int = feature['day'].value  # type: ignore
                min_dose, max_dose = self._dose_range
                min_delta = min_dose - last_dose
                max_delta = max_dose - last_dose
                d_list = self.generate_dose_values(min_dose, max_dose)
                d_list = set((*d_list, *(-x for x in d_list)))
                d_mask = {
                    d: min_delta if d < min_delta else max_delta
                    for d in d_list
                    if not (min_delta <= d <= max_delta)
                }
                self.action_gen_set.mask('dose_change', d_mask)

                if day >= 5:
                    duration_mask = int_fixed[7]
                elif day == 2:
                    duration_mask = int_fixed[3]
                elif day == 0:
                    duration_mask = int_fixed[2]
                else:
                    raise ValueError(f'wrong day: {day}.')

                self.action_gen_set.mask('duration', duration_mask)

                return self.action_gen_set.make_generator()

            return delta_dose, 'day_and_last_dose'

        if name == 'percent_duration':
            def percent_dose(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_percent_change')
                self.action_gen_set.unmask('duration')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                min_dose, max_dose = self._dose_range
                all_ps: tuple[float, ...] = \
                    self.feature_gen_set['dose_percent_change'].fixed_values  # type: ignore
                permissibles = [
                    p for p in all_ps
                    if (min_dose <= last_dose * (1 + p) <= max_dose)
                ]
                min_p = min(permissibles)
                max_p = max(permissibles)
                p_mask = {
                    p: min_p if p < min_p else max_p
                    for p in all_ps
                    if p not in permissibles
                }
                self.action_gen_set.mask('dose_percent_change', p_mask)

                # if day >= 5:
                #     duration_mask = int_fixed[7]
                #     self.action_gen_set.mask('duration', duration_mask)

                return self.action_gen_set.make_generator()

            return percent_dose, 'day_and_last_dose'

        if name == 'semi':
            def semi(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('duration')
                self.action_gen_set.mask('duration', int_semi_free)

                return self.action_gen_set.make_generator()

            return semi, 'day'

        if name == 'percent':
            def percent_dose(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_percent_change')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                min_dose, max_dose = self._dose_range
                all_ps: tuple[float, ...] = \
                    self.feature_gen_set['dose_percent_change'].fixed_values  # type: ignore
                permissibles = [
                    p for p in all_ps
                    if (min_dose <= last_dose * (1 + p) <= max_dose)
                ]
                min_p = min(permissibles)
                max_p = max(permissibles)
                p_mask = {
                    p: min_p if p < min_p else max_p
                    for p in all_ps
                    if p not in permissibles
                }
                self.action_gen_set.mask('dose_percent_change', p_mask)

                return self.action_gen_set.make_generator()

            return percent_dose, 'day_and_last_dose'

        if name == 'percent_guided':
            def percent_dose_guided(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_percent_change')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                last_INR: float = \
                    feature['daily_INR_history'].value[-1]  # type: ignore
                min_dose, max_dose = self._dose_range
                all_ps: tuple[float, ...] = \
                    self.feature_gen_set['dose_percent_change'].fixed_values  # type: ignore
                permissibles = [
                    p for p in all_ps
                    if (min_dose <= last_dose * (1 + p) <= max_dose)
                ]
                if last_INR > 3.0:
                    permissibles = [
                        p for p in permissibles
                        if p <= 0.0
                    ]
                elif last_INR < 2.0:
                    permissibles = [
                        p for p in permissibles
                        if p >= 0.0
                    ]
                min_p = min(permissibles)
                max_p = max(permissibles)
                p_mask = {
                    p: min_p if p < min_p else max_p
                    for p in all_ps
                    if p not in permissibles
                }
                self.action_gen_set.mask('dose_percent_change', p_mask)

                return self.action_gen_set.make_generator()

            return percent_dose_guided, 'day_and_last_dose_INR'

        if name == 'percent_semi':
            def percent_dose(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_percent_change')
                self.action_gen_set.unmask('duration')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                # day: int = feature['day'].value  # type: ignore
                min_dose, max_dose = self._dose_range
                all_ps: tuple[float, ...] = \
                    self.feature_gen_set['dose_percent_change'].fixed_values  # type: ignore
                permissibles = [
                    p for p in all_ps
                    if (min_dose <= last_dose * (1 + p) <= max_dose)
                ]
                min_p = min(permissibles)
                max_p = max(permissibles)
                p_mask = {
                    p: min_p if p < min_p else max_p
                    for p in all_ps
                    if p not in permissibles
                }
                self.action_gen_set.mask('dose_percent_change', p_mask)
                self.action_gen_set.mask('duration', int_semi_free)

                return self.action_gen_set.make_generator()

            return percent_dose, 'day_and_last_dose'

        if name == 'percent_semi_joint':
            def percent_dose(feature: FeatureSet) -> FeatureGeneratorType:
                self.action_gen_set.unmask('dose_percent_change_joint')
                last_dose: float = \
                    feature['daily_dose_history'].value[-1]  # type: ignore
                # day: int = feature['day'].value  # type: ignore
                min_dose, max_dose = self._dose_range
                all_ps: list[float]
                durations: list[int]
                all_ps, durations = tuple(
                    zip(*self.feature_gen_set['dose_percent_change_joint'].categories))
                permissibles = [
                    p for p in all_ps
                    if (min_dose <= last_dose * (1 + p) <= max_dose)
                ]
                min_p = min(permissibles)
                max_p = max(permissibles)
                p_mask = {
                    (p, d): min_p if p < min_p else max_p
                    for p, d in zip(all_ps, durations)
                    if p not in permissibles
                }
                self.action_gen_set.mask(
                    'dose_percent_change_joint', p_mask)

                return self.action_gen_set.make_generator()

            return percent_dose, 'day_and_last_dose'

    def _reward_def_reference(
        self, name: str
    ) -> tuple[reil_functions.ReilFunction, str] | None:
        try:
            return reward_definitions[name]
        except KeyError:
            return super()._reward_def_reference(name)

    def _statistic_def_reference(self, name: str):
        if name == 'PTTR_exact_basic':
            return statistic_PTTR, 'daily_INR', 'patient_w_sensitivity_basic'

        if name == 'PTTR_exact':
            return statistic_PTTR, 'daily_INR', 'patient_w_sensitivity'

    def _sub_comp_age(self, _id: int, **kwargs: Any) -> Feature:
        return super()._numerical_sub_comp('age')

    def _sub_comp_weight(self, _id: int, **kwargs: Any) -> Feature:
        return self._numerical_sub_comp('weight')

    def _sub_comp_height(self, _id: int, **kwargs: Any) -> Feature:
        return self._numerical_sub_comp('height')

    def _sub_comp_gender(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('gender')

    def _sub_comp_race(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('race')

    def _sub_comp_tobaco(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('tobaco')

    def _sub_comp_amiodarone(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('amiodarone')

    def _sub_comp_fluvastatin(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('fluvastatin')

    def _sub_comp_CYP2C9(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('CYP2C9')

    def _sub_comp_CYP2C9_masked(
            self, _id: int, days: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('CYP2C9', self._day < days)

    def _sub_comp_VKORC1(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('VKORC1')

    def _sub_comp_VKORC1_masked(
            self, _id: int, days: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('VKORC1', self._day < days)

    def _sub_comp_sensitivity(self, _id: int, **kwargs: Any) -> Feature:
        return self._categorical_sub_comp('sensitivity')

    def _sub_comp_INR_history(
            self, _id: int, length: int = 1, **kwargs: Any
    ) -> Feature:
        return self._sub_comp_measurement_history(
            _id, length, backfill=self._backfill, **kwargs)

    def _sub_comp_daily_INR_history(
            self, _id: int, length: int = 1, **kwargs: Any
    ) -> Feature:
        return self._sub_comp_daily_measurement_history(
            _id, length, backfill=self._backfill, **kwargs)

    def _sub_comp_INR_within(
            self, _id: int, length: int = 1, **kwargs: Any
    ) -> Feature:
        durations = self._get_history('duration_history', length).value
        return self._get_history(
            'daily_INR_history', sum(durations))  # type: ignore

    def _sub_comp_consecutive_in_range(
            self, _id: int, lo: float = 2.0, hi: float = 3.0,
            at_decision: bool = False, **kwargs: Any
    ) -> Feature:
        '''Consecutive in-range (INR in [lo, hi]) days -- the stability signal
        `s` (a daily-INR-derived proxy for Aurora's `number_of_stable_days`).

        Two evaluation contexts, selected by `at_decision`:

        * `at_decision=False` (default -- REWARD use): the reward runs AFTER
          `_take_effect` advanced `self._day` to the END of the just-chosen
          window, so counting from `self._day` would fold in the current
          window's outcome and retroactively excuse an over-long interval that
          happened to stay in range. We start the backward count from the
          DECISION day, `self._day - <current window duration>`, so `s`
          reflects stability BEFORE this interval was chosen.

        * `at_decision=True` (OBSERVATION use -- the Paper-3 stability-augmented
          policy state): the state is queried at decision time, BEFORE
          `_take_effect`, so `self._day` IS the decision day and the daily INRs
          up to it are already recorded. Counting from `self._day` (offset 0) is
          the correct decision-time `s` -- the same value the reward will see
          for THIS decision one iteration later. Using the reward offset here
          would return a stale `s` (one decision behind).

        Read-only over `_full_measurement_history`; no state mutation.
        '''
        if at_decision:
            start = self._day
        else:
            idx = self._decision_points_index
            cur_dur = (self._decision_points_duration_history[idx - 1]
                       if idx > 0 else 0)
            start = self._day - cur_dur
        hist = self._full_measurement_history
        s = 0
        for i in range(start, 0, -1):
            if lo <= hist[i] <= hi:
                s += 1
            else:
                break
        return self.feature_gen_set['consecutive_in_range'](
            value=min(s, self._max_day))

    def _sub_comp_extrap_exit(
            self, _id: int, lo: float = 2.0, hi: float = 3.0, **kwargs: Any
    ) -> Feature:
        '''Linear-extrapolation expected exit-day (Paper-3, doc 220 §9.5).

        Extrapolate the recent INR trend to the nearest range edge:
          velocity = (INR_now - INR_prev_decision) / last_interval_tau
          exit = (hi - INR_now)/velocity      if rising  (velocity > 0)
               = (INR_now - lo)/|velocity|     if falling (velocity < 0)
               = max_day (cap)                 if ~flat
               = 0                             if already out of range.
        A model-free tau* estimate the policy can use to pick the interval.
        Read-only over `_full_measurement_history` / `_decision_points_*`; queried
        at decision time (`self._day` is the decision day). Bounded [0, max_day].
        '''
        cap = float(self._max_day)
        hist = self._full_measurement_history
        day = self._day
        inr2 = hist[day]
        idx = self._decision_points_index
        last_tau = (self._decision_points_duration_history[idx - 1]
                    if idx > 0 else 0)
        if last_tau <= 0 or day - last_tau < 0:
            v = 0.0
        else:
            v = (inr2 - hist[day - last_tau]) / last_tau
        if inr2 < lo or inr2 > hi:
            e = 0.0
        elif v > 1e-3:
            e = (hi - inr2) / v
        elif v < -1e-3:
            e = (inr2 - lo) / (-v)
        else:
            e = cap
        return self.feature_gen_set['extrap_exit'](
            value=max(0.0, min(e, cap)))
