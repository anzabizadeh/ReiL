# -*- coding: utf-8 -*-
# type: ignore
'''
HambergPKPD class
=================

A warfarin PK/PD model proposed by Hamberg et al. (2010).
DOI: 10.1038/clpt.2010.37
'''
import logging
import math
from typing import Any, Final

import numpy as np
from scipy.signal import lfilter

import reil
from reil.datatypes.feature import (Feature, FeatureGenerator,
                                    FeatureGeneratorSet, FeatureSet)
from reil.healthcare.mathematical_models.health_math_model import \
    HealthMathModel
from reil.utils.functions import random_lognormal_truncated

_log = logging.getLogger(__name__)


class HambergPKPD2010(HealthMathModel):
    '''
    Hamberg PK/PD model for warfarin.
    '''
    # region: class attributes
    _per_day: Final[int] = 24

    # Hamberg et al. (2010) - Table 2
    # CL: Apparent oral clearance
    _CL_alleles: Final = {
        '*1': 0.174,
        '*2': 0.0879,
        '*3': 0.0422}
    _CL_age: Final = -0.00571  # Effect of age on CL centered around 71 years
    _V: Final = 14.3  # (l) Apparent central volume of distribution
    _k_a: Final = 2.0  # (1/hr) Absorption rate constant

    _eta_CL: Final = 0.089  # Interindividual variability for CL
    _eta_V: Final = 0.054  # Interindividual variability for V

    _epsilon_s: Final = 0.099  # Residual error S-warfarin

    # Hamberg et al. (2010) - Table 4
    # There are three sets of numbers, Section RESULTS - Available Data states
    # that dataset C is used for final model parameters. So, we use its values
    # here.
    _E_max: Final = 1.0
    _gamma: Final = 1.15
    _EC_50_G: Final = 2.05  # (mg/l) EC_50 for VKORC1 allele G
    _EC_50_A: Final = 0.96  # (mg/l) EC_50 for VKORC1 allele A
    _MTT_1: Final = 28.6  # (h) Mean Transit Time
    _MTT_2: Final = 118.3  # (h) Mean Transit Time

    _eta_EC_50: Final = 0.34  # Interindividual variability for EC_50
    _eta_KDE: Final = 0.589  # Interindividual variability for KDE
    _epsilon_INR: Final = 0.20  # Residual error for INR

    _INR_max: Final = 20.0

    # Note: In Hamberg et al. (2007), BASE_i is the measured baseline INR
    # for patients, but Ravvaz fixed it to 1.
    _baseINR: Final = 1.0    # Ravvaz source code

    __log_V = math.log(_V)
    __sqrt_eta_V = math.sqrt(_eta_V)
    __sqrt_eta_CL = math.sqrt(_eta_CL)
    __sqrt_eta_KDE = math.sqrt(_eta_KDE)
    __log_EC_50_GG = math.log(_EC_50_G * 2)
    __log_EC_50_GA = math.log(_EC_50_G + _EC_50_A)
    __log_EC_50_AA = math.log(_EC_50_A * 2)
    __sqrt_eta_EC_50 = math.sqrt(_eta_EC_50)

    __FeatureGenerator_EC_50_non_random = FeatureGenerator.continuous(
        name='EC_50',  # (mg/L) Hamberg PK/PD
        mean=1.0,
        stdev=__sqrt_eta_EC_50,
        generator=random_lognormal_truncated,
        randomized=False)

    __FeatureGenerator_EC_50_GG = FeatureGenerator.continuous(
        name='EC_50',  # (mg/L) Hamberg PK/PD
        mean=__log_EC_50_GG,
        stdev=__sqrt_eta_EC_50,
        generator=random_lognormal_truncated,
        randomized=True)

    __FeatureGenerator_EC_50_GA = FeatureGenerator.continuous(
        name='EC_50',  # (mg/L) Hamberg PK/PD
        mean=__log_EC_50_GA,
        stdev=__sqrt_eta_EC_50,
        generator=random_lognormal_truncated,
        randomized=True)

    __FeatureGenerator_EC_50_AA = FeatureGenerator.continuous(
        name='EC_50',  # (mg/L) Hamberg PK/PD
        mean=__log_EC_50_AA,
        stdev=__sqrt_eta_EC_50,
        generator=random_lognormal_truncated,
        randomized=True)

    _parameter_generators = FeatureGeneratorSet((
        FeatureGenerator.continuous(
            name='V',  # (L) Volume in central compartment
            mean=__log_V,
            stdev=__sqrt_eta_V,
            generator=random_lognormal_truncated,
            randomized=True),
        FeatureGenerator.continuous(
            name='eta_CL',
            mean=0.,
            stdev=__sqrt_eta_CL,
            generator=random_lognormal_truncated,
            randomized=True),
        FeatureGenerator.continuous(
            name='eta_KDE',
            mean=0.,
            stdev=__sqrt_eta_KDE,
            generator=random_lognormal_truncated,
            randomized=True),
    ))

    # endregion

    def __init__(
            self, randomized: bool = True,
            cache_size: int = 100) -> None:
        """
        Arguments
        ---------
        randomized:
            Whether to have random effects in patient response to warfarin.

        cache_size:
            Size of the cache used to store pre-computed values needed for
            INR computation. Default 100 covers the 90-day max trial without
            triggering `_expand_caches`; pre-2026-06 default was 30, which
            forced ≥2 expansions per patient on long trials.
        """
        self._randomized = randomized
        self._cache_size = math.ceil(cache_size)
        self._last_computed_day: int = 0
        self._cached_A: dict[float, list[float]] = {}
        self._age = None

    @classmethod
    def generate(
            cls,
            rnd_generators: reil.RandomGeneratorT,
            input_features: FeatureSet | None = None,
            **kwargs: Any) -> FeatureSet:
        with reil.random_generator_context(*rnd_generators):
            feature_set = super(HambergPKPD2010, cls).generate(
                rnd_generators=rnd_generators,
                input_features=input_features, **kwargs)

            temp = (input_features or {}).get('VKORC1')
            if temp is None:
                if 'EC_50' not in kwargs:
                    raise ValueError(
                        'Either input_features should contain VKORC1 or '
                        'EC_50 should be provided as a keyword argument.'
                    )
                feature_set += cls.__FeatureGenerator_EC_50_non_random(
                    kwargs['EC_50'])

            else:
                vkorc1 = temp.value
                if vkorc1 == 'G/G':
                    feature_set += cls.__FeatureGenerator_EC_50_GG()
                elif vkorc1 in ('G/A', 'A/G'):
                    feature_set += cls.__FeatureGenerator_EC_50_GA()
                else:  # 'A/A'
                    feature_set += cls.__FeatureGenerator_EC_50_AA()

        return feature_set

    def setup(
            self, rnd_generators: reil.RandomGeneratorT,
            input_features: FeatureSet | None = None) -> None:
        '''
        Set up the model.

        Arguments
        ---------
        arguments:
            `Feature` instances required to setup the model.

        Notes
        -----
        This model requires `age`, `CYP2C9`, `MTT_1`, `MTT_2`, `EC_50`,
        `cyp_1_1`, `V1`, and `V2`. The genotype of `VKORC1` is not directly
        used in this implementation. Instead, one should use it to generate
        `EC_50`. See `WarfarinPatient` class.

        Raises
        ------
        ValueError:
            `CYP2C9` is not one of the acceptable values:
            *1/*1, *1/*2, *1/*3, *2/*2, *2/*3, *3/*3
        '''

        if input_features is None:
            raise TypeError('input_features expected.')

        args = input_features.value

        self._age = float(args['age'])  # type: ignore
        self._CYP2C9 = str(args['CYP2C9'])
        self._V = float(args['V'])  # type: ignore
        self._EC_50 = float(args['EC_50'])  # type: ignore
        self._eta_CL = float(args['eta_CL'])  # type: ignore
        self._eta_KDE = float(args['eta_KDE'])  # type: ignore

        alleles = self._CYP2C9.split(sep='/')

        if any(a not in self._CL_alleles for a in alleles):
            raise ValueError('The CYP2C9 genotype not recognized!')

        CL = (
            sum(self._CL_alleles[a] for a in alleles) *
            (1.0 - (self._CL_age * (self._age - 71.0)))
        ) * self._eta_CL

        # -------- Implementation of the one compartment model ------------

        # Similar to "Equations to use for time points when absorption can
        # occur and compartment A is unsaturated" available at
        # http://www.rsc.org/suppdata/c7/md/c7md00586e/c7md00586e1.pdf
        # There, compartment B is equivalent of compartment 1 here.
        # Also, k_1 and k_4 are K_a, k_e, respectively.
        # Note that $x_B^0$ term is zero as described in that document.

        # bioavilability fraction 0-1 (from: "Applied Pharmacokinetics &
        # Pharmacodynamics 4th edition, p.717", some other references)
        # is 0.9. Ravvaz and our original PK/PD uses that number.
        # Here, we use F = 1.0 as seem to be the value used in the PK/PD
        # paper.
        F = 1.0  # 0.9

        self._k_e = CL / self._V  # Elimination rate constant (p. 733)

        # Note: Here we halved the value for KaF_2V1, because half of the
        # warfarin is S, and only S affects the INR.
        self._coef = (
            (self._k_a * F / 2) / (self._V * (self._k_e - self._k_a)))

        # ---- End of the implementation of the one compartment model -------

        self._KDE = self._k_e * self._eta_KDE
        self._EDK_50_gamma = (CL * self._EC_50) ** self._gamma

        ktr1 = 3.0 / self._MTT_1  # (1/hours)
        ktr2 = 3.0 / self._MTT_2  # (1/hours)
        self._ktr = np.array([ktr1, ktr2])

        self._C = np.ones([4, 2])
        self._last_computed_day = 0

        self._dose_records: dict[int, float] = {}
        self._computed_INRs: dict[int, float] = {}  # daily

        self._current_cache_size = self._cache_size
        detailed_cache_size = self._cache_size * self._per_day

        if self._randomized:
            self._random_generator = rnd_generators[1]

            def _gen_err():  # type: ignore
                return np.exp(  # type: ignore
                    self._random_generator.normal(
                        0.0, self._epsilon_s, detailed_cache_size))

            def _gen_exp_e_INR():  # type: ignore
                return (  # type: ignore
                    self._random_generator.normal(
                        0.0, self._epsilon_INR, self._cache_size))
        else:
            def _gen_exp_e_INR():
                return np.ones(shape=self._cache_size)

            def _gen_err():
                return np.ones(shape=detailed_cache_size)

        self._gen_err = _gen_err
        self._gen_exp_e_INR = _gen_exp_e_INR

        self._err_list = self._gen_err()  # hourly
        self._exp_e_INR_list = self._gen_exp_e_INR()  # daily

        self._cached_A = self._base_concentration(
            coef=self._coef, k_a=self._k_a, k_e=self._k_e,
            max_time=detailed_cache_size)
        self._total_A = np.array([0.0] * detailed_cache_size)  # hourly

    def _expand_caches(self, segment_count: int = 1):
        # See note in hamberg_pkpd.py: with the default `cache_size=100`
        # this should not fire on the Paper-2 path; the log helps catch
        # silent regressions.
        _log.debug(
            'HambergPKPD2010._expand_caches: segment_count=%d, '
            'current_cache_size=%d',
            segment_count, self._current_cache_size)
        per_day = self._per_day
        self._current_cache_size += self._cache_size * segment_count

        self._err_list = np.concatenate(
            [
                self._err_list,
                *(self._gen_err() for _ in range(segment_count))],
            axis=0)

        self._exp_e_INR_list = np.concatenate(
            [
                self._exp_e_INR_list,
                *(self._gen_exp_e_INR() for _ in range(segment_count))],
            axis=0)

        self._cached_A = self._base_concentration(
            coef=self._coef, k_a=self._k_a, k_e=self._k_e,
            max_time=self._current_cache_size * per_day)

        total_len = self._current_cache_size * per_day
        self._total_A = np.zeros(total_len)

        for day, dose in self._dose_records.items():
            pad = day * per_day
            n = total_len - pad
            if n > 0:
                self._total_A[pad:pad + n] += dose * self._cached_A[:n]

    def run(self, **inputs: Any) -> dict[str, Any]:
        '''
        Run the model.

        Arguments
        ---------
        inputs:
            - A dictionary called "dose" with days for each dose as keys and
              the amount of dose as values.
            - A list called "measurement_days" that shows INRs of which days
              should be returned.

        Returns
        -------
        :
            A dictionary with keyword "INR" and a list of doses for the
            specified days.
        '''
        self.prescribe(inputs.get('dose', {}))

        if days := inputs.get('measurement_days'):
            return {'INR': self.INR(days)}

        return {'INR': {}}

    @property
    def dose(self) -> dict[int, float]:
        '''
        Return doses for each day.

        Returns
        -------
        :
            A dictionary with days as keys and doses as values.
        '''
        return {t: dose
                for t, dose in self._dose_records.items()}

    def prescribe(self, dose: dict[int, float]) -> None:
        '''
        Add warfarin doses at the specified days.

        Arguments
        ---------
        dose:
            A dictionary with days as keys and doses as values.
        '''
        # if a dose is added ealier in the list, INRs should be updated all
        # together because the history of "A" array is not kept.
        try:
            if self._last_computed_day > min(dose.keys()):
                self._last_computed_day = 0
        except ValueError:  # no doses
            pass

        if max(dose or {0: 0.0}) >= self._current_cache_size:
            segment_count = (
                max(dose) - self._current_cache_size
            ) // self._cache_size + 1
            self._expand_caches(segment_count=segment_count)

        per_day = self._per_day
        total_len = self._total_A.shape[0]
        for day, _dose in dose.items():
            if _dose != 0.0:
                prev_dose = self._dose_records.get(day, 0.0)
                new_dose = prev_dose + _dose
                if new_dose < 0.0:
                    raise ValueError(
                        'Total dose for a day cannot be negative.')

                pad = day * per_day
                n = total_len - pad
                if n > 0:
                    self._total_A[pad:pad + n] += (
                        new_dose * self._cached_A[:n])

                self._dose_records[day] = new_dose

    def INR(self, measurement_days: int | list[int]) -> list[float]:
        '''
        Compute INR values for the specified days.

        Arguments
        ---------
        measurement_days:
            One of a list of all days for which INR should be computed.

        Returns
        -------
        :
            A list of INRs for the specified days.
        '''
        days: list[int]

        days = (measurement_days if hasattr(measurement_days, '__iter__')
                else [measurement_days])  # type: ignore

        if max(days) >= self._current_cache_size:
            segment_count = (
                max(days) - self._current_cache_size
            ) // self._cache_size + 1
            self._expand_caches(segment_count=segment_count)

        not_computed_days = set(days).difference(self._computed_INRs)
        if (not_computed_days and
                min(not_computed_days) < self._last_computed_day):
            self._last_computed_day: int = 0
            self._computed_INRs = {}
            not_computed_days = days

        if self._last_computed_day == 0:
            self._C = np.ones([4, 2])

        stop_points = [self._last_computed_day] + sorted(
            list(not_computed_days))
        self._last_computed_day = stop_points[-1]

        start_day = stop_points[0]
        per_day = HambergPKPD2010._per_day
        start_point = start_day * per_day
        end_point = stop_points[-1] * per_day

        C_init = self._C

        if end_point > start_point:
            all_points = slice(start_point, end_point)

            # NOTE: In the paper's formulation, DR = KDE * A
            # However, it does not produce the expected INR values.
            # Using DR = V * KDE * A seem to fix the issue.
            DR_gamma = np.power(  # type: ignore
                self._V * self._KDE * self._total_A[all_points]
                * self._err_list[all_points],
                HambergPKPD2010._gamma)
            E = 1.0 - np.divide(  # type: ignore
                HambergPKPD2010._E_max * DR_gamma,  # type: ignore
                self._EDK_50_gamma + DR_gamma)

            # Pre-vectorisation recurrence:
            #     C[0] = E[t]                            # override
            #     C[1:] += ktr * (C[:-1] - C[1:])         # RHS uses old C
            #     C = np.clip(C, 0, 1)                    # per-step
            # numpy evaluates the RHS using post-override C[0], so per chain k:
            #     C1_k = (1-r_k)*C1_k_old + r_k*E[t]              (direct)
            #     C2_k = (1-r_k)*C2_k_old + r_k*C1_k_old          (delayed)
            #     C3_k = (1-r_k)*C3_k_old + r_k*C2_k_old          (delayed)
            # ktr1=3/MTT_1≈0.105 and ktr2=3/MTT_2≈0.025 are CLASS CONSTANTS
            # (MTT_1, MTT_2 are not patient-specific in this model), so r ∈ [0,1]
            # for all patients and the convex-combination invariant keeps
            # C ∈ [0,1] given E ∈ [0,1] — the per-step clip is mathematically a
            # no-op and is safely subsumed by lfilter.
            ktr1 = float(self._ktr[0])
            ktr2 = float(self._ktr[1])
            a_pole_1 = [1.0, -(1.0 - ktr1)]
            a_pole_2 = [1.0, -(1.0 - ktr2)]
            b_delay_1 = [0.0, ktr1]
            b_delay_2 = [0.0, ktr2]

            # Chain k=0 (ktr1)
            C1_0, _ = lfilter(
                [ktr1], a_pole_1, E,
                zi=[(1.0 - ktr1) * float(C_init[1, 0])])
            C2_0, _ = lfilter(
                b_delay_1, a_pole_1, C1_0,
                zi=[(1.0 - ktr1) * float(C_init[2, 0])
                    + ktr1 * float(C_init[1, 0])])
            C3_0, _ = lfilter(
                b_delay_1, a_pole_1, C2_0,
                zi=[(1.0 - ktr1) * float(C_init[3, 0])
                    + ktr1 * float(C_init[2, 0])])

            # Chain k=1 (ktr2)
            C1_1, _ = lfilter(
                [ktr2], a_pole_2, E,
                zi=[(1.0 - ktr2) * float(C_init[1, 1])])
            C2_1, _ = lfilter(
                b_delay_2, a_pole_2, C1_1,
                zi=[(1.0 - ktr2) * float(C_init[2, 1])
                    + ktr2 * float(C_init[1, 1])])
            C3_1, _ = lfilter(
                b_delay_2, a_pole_2, C2_1,
                zi=[(1.0 - ktr2) * float(C_init[3, 1])
                    + ktr2 * float(C_init[2, 1])])
        else:
            E = None
            C3_0 = None
            C3_1 = None

        for d1, d2 in zip(stop_points[:-1], stop_points[1:]):
            if d2 == start_day:
                # Empty steps in the original loop → INR uses C_init
                C3_mean = 0.5 * (float(C_init[3, 0]) + float(C_init[3, 1]))
            else:
                k_end = d2 * per_day - 1 - start_point
                C3_mean = 0.5 * (
                    float(C3_0[k_end]) + float(C3_1[k_end]))  # type: ignore[index]
            self._computed_INRs[d2] = (
                HambergPKPD2010._baseINR +
                HambergPKPD2010._INR_max * (1.0 - C3_mean)
            ) + self._exp_e_INR_list[d2]

        return [self._computed_INRs[i] for i in days]

    @staticmethod
    def _base_concentration(coef, k_a, k_e, max_time):
        times = np.arange(max_time)
        # clipping to avoid underflow. We do not need to be that accurate!
        k_a_t = np.clip(k_a * times, a_min=None, a_max=100.0)
        k_e_t = np.clip(k_e * times, a_min=None, a_max=100.0)
        temp = np.exp(-k_a_t) - np.exp(-k_e_t)
        base_cs = coef * temp
        base_cs = np.clip(base_cs, a_min=0.0, a_max=None)

        return base_cs

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update({
            'randomized': self._randomized,
            'cache_size': self._cache_size})
        if self._age is not None:
            config.update(
                dict(
                    random_generator=self._random_generator,
                    age=self._age,
                    CYP2C9=self._CYP2C9,
                    MTT_1=self._MTT_1,
                    MTT_2=self._MTT_2,
                    V=self._V,
                    EC_50=self._EC_50,
                    eta_CL=self._eta_CL,
                    eta_KDE=self._eta_KDE))

        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        instance = cls(
            randomized=config['randomized'],
            cache_size=config['cache_size'])
        if 'age' in config:
            input_features = FeatureSet(
                Feature.categorical(name=name, value=config[name])
                for name in (
                    'age', 'CYP2C9', 'MTT_1', 'MTT_2',
                    'V', 'EC_50', 'eta_CL', 'eta_KDE'))
            instance.setup(
                rnd_generators=(
                    None, config['random_generator'], None),  # type: ignore
                input_features=input_features)

            instance.prescribe(config.get('dose', {}))

            return instance
