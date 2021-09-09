# -*- coding: utf-8 -*-
'''
HambergPKPD class
=================

A warfarin PK/PD model proposed by Hamberg et al. (2010).
DOI: 10.1038/clpt.2010.37
'''
import math
from typing import (Any, Callable, Dict, Final, Iterable, List, NamedTuple,
                    NewType, Union)

import numpy as np
from reil.datatypes.feature import Feature
from reil.healthcare.mathematical_models.health_math_model import \
    HealthMathModel

Day = NewType('Day', float)
Hour = NewType('Hour', int)
dT = NewType('dT', int)


class DoseEffect(NamedTuple):
    dose: float
    Cs: Callable[[Iterable[dT]], List[float]]


class HambergPKPD2010(HealthMathModel):
    '''
    Hamberg PK/PD model for warfarin.
    '''
    _per_hour: int = 1

    # Hamberg et al. (2010) - Table 2
    # CL_s: Apparent oral clearance
    _CL_alleles: Final = {  # Effect of genotypes diff. from *1/*1 on CL_s
        '*1': 0.174,
        '*2': 0.0879,
        '*3': 0.0422}
    _CL_age: Final = -0.571  # Effect of age on CL_s centered around 71 years
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

    # Hamberg et al. (2010) - Misc.
    _INR_max: Final = 20.0  # page 733

    def __init__(
            self, randomized: bool = True,
            cache_size: Day = Day(30)) -> None:
        """
        Arguments
        ---------
        randomized:
            Whether to have random effects in patient response to warfarin.

        cache_size:
            Size of the cache used to store pre-computed values needed for
            INR computation.
        """
        self._randomized = randomized
        self._cache_size = math.ceil(cache_size)
        self._last_computed_day: Day = Day(0)
        self._cached_cs: Dict[float, List[float]] = {}

    def setup(self, **arguments: Feature) -> None:
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
        # Note: In Hamberg et al. (2007), BASE_i is the measured baseline INR
        # for patients, but Ravvaz fixed it to 1.
        self._baseINR = 1.0    # Ravvaz source code

        age = float(arguments['age'].value)  # type: ignore
        CYP2C9 = str(arguments['CYP2C9'].value)
        MTT_1 = float(arguments['MTT_1'].value)  # type: ignore
        MTT_2 = float(arguments['MTT_2'].value)  # type: ignore
        V = float(arguments['V'].value)  # type: ignore
        EC_50 = float(arguments['EC_50'].value)  # type: ignore
        CL_S = float(arguments['CL_S'].value)  # type: ignore

        CYP_alleles = CYP2C9.split('/')
        if (CYP_alleles[0] not in self._CL_alleles
                or CYP_alleles[1] not in self._CL_alleles):
            raise ValueError('Unknown alleles!')

        CL_s = CL_S * (
            1.0 - (self._CL_s_age * (age - 71.0))
        ) * (
            1 - self._CL_alleles[CYP_alleles[0]]
            + self._CL_alleles[CYP_alleles[1]])

        self._KDE = CL_s / V

        self._ktr = np.array([[3.0/MTT_1], [3.0/MTT_2]])  # type: ignore
        self._EC_50_gamma = EC_50 ** self._gamma

        self._dose_records: Dict[Day, DoseEffect] = {}
        cs_size = self._cache_size * 24 * self._per_hour
        self._total_cs = np.array([0.0] * cs_size)  # type: ignore  hourly
        self._computed_INRs: Dict[Day, float] = {}  # daily
        self._err_list: List[List[float]] = []  # hourly
        self._err_ss_list: List[List[float]] = []  # hourly
        self._exp_e_INR_list: List[List[float]] = []  # daily

        day_0 = Day(0)
        dt_0 = dT(0)
        self._last_computed_day = day_0

        temp_cs_generator = self._CS_function_generator(dt_0, 1.0)
        self._cached_cs = {
            1.0: temp_cs_generator(range(cs_size))}  # type: ignore

        self._C = np.array([0.0] + [1.0] * 8)  # type: ignore
        self._A: float = 0.0  # TODO: Make sure this is correct!

        # running _err to initialize their cache for reproducibility purposes
        self._err(dt_0, False)
        self._err(dt_0, True)
        self._computed_INRs[day_0] = self._INR(self._C, day_0)

    def run(self, **inputs: Any) -> Dict[str, Any]:
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
        self.dose = inputs.get('dose', {})

        if days := inputs.get('measurement_days'):
            return {'INR': self.INR(days)}

        return {'INR': {}}

    @property
    def dose(self) -> Dict[Day, float]:
        '''
        Return doses for each day.

        Returns
        -------
        :
            A dictionary with days as keys and doses as values.
        '''
        return {t: info.dose
                for t, info in self._dose_records.items()}

    @dose.setter
    def dose(self, dose: Dict[Day, float]) -> None:
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
                self._last_computed_day = Day(0)
        except ValueError:  # no doses
            pass

        for day, _dose in dose.items():
            if _dose != 0.0:
                dt = dT(math.ceil(day * 24 * self._per_hour))
                if day in self._dose_records:
                    # TODO: Implement!
                    raise NotImplementedError

                self._dose_records[day] = DoseEffect(
                    _dose, self._CS_function_generator(dt, _dose))

                self._total_cs += np.array(  # type: ignore
                    self._dose_records[day].Cs(range(
                        self._cache_size * 24 * self._per_hour)  # type: ignore
                    )
                )

    def INR(self, measurement_days: Union[Day, List[Day]]) -> List[float]:
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
        days: List[Day]

        days = (measurement_days if hasattr(measurement_days, '__iter__')
                else [measurement_days])  # type: ignore

        not_computed_days = set(days).difference(self._computed_INRs)
        if (not_computed_days and
                min(not_computed_days) < self._last_computed_day):
            self._last_computed_day = Day(0)
            self._computed_INRs = {}
            not_computed_days = days

        if self._last_computed_day == 0:
            self._C = np.array(  # type: ignore
                [[0.0] * 3, [0.0] * 3], np.float)

        stop_points = [self._last_computed_day] + list(not_computed_days)
        for d1, d2 in zip(stop_points[:-1], stop_points[1:]):
            delta_Ts = 24 * self._per_hour
            for dt in range(int(d1 * delta_Ts), int(d2 * delta_Ts)):
                self._C[:, 0] = self._inflow(dt)  # type: ignore
                self._C[:, 1:] += self._ktr * (
                    self._C[:, :-1] - self._C[:, 1:]) / self._per_hour

            self._computed_INRs[d2] = self._INR(self._C, Day(int(d2)))

        self._last_computed_day = stop_points[-1]

        return [self._computed_INRs[i] for i in days]

    def _CS_function_generator(
            self, dt_dose: dT, dose: float
    ) -> Callable[[Iterable[dT]], List[float]]:
        '''
        Generate a Cs function.

        Arguments
        ---------
        dt_dose:
            The time in which the dose is administered.

        dose:
            The value of the dose administered.

        Returns
        -------
        :
            A function that gets the time and returns that time's
            warfarin concentration.

        Notes
        -----
        To speed up the process, the generated function uses a pre-computed
        cache of concentrations and only computes the concentration
        if the requested day is beyond the cached range.
        '''
        if dose == 1.0:
            cached_cs_temp = []
        else:
            if dose not in self._cached_cs:
                self._cached_cs[dose] = [dose * cs
                                         for cs in self._cached_cs[1.0]]
            cached_cs_temp = self._cached_cs[dose]

        def Cs(dts: Iterable[dT]) -> List[float]:
            '''
            Get delta_t list and return the warfarin concentration of
            those times.

            Arguments
            ---------
            dts:
                The times for which concentration value is needed.

            Returns
            -------
            :
                Warfarin concentration
            '''
            max_diff = len(cached_cs_temp)
            coef_alpha = self._coef_alpha
            coef_beta = self._coef_beta
            coef_k_a = self._coef_k_a
            alpha = self._alpha
            beta = self._beta
            k_aS = self._k_aS

            # For hour_diff == 0, the Cs equation itself is zero, so included
            # it in the main if to avoid unnecessary computation.
            return [
                0.0 if (dt_diff := dt - dt_dose) <= 0
                else (
                    cached_cs_temp[dt_diff] if dt_diff < max_diff
                    else (
                        coef_alpha *
                        math.exp(-alpha * dt_diff / self._per_hour) +
                        coef_beta *
                        math.exp(-beta * dt_diff / self._per_hour) +
                        coef_k_a *
                        math.exp(-k_aS * dt_diff / self._per_hour)
                    ) * dose
                ) for dt in dts]

        return Cs

    def _err(self, dt: dT, ss: bool = False) -> float:
        '''
        Generate error term for the requested day.

        Arguments
        ---------
        dt:
            The time for which the error is requested.

        ss:
            Whether the error is for the steady-state case. For single dose
            it should be `False`, and for multiple doses, it should be `True`.

        Returns
        -------
        :
            The error value.

        Notes
        -----
        To speed up the process and generate reproducible results in each run,
        the errors are cached in batches.
        For each call of the function, the cached error is returned. If
        the `hour` is beyond the cached range, a new range of error values
        are generated and added to the cache.
        Also, error is per hour. So, for any fraction of hour, the same
        value will be returned.
        '''
        if self._randomized:
            h = dt // self._per_hour
            hourly_cache_size = self._cache_size * 24
            index_0 = h // hourly_cache_size
            index_1 = h % hourly_cache_size
            e_list = self._err_ss_list if ss else self._err_list
            try:
                return e_list[index_0][index_1]
            except IndexError:
                missing_rows = index_0 - len(e_list) + 1
                stdev = self._sigma_ss if ss else self._sigma_s
                for _ in range(missing_rows):
                    e_list.append(np.exp(np.random.normal(  # type:ignore
                        0, stdev, hourly_cache_size)))

            return e_list[index_0][index_1]
        else:
            return 1.0

    def _exp_e_INR(self, d: Day) -> float:
        '''
        Generate exp(error) term of INR for the requested day.

        Arguments
        ---------
        d:
            The day for which the error is requested.

        Returns
        -------
        :
            The error value.

        Notes
        -----
            error is generated per day. So, if the given `d` is fractional,
            it will be truncated and the respective error is returned.

            To speed up the process and generate reproducible results in each
            run, the errors are cached in batches.
            For each call of the function, the cached error is returned. If
            the `day` is beyond the cached range, a new range of error values
            are generated and added to the cache.
        '''
        _d = int(d)
        if self._randomized:
            index_0 = _d // self._cache_size
            index_1 = _d % self._cache_size
            try:
                return self._exp_e_INR_list[index_0][index_1]
            except IndexError:
                missing_rows = index_0 - len(self._exp_e_INR_list) + 1
                for _ in range(missing_rows):
                    self._exp_e_INR_list.append(
                        np.exp(np.random.normal(  # type:ignore
                            0, self._sigma_INR,
                            self._cache_size)))

            return self._exp_e_INR_list[index_0][index_1]

        else:
            return 1.0

    def _inflow(self, t: dT) -> float:
        '''
        Compute the warfarin concentration that enters the two compartments
        in the PK/PD model.

        Arguments
        ---------
        t:
            The time for which the input is requested.
            t = Hours * _per_hour + delta_t

        Returns
        -------
        :
            The input value.

        Notes
        -----
        To speed up the process, total concentration is being cached for a
        number of days. For days beyond this range, concentration values are
        computed and used on each call.
        '''
        try:
            Cs = self._total_cs[t]
        except IndexError:
            print('compute Cs')
            Cs = sum(v.Cs([t])[0]
                     for v in self._dose_records.values())

        Cs_gamma = (Cs * self._err(t, t > 0)) ** self._gamma
        inflow = 1 - (
            (self._E_max * Cs_gamma) / (self._EC_50_gamma + Cs_gamma))

        return inflow

    def _INR(self, C: np.ndarray, d: Day) -> float:
        '''
        Compute the INR on day `d`.

        Arguments
        ---------

        d:
            The day for which the input is requested.

        Returns
        -------
        :
            The INR value.

        Notes
        -----
        To speed up the process, total concentration is being cached for a
        number of days. For days beyond this range, concentration values are
        computed and used on each call.
        '''

        # Note: we defined `C` in such a way to compute changes in `C`s
        # easier. In our implementation, `C*4` is the `C*3` in Hamberg et al.
        return (
            self._baseINR + (
                self._INR_max * (1 - (C[0, 2] + C[1, 2])/2))
        ) * self._exp_e_INR(d)
