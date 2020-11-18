import math
from collections import namedtuple
from typing import Any, Callable, Dict, List, Union

import numpy as np
from reil.subjects.healthcare.mathematical_models import HealthMathModel

DoseEffect = namedtuple('DoseEffect', ['dose', 'Cs'])


class HambergPKPD(HealthMathModel):
    '''
    Hamberg PK/PD model for warfarin.
    '''
    def __init__(self, randomized: bool = True, cache_size: int = 30) -> None:
        self._randomized = randomized
        self._d2h = 24

        self._gamma = 0.424  # no units
        self._Q = 0.131    # (L/h)
        self._lambda = 3.61
        self._E_max = 1	 # no units

        self._cache_size = cache_size
        self._cached_cs = []

    def setup(self, **arguments):
        age: float = arguments['age'].value  # type:ignore
        CYP2C9 = str(arguments['CYP2C9'].value)

        # VKORC1 is not used inside this implementation of Hamberg. It specifies
        # the distribution of EC_50 which is now being determined in WarfarinPatient
        # class.
        # self._VKORC1 = str(arguments['VKORC1'].value)

        MTT_1: float = arguments['MTT_1'].value  # type:ignore
        MTT_2: float = arguments['MTT_2'].value  # type:ignore

        self._EC_50_gamma: float = arguments['EC_50'].value ** self._gamma

        cyp_1_1: float = arguments['cyp_1_1'].value  # type:ignore
        V1: float = arguments['V1'].value  # type:ignore
        V2: float = arguments['V2'].value  # type:ignore

        ktr1 = 6/MTT_1  # type:ignore  # 1/hours; changed from 1/MTT_1
        ktr2 = 1/MTT_2  # type:ignore  # 1/hours
        self._ktr = np.array([0.0] + [ktr1] * 6 + [0.0, ktr2])

        Q = 0.131    # (L/h)

        # bioavilability fraction 0-1 (from: "Applied Pharmacokinetics & Pharmacodynamics 4th edition, p.717", some other references)
        F = 0.9

        self._ka = 2  # absorption rate (1/hr)

        temp = {'*1/*1': 0.0,
                '*1/*2': 0.315,
                '*1/*3': 0.453,
                '*2/*2': 0.722,
                '*2/*3': 0.69,
                '*3/*3': 0.852}

        if CYP2C9 not in temp:
            raise ValueError('The CYP2C9 genotype not recognized fool!')

        CL_s = 1.0 - (0.0091 * (age - 71) if age > 71 else 0)
        CL_s *= cyp_1_1 * (1 - temp[CYP2C9])

        k12 = Q / V1
        k21 = Q / V2
        k10 = CL_s / V1

        b = k10 + k21 + k12
        c = k10 * k21
        self._alpha = (b + math.sqrt(b ** 2 - 4*c)) / 2
        self._beta = (b - math.sqrt(b ** 2 - 4*c)) / 2

        kaF_2V1 = (self._ka * F / 2) / V1
        self._coef_alpha = max(
            0.0, ((k21 - self._alpha) / ((self._ka - self._alpha)*(self._beta - self._alpha)))) * kaF_2V1
        self._coef_beta = max(
            0.0, ((k21 - self._beta) / ((self._ka - self._beta)*(self._alpha - self._beta)))) * kaF_2V1
        self._coef_k_a = max(0.0, ((k21 - self._ka) / ((self._ka - self._alpha)*(self._ka - self._beta)))
                             ) * kaF_2V1

        self._dose_records: Dict[int, DoseEffect] = {}
        self._total_cs = np.array(
            [0.0] * self._cache_size * self._d2h)  # hourly
        self._list_of_INRs = [0.0] * (self._cache_size + 1)  # daily
        self._err_list = []  # daily
        self._err_ss_list = []  # daily
        self._exp_e_INR_list = []  # daily
        self._last_computed_day: int = 0

        temp_cs_generator = self._CS_function_generator(0, 1.0)
        self._cached_cs = [temp_cs_generator(t)
                           for t in range(self._cache_size * self._d2h)]

        self._A = np.array([0.0] + [1.0] * 8)
        self._list_of_INRs[0] = self._INR(self._A, 0)

    def run(self, **inputs) -> Dict[str, Any]:
        '''
        inputs should include:
            - a dictionary called "dose" with days for each dose as keys and the amount of dose as values.
            - a list called "measurement_days" that shows INRs of which days should be returned.
        '''
        self.dose = inputs.get('dose', {})

        return {'INR': self.INR(inputs.get('measurement_days', []))}
        # if 'measurement_days' in inputs:
        #     hours = [d * self._d2h for d in inputs['measurement_days']]
        # else:
        #     hours = inputs.get('measurement_hours', [])

        # return {'INR': self.INR(hours)}

    @property
    def dose(self) -> Dict[int, float]:
        return dict((t, info.dose)
                    for t, info in self._dose_records.items())

    @dose.setter
    def dose(self, dose: Dict[int, float]) -> None:
        # if a dose is added ealier in the list, INRs should be updated all together.
        # because the history of "A" array is not kept.
        try:
            if self._last_computed_day > min(dose.keys()):
              self._last_computed_day = 0
        except ValueError:  # no doses
            pass

        for t, d in dose.items():
            if d != 0.0:
                h = t * self._d2h
                if t in self._dose_records:
                    self._total_cs -= (np.array([0.0]*h + self._cached_cs[:-h])
                                       * self._dose_records[t])

                self._dose_records[t] = DoseEffect(
                    d, self._CS_function_generator(h, d))

                try:
                    self._total_cs += np.array([0.0]*h + self._cached_cs[:-h])[:self._cache_size * self._d2h] * d
                except ValueError:  #_t == 0
                    self._total_cs += np.array(self._cached_cs) * d

    def INR(self, measurement_days: Union[int, List[int]]) -> List[float]:
        if isinstance(measurement_days, int):
            days = [measurement_days]
        else:
            days = measurement_days

        if self._last_computed_day == 0:
            self._A = np.array([0.0] + [1.0] * 8)

        max_days = max(days)
        self._list_of_INRs.extend(
            [0.0] * (max_days - len(self._list_of_INRs) + 1))

        for d in range(self._last_computed_day, max_days):
            for i in range(d * 24, (d + 1) * 24):
                self._A[0] = self._A[7] = self._inflow(i)
                self._A[1:] += self._ktr[1:] * (self._A[0:-1] - self._A[1:])
            self._list_of_INRs[d + 1] = self._INR(self._A, d + 1)

        self._last_computed_day = max_days

        return [self._list_of_INRs[i] for i in days]

    def _CS_function_generator(self, t_dose: int, dose: float) -> Callable[[int], float]:
        cached_cs_temp = [dose * cs for cs in self._cached_cs]
        def Cs(t: int) -> float:
            if t <= t_dose:
                return 0.0

            try:
                return cached_cs_temp[t - t_dose]
            except IndexError:
                return (self._coef_alpha * math.exp(-self._alpha * (t - t_dose)) +
                        self._coef_beta * math.exp(-self._beta * (t - t_dose)) +
                        self._coef_k_a * math.exp(-self._ka * (t - t_dose))) * dose

        return Cs

    def _err(self, t: int, ss: bool=False) -> float:
        if self._randomized:
            index_0 = t // self._cache_size
            index_1 = t % self._cache_size
            e_list = self._err_ss_list if ss else self._err_list
            try:
                return e_list[index_0][index_1]
            except IndexError:
                stdev = 0.30 if ss else 0.09
                e_list.append(np.exp(np.random.normal(  # type:ignore
                    0, stdev, self._cache_size)))

            return e_list[index_0][index_1]
        else:
            return 1.0

    def _exp_e_INR(self, t: int) -> float:
        if self._randomized:
            index_0 = t // self._cache_size
            index_1 = t % self._cache_size
            try:
                return self._exp_e_INR_list[index_0][index_1]
            except IndexError:
                self._exp_e_INR_list.append(np.exp(np.random.normal(  # type:ignore
                    0, 0.0325, self._cache_size)))

            return self._exp_e_INR_list[index_0][index_1]

        else:
            return 1.0

    def _inflow(self, t: int) -> float:
        try:
            Cs = self._total_cs[t]
        except IndexError:
            Cs = sum(v.Cs(t)
                    for v in self._dose_records.values())

        Cs_gamma = (Cs * self._err(t, t > 0)) ** self._gamma
        inflow = 1 - self._E_max * \
            Cs_gamma / (self._EC_50_gamma + Cs_gamma)

        return inflow

    def _INR(self, A: List[float], t: int) -> float:
        INR_max = 20
        baseINR = 1

        return (baseINR +
                (INR_max*(1-A[6]*A[8]) ** self._lambda)
               ) * self._exp_e_INR(t)
