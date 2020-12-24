import math
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from reil.subjects.healthcare.mathematical_models import HealthMathModel


class HambergPKPDFixedMaxTime(HealthMathModel):
    def __init__(self,
                 randomized: bool = True,
                 max_time: int = 24,
                 dose_interval: int = 24,
                 lazy: bool = False) -> None:
        self._randomized = randomized
        self._max_time = max_time  # The last hour of experiment
        self._dose_interval = dose_interval
        self._lazy = lazy

        self._gamma = 0.424  # no units

        self._Q = 0.131    # (L/h)
        self._lambda = 3.61

        # bioavilability fraction 0-1 (from: "Applied Pharmacokinetics &
        # Pharmacodynamics 4th edition, p.717", some other references)
        self._F = 0.9

        self._ka = 2  # absorption rate (1/hr)
        self._E_MAX = 1	 # no units

        self._dose = {}

    def setup(self, **arguments):
        self._age: float = arguments['age'].value  # type:ignore
        self._CYP2C9 = str(arguments['CYP2C9'].value)
        self._VKORC1 = str(arguments['VKORC1'].value)

        self._MTT_1: float = arguments['MTT_1'].value  # type:ignore
        self._MTT_2: float = arguments['MTT_2'].value  # type:ignore

        self._EC_50_gamma: float = arguments['EC_50'].value ** self._gamma

        self._cyp_1_1: float = arguments['cyp_1_1'].value  # type:ignore
        self._V1: float = arguments['V1'].value  # type:ignore
        self._V2: float = arguments['V2'].value  # type:ignore

        ktr1 = 6/self._MTT_1  # type:ignore  # 1/hours; changed from 1/MTT_1
        ktr2 = 1/self._MTT_2  # type:ignore  # 1/hours
        self._ktr = np.array([0.0] + [ktr1] * 6 + [0.0, ktr2])

        temp = {'*1/*1': 0.0,
                '*1/*2': 0.315,
                '*1/*3': 0.453,
                '*2/*2': 0.722,
                '*2/*3': 0.69,
                '*3/*3': 0.852}

        if self._CYP2C9 not in temp:
            raise ValueError('The CYP2C9 genotype not recognized fool!')

        self._CL_s = 1.0 - (0.0091 * (self._age - 71) if self._age > 71 else 0)
        self._CL_s *= self._cyp_1_1 * (1 - temp[self._CYP2C9])

        self._last_computed_day = 1
        self._A = np.array([0.0] + [1.0] * 8)

        # prepend time 0 to the list of times for deSolve
        # initial conditions (remove when returning list of times)
        len_times = self._max_time + 1
        times = list(range(len_times))
        # times also equals the time-step for deSolve

        k12 = self._Q / self._V1  # type:ignore
        k21 = self._Q / self._V2  # type:ignore
        k10 = self._CL_s / self._V1
        b = k10 + k21 + k12
        c = k10 * k21
        alpha = (b + math.sqrt(b ** 2 - 4*c)) / 2
        beta = (b - math.sqrt(b ** 2 - 4*c)) / 2

        # 2-compartment model
        part_1 = np.array(
            list(((k21 - alpha) / ((self._ka - alpha)*(beta - alpha)))
                 * temp for temp in (math.exp(-alpha * t) for t in times)))
        part_2 = np.array(
            list(((k21 - beta) / ((self._ka - beta)*(alpha - beta)))
                 * temp for temp in (math.exp(-beta * t) for t in times)))
        part_3 = np.array(
            list(((k21 - self._ka) / ((self._ka - alpha)*(self._ka - beta)))
                 * temp for temp in (math.exp(-self._ka * t) for t in times)))

        multiplication_term = (
            (self._ka * self._F / 2) /
            self._V1) * (part_1 + part_2 + part_3).clip(min=1e-5)

        if self._randomized:
            self._Cs_error = np.exp(  # type:ignore
                np.random.normal(0, 0.09, len_times))
            self._Cs_error_ss = np.exp(  # type:ignore
                np.random.normal(0, 0.30, len_times))
            self._exp_e_INR = np.exp(  # type:ignore
                np.random.normal(0, 0.0325, len_times))
        else:
            self._Cs_error = self._Cs_error_ss = np.ones(len_times)
            self._exp_e_INR = np.ones(len_times)

        self.dose = {0: 0.0}
        if self._lazy:
            self._Cs_values = pd.DataFrame(columns=times)
        else:
            self._total_Cs = np.zeros(len_times)
            multiplication_term = np.trim_zeros(multiplication_term, 'b')

        self._multiplication_term_err = np.multiply(  # type:ignore
            multiplication_term, self._Cs_error).clip(min=0)
        self._multiplication_term_err_ss = np.multiply(  # type:ignore
            multiplication_term, self._Cs_error_ss).clip(min=0)

    def run(self, **inputs) -> Dict[str, Any]:
        '''
        inputs should include:
        - a dictionary called "dose" with days for each dose as keys and
          the amount of dose as values.
        - a list called "measurement_days" that shows INRs of which days
          should be returned.
        '''
        self.dose = inputs.get('dose', {})

        return {'INR': self.INR(inputs.get('measurement_days', []))}

    @property
    def dose(self) -> Dict[int, float]:
        return self._dose

    @dose.setter
    def dose(self, dose: Dict[int, float]) -> None:
        if self._lazy:
            for day, v in dose.items():
                if v != 0.0:
                    self._Cs_values.loc[day] = self._Cs(
                        dose=v, t0=day * self._dose_interval)
                    self._dose[day] = v
                    self._last_computed_day = min(day, self._last_computed_day)
        else:
            for day, v in dose.items():
                if v != 0.0:
                    self._dose[day] = v
                    range_start = day * self._dose_interval
                    Cs = self._Cs(dose=v, t0=range_start, zero_padding=False)
                    range_end = min(
                        range_start + Cs.shape[0], self._max_time + 1)
                    self._total_Cs[range_start:range_end] += \
                        Cs[:range_end-range_start]
                    self._last_computed_day = min(day, self._last_computed_day)

    def _Cs(self, dose: float, t0: int, zero_padding: bool = True) -> np.array:
        # C_s_pred = dose * self._multiplication_term

        # if t0 == 0:  # non steady-state
        #     C_s_error = np.array([exp(gauss(0, 0.09))
        #                           for _ in range(len(C_s_pred))]) \
        #                   if self._randomized \
        #         else np.ones(len(C_s_pred))  # Sadjad
        # else:  # steady-state
        #     C_s_error = np.array([exp(gauss(0, 0.30)) 
        #                           for _ in range(len(C_s_pred))]) \
        #                   if self._randomized \
        #         else np.ones(len(C_s_pred))

        # C_s = np.multiply(C_s_pred, C_s_error).clip(min=0)

        # return np.pad(C_s, (t0, 0), 'constant', 
        #     constant_values=(0,))[:self._max_time+1]

        if t0 == 0:
            C_s = dose * self._multiplication_term_err
        else:
            C_s = dose * self._multiplication_term_err_ss

        return np.concatenate(
            (np.array([0]*t0), C_s[:-t0])) if zero_padding else C_s

    def INR(self, days: Union[int, List[int]]) -> List[float]:
        if isinstance(days, int):
            days = [days]

        if self._lazy:
            base_term = np.sum(self._Cs_values[list(
                range(int(max(days)*self._dose_interval)+1))], axis=0)
        else:
            base_term = self._total_Cs[0:int(max(days)*self._dose_interval)+1]

        Cs_gamma = np.power(base_term, self._gamma)  # type:ignore

        start_days = sorted([0
                             if days[0] < self._last_computed_day
                             else
                             self._last_computed_day] + days[:-1])

        end_days = sorted(days)

        if start_days[0] == 0:
            self._A = np.array([0.0] + [1.0] * 8)
            # self._A = [1]*7
            # self._dA = [0]*7

        INR_max = 20
        baseINR = 1
        INR = []
        for d1, d2 in zip(start_days, end_days):
            for i in range(int(d1*self._dose_interval), int(d2*self._dose_interval)):
                inflow = 1 - self._E_MAX * \
                    Cs_gamma[i] / (self._EC_50_gamma + Cs_gamma[i])
                self._A[0] = inflow
                self._A[7] = inflow
                self._A[1:] += self._ktr[1:] * (self._A[0:-1] - self._A[1:])

                # self._dA[0] = self._ktr1 * (1 - self._E_MAX * Cs_gamma[i] /
                #                             (self._EC_50 ** self._gamma + Cs_gamma[i])) - self._ktr1*self._A[0]
                # for j in range(1, 6):
                #     self._dA[j] = self._ktr1 * (self._A[j-1] - self._A[j])

                # self._dA[6] = self._ktr2 * (1 - self._E_MAX * Cs_gamma[i] /
                #                             (self._EC_50 ** self._gamma + Cs_gamma[i])) - self._ktr2*self._A[6]
                # for j in range(7):
                #     self._A[j] += self._dA[j]

            INR.append(
                (baseINR + (INR_max*(1-self._A[6]*self._A[8]) ** self._lambda)) * self._exp_e_INR[d2])
            # INR.append(
            #     (baseINR + (INR_max*(1-self._A[5]*self._A[6]) ** self._lambda)) * self._exp_e_INR[d2])

        self._last_computed_day = end_days[-1]

        return INR
