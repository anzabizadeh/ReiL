import numpy as np
import pandas as pd
from math import exp, log, sqrt
from random import seed, normalvariate, sample
from scipy.stats import lognorm


def rlnormRestricted(meanVal, stdev):
    # capture 50% of the data.  This restricts the log values to a "reasonable" range
    quartileRange = (0.25, 0.75)
    lnorm = lognorm(stdev, scale=exp(meanVal))
    qValues = lnorm.ppf(quartileRange)
    values = list(v for v in lnorm.rvs(size=1000)
                  if (v > qValues[0]) & (v < qValues[1]))
    return sample(values, 1)[0]


class Patient:
    def __init__(self, age=50, CYP2C9='*3/*3', VKORC1='G/G', randomized=True, max_time=24,
                 dose_interval=24, dose={}, **kwargs):
        self._age = age
        self._CYP2C9 = CYP2C9
        self._VKORC1 = VKORC1
        self._randomized = randomized
        self._MTT_1 = kwargs.get('MTT_1', rlnormRestricted(
            log(11.6), sqrt(0.141)) if randomized else 11.6)
        self._MTT_2 = kwargs.get('MTT_2', rlnormRestricted(
            log(120), sqrt(1.02)) if randomized else 120)
        # EC_50 in mg/L
        self._EC_50 = kwargs.get('EC_50', None)
        if self._EC_50 is None:
            if VKORC1 == "G/G":  # Order of genotypes changed
                self._EC_50 = rlnormRestricted(
                    log(4.61), sqrt(0.409)) if randomized else 4.61
            elif VKORC1 in ["G/A", "A/G"]:
                self._EC_50 = rlnormRestricted(
                    log(3.02), sqrt(0.409)) if randomized else 3.02
            elif VKORC1 == "A/A":
                self._EC_50 = rlnormRestricted(
                    log(2.20), sqrt(0.409)) if randomized else 2.20
            else:
                raise ValueError('The VKORC1 genotype is not supported!')

        self._cyp_1_1 = kwargs.get('cyp_1_1', rlnormRestricted(
            log(0.314), sqrt(0.31)) if randomized else 0.314)
        self._V1 = kwargs.get('V1', rlnormRestricted(
            log(13.8), sqrt(0.262)) if randomized else 13.8)
        self._V2 = kwargs.get('V2', rlnormRestricted(
            log(6.59), sqrt(0.991)) if randomized else 6.59)
        self._Q = 0.131    # (L/h)
        self._lambda = 3.61

        self._gamma = 0.424  # no units

        # bioavilability fraction 0-1 (from: "Applied Pharmacokinetics & Pharmacodynamics 4th edition, p.717", some other references)
        self._F = 0.9

        self._ka = 2  # absorption rate (1/hr)

        self._ktr1 = 6/self._MTT_1					# 1/hours; changed from 1/MTT_1
        self._ktr2 = 1/self._MTT_2					# 1/hours
        self._E_MAX = 1					        	# no units

        self._CL_s = 1
        if self._age > 71:
            self._CL_s = 1 - 0.0091 * (self._age - 71)

        if self._CYP2C9 == "*1/*1":
            self._CL_s = self._CL_s * self._cyp_1_1
        elif self._CYP2C9 == "*1/*2":
            self._CL_s = self._CL_s * self._cyp_1_1 * (1 - 0.315)
        elif self._CYP2C9 == "*1/*3":
            self._CL_s = self._CL_s * self._cyp_1_1 * (1 - 0.453)
        elif self._CYP2C9 == "*2/*2":
            self._CL_s = self._CL_s * self._cyp_1_1 * (1 - 0.722)
        elif self._CYP2C9 == "*2/*3":
            self._CL_s = self._CL_s * self._cyp_1_1 * (1 - 0.69)
        elif self._CYP2C9 == "*3/*3":
            self._CL_s = self._CL_s * self._cyp_1_1 * (1 - 0.852)
        else:
            raise ValueError('The CYP2C9 genotype not recognized fool!')

        self._max_time = max_time  # The last hour of experiment
        self._dose_interval = dose_interval

        # prepend time 0 to the list of times for deSolve initial conditions (remove when returning list of times)
        times = list(range(self._max_time+1))
        # times also equals the time-step for deSolve

        k12 = self._Q / self._V1
        k21 = self._Q / self._V2
        k10 = self._CL_s / self._V1
        b = k10 + k21 + k12
        c = k10 * k21
        alpha = (b + sqrt(b ** 2 - 4*c)) / 2
        beta = (b - sqrt(b ** 2 - 4*c)) / 2

        # 2-compartment model
        part_1 = np.array(list(((k21 - alpha) / ((self._ka - alpha)*(beta - alpha)))
                               * temp for temp in (exp(-alpha * t) for t in times)))
        part_2 = np.array(list(((k21 - beta) / ((self._ka - beta)*(alpha - beta)))
                               * temp for temp in (exp(-beta * t) for t in times)))
        part_3 = np.array(list(((k21 - self._ka) / ((self._ka - alpha)*(self._ka - beta)))
                               * temp for temp in (exp(-self._ka * t) for t in times)))
        self._multiplication_term = part_1 + part_2 + part_3

        self._data = pd.DataFrame(columns=['dose']+times)
        self.dose = dose

    @property
    def dose(self):
        return self._data.dose

    @dose.setter
    def dose(self, dose):
        for d, v in dose.items():
            self._data.loc[d] = np.insert(
                self.Cs(dose=v, t0=d*self._dose_interval), 0, v)

    def Cs(self, dose, t0):
        # NOTE: Here the error is one value, but it seemed to be Cij

        if t0 == 0:  # non steady-state
            C_s_error = exp(normalvariate(0, 0.30)
                            ) if self._randomized else 1  # Sadjad
        else:  # steady-state
            C_s_error = exp(normalvariate(0, 0.09)) if self._randomized else 1

        C_s_pred = ((self._ka * self._F * dose / 2) /
                    self._V1) * self._multiplication_term
        C_s = C_s_pred * C_s_error

        return np.pad(C_s, (t0, 0), 'constant', constant_values=(0,))[:self._max_time+1]

    def INR(self, days):
        if isinstance(days, int):
            days = [days]

        Cs_gamma = np.nan_to_num(np.power(np.sum(self._data[list(
            range(int(max(days)*self._dose_interval)+1))], axis=0), np.array(self._gamma)))

        A = [1]*7
        dA = [0]*7

        INR_max = 20
        baseINR = 1
        INR = []
        for d1, d2 in zip(sorted([0]+days[:-1]), sorted(days)):
            for i in range(int(d1*self._dose_interval), int(d2*self._dose_interval)):
                dA[0] = self._ktr1 * (1 - self._E_MAX * Cs_gamma[i] /
                                      (self._EC_50 ** self._gamma + Cs_gamma[i])) - self._ktr1*A[0]
                for j in range(1, 6):
                    dA[j] = self._ktr1 * (A[j-1] - A[j])

                dA[6] = self._ktr2 * (1 - self._E_MAX * Cs_gamma[i] /
                                      (self._EC_50 ** self._gamma + Cs_gamma[i])) - self._ktr2*A[6]
                for j in range(7):
                    A[j] += dA[j]

            e_INR = normalvariate(0, 0.0325) if self._randomized else 0
            INR.append(
                (baseINR + (INR_max*(1-A[5]*A[6]) ** self._lambda)) * exp(e_INR))

        return INR


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    max_day = 100
    p = Patient(randomized=False, max_time=24*max_day + 1)
    p.dose = {i: 7.5 for i in range(max_day)}
    plt.plot(p.INR(list(i/24 for i in range(1, max_day*24 + 1))))
    plt.plot(list(range(24, (max_day+1)*24, 240)),
             p.INR(list(i for i in range(1, max_day+1, 10))))
    plt.show()
