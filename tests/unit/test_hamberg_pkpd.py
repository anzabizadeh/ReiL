from logging import warning
import math
import random
from typing import Callable, List, Literal
import unittest

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from reil.datatypes import Feature, FeatureGenerator
from reil.healthcare.mathematical_models import HambergPKPD
from reil.utils.functions import (random_categorical, random_lognormal,
                                  random_normal_truncated, random_uniform)


class testHambergPKPD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(1000)

    def test_replicate_fig_3(self) -> None:
        # Replicating Hamberg et al. (2007) Figure 3

        # keys are subplots of the Fig 3, the tuple represents each
        # patient's 'age', 'CYP2C9', and 'VKORC1' along with 'dose'
        # and the last 'INR'. Since the paper does not have exact
        # values, we ran the model as best as we could to replicate
        # the paper, and then recorded those numbers for reference.
        # This allows us to test any changes in the PKPD
        # implementation, but is no guarantee that these numbers are
        # what they should be. Also, this code saves a chart that
        # can be studied and compared with Fig 3. The zigzag
        # behavior in Fig 3 is the result of hourly INR measurement.
        # We only measure INR daily, hence the smooth lines.
        patients = {
            (0, 0): [(50.0, '*1/*1', 'G/G', 10.0, 2.41),
                     (50.0, '*1/*2', 'G/G', 10.0, 3.00),
                     (50.0, '*1/*3', 'G/G', 10.0, 3.42),
                     (50.0, '*2/*2', 'G/G', 10.0, 5.03),
                     (50.0, '*2/*3', 'G/G', 10.0, 4.73),
                     (50.0, '*3/*3', 'G/G', 10.0, 6.90)],
            (0, 1): [(50.0, '*1/*1', 'G/G', 10.0, 2.41),
                     (50.0, '*1/*1', 'G/A', 10.0, 3.05),
                     (50.0, '*1/*1', 'A/A', 10.0, 3.66)],
            (1, 0): [(50.0, '*1/*1', 'G/G', 10.0, 2.41),
                     (70.0, '*1/*1', 'G/G', 10.0, 2.65),
                     (90.0, '*1/*1', 'G/G', 10.0, 2.97)],
            (1, 1): [(50.0, '*1/*1', 'G/G', 10.0, 2.41),
                     (50.0, '*1/*1', 'A/A', 10.0, 3.66),
                     (50.0, '*1/*3', 'G/A', 10.0, 4.35),
                     (50.0, '*3/*3', 'G/G', 10.0, 6.90)]}

        other_features = {
            'MTT_1': Feature(name='MTT_1', value=HambergPKPD._MTT_1),
            'MTT_2': Feature(name='MTT_2', value=HambergPKPD._MTT_2),
            'CL_S_cyp_1_1': Feature(
                name='CL_S_cyp_1_1', value=HambergPKPD._CL_s_1_1),
            'V1': Feature(name='V1', value=HambergPKPD._V1),
            'V2': Feature(name='V2', value=HambergPKPD._V2)
        }

        _, axes = plt.subplots(
            2, 2, figsize=(10, 10), sharex=True, sharey=True)

        h = HambergPKPD(cache_size=60, randomized=False)  # type: ignore
        h._per_hour = 1
        errors = ''
        for (i, j), info in patients.items():
            for age, cyp, vkor, dose, inr in info:
                if vkor == 'G/G':
                    v = HambergPKPD._EC_50_GG
                elif vkor in ('G/A', 'A/G'):
                    v = HambergPKPD._EC_50_GA
                else:  # 'A/A'
                    v = HambergPKPD._EC_50_AA

                other_features['EC_50'] = Feature(name='EC_50', value=v)

                h.setup(
                    age=Feature(name='age', value=age),
                    CYP2C9=Feature(name='CYP2C9', value=cyp),
                    VKORC1=Feature(name='VKORC1', value=vkor),
                    **other_features)
                h.run(
                    dose={d: dose for d in range(60)},
                    measurement_days=range(61))
                try:
                    self.assertAlmostEqual(
                        h._computed_INRs[60], inr, 2)  # type: ignore
                except AssertionError as e:
                    errors += f'\n({age:4.2f}, {cyp}, {vkor})\t{e}'

                sns.lineplot(data=h._computed_INRs, ax=axes[i][j])
                axes[i][j].set_ylabel('INR', fontsize=12)
                axes[i][j].set_ylim((0, 8))
                axes[i][j].set_xlim((0, 59))
                axes[i][j].set_xlabel('Time (days)', fontsize=12)
                axes[i][j].set_xticks([0, 10, 20, 30, 40, 50])
                for x in (2, 2.5, 3, 4, 5, 6, 7):
                    axes[i][j].axhline(x, ls='--')

        plt.tight_layout()
        plt.savefig('replicated_Hamberg_et_al_2007_Fig_3.png')
        if errors:
            raise AssertionError(errors)

    def test_replicate_fig_4a(self) -> None:
        data = testHambergPKPD.fig_4_generator(
            100, random_pkpd=False)
        data = data[data.t > 0]
        data['t_w_jitter'] = data.t  # + data.x_jitter
        ax = sns.scatterplot(
            data=data, x='t_w_jitter', y='Cs', size=1, legend=False)
        print(data.groupby(['t'])['Cs'].quantile(0.025),
              data.groupby(['t'])['Cs'].median(),
              data.groupby(['t'])['Cs'].quantile(0.975),
              )
        sns.lineplot(data=data.groupby(['t'])['Cs'].quantile(0.025), ax=ax)
        sns.lineplot(data=data.groupby(['t'])['Cs'].median(), ax=ax)
        sns.lineplot(data=data.groupby(['t'])['Cs'].quantile(0.975), ax=ax)
        ax.set_ylim((0.0, 0.8))
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Concentration (mg/l)')
        plt.savefig('replicated_Hamberg_et_al_2007_Fig_4a.png')

    def test_replicate_fig_4c(self) -> None:
        data = testHambergPKPD.fig_4_generator(
            300, INR=True,
            random_parameters=True,
            random_demographic=True,
            random_pkpd=True,
            measurement_days=[0, 0.5, 1.5, 2.5]
            )
        data['t_w_jitter'] = data.t + data.x_jitter
        ax = sns.scatterplot(
            data=data, x='t_w_jitter', y='INR', size=1, legend=False)
        # style='run')
        print(data.groupby(['t'])['INR'].quantile(0.025),
              data.groupby(['t'])['INR'].median(),
              data.groupby(['t'])['INR'].quantile(0.975),
              )
        sns.lineplot(data=data.groupby(['t'])['INR'].quantile(0.025), ax=ax)
        sns.lineplot(data=data.groupby(['t'])['INR'].median(), ax=ax)
        sns.lineplot(data=data.groupby(['t'])['INR'].quantile(0.975), ax=ax)
        ax.set_ylim((0.8, 2.0))
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('INR')
        plt.savefig('replicated_Hamberg_et_al_2007_Fig_4c.png')

    @staticmethod
    def fig_4_generator(
        n,
        days: int = 60,
        measurement_days: List[float] = [0, 0.5, 1.5, 2.5],
        INR: bool = False,
        random_parameters: bool = True,
        random_demographic: bool = True,
        random_pkpd: bool = True,
        random_gen: Callable[[FeatureGenerator], float] = random_lognormal,
        data_source: Literal[
            'Study I', 'Study II', 'Total', 'Ravvaz'] = 'Total'
    ) -> pd.DataFrame:
        if data_source == 'Study I':
            age_info = dict(lower=46.0, upper=87.0, mean=72.0)
            CYP2C9_info = dict(probabilities=(
                0.596, 0.193, 0.158, 0.053, 0.0, 0.0))
            VKORC1_info = dict(probabilities=(0.39, 0.41, 0.20))
        elif data_source == 'Study II':
            age_info = dict(lower=22.0, upper=84.0, mean=70.0)
            CYP2C9_info = dict(probabilities=(
                0.580, 0.161, 0.172, 0.022, 0.043, 0.022))
            warning.warn(
                'VKORC1 is not available for Study II. Total is used instead.')
            VKORC1_info = dict(probabilities=(0.39, 0.41, 0.20))
        elif data_source == 'Total':
            age_info = dict(lower=22.0, upper=87.0, mean=71.0)
            CYP2C9_info = dict(probabilities=(
                0.587, 0.173, 0.167, 0.033, 0.027, 0.013))
            VKORC1_info = dict(probabilities=(0.39, 0.41, 0.20))
        elif data_source == 'Ravvaz':
            age_info = dict(lower=18.0, upper=100.0, mean=67.30, stdev=13.43)
            CYP2C9_info = dict(probabilities=(
                0.6739, 0.1486, 0.0925, 0.0651, 0.0197, 2e-4))
            VKORC1_info = dict(probabilities=(0.3837, 0.4418, 0.1745))
        else:
            raise ValueError

        # VKORC1 and EC_50 are not important in Cs (They affect INR)
        feature_gen_set = {
            'age': FeatureGenerator.numerical(
                name='age',
                **age_info,
                generator=(random_uniform if data_source != 'Ravvaz'
                           else random_normal_truncated),
                randomized=random_demographic),
            'CYP2C9': FeatureGenerator.categorical(
                name='CYP2C9',
                categories=('*1/*1', '*1/*2', '*1/*3',
                            '*2/*2', '*2/*3', '*3/*3'),
                **CYP2C9_info,
                generator=random_categorical,
                randomized=random_demographic,
                allow_missing=True),
            'VKORC1': FeatureGenerator.categorical(
                name='VKORC1',
                categories=('G/G', 'G/A', 'A/A'),
                **VKORC1_info,
                generator=random_categorical,
                randomized=random_demographic,
                allow_missing=True),

            'CL_S_cyp_1_1': FeatureGenerator.numerical(
                name='CL_S_cyp_1_1',
                mean=math.log(HambergPKPD._CL_s_1_1),
                stdev=math.sqrt(HambergPKPD._omega_CL_s),
                generator=random_gen,
                randomized=random_parameters),
            'MTT_1': FeatureGenerator.numerical(
                name='MTT_1',
                mean=math.log(HambergPKPD._MTT_1),
                stdev=math.sqrt(HambergPKPD._omega_MTT_1),
                generator=random_gen,
                randomized=random_parameters),
            'MTT_2': FeatureGenerator.numerical(
                name='MTT_2',
                mean=math.log(HambergPKPD._MTT_2),
                stdev=math.sqrt(HambergPKPD._omega_MTT_2),
                generator=random_gen,
                randomized=random_parameters),
            'V1': FeatureGenerator.numerical(
                name='V1',
                mean=math.log(HambergPKPD._V1),
                stdev=math.sqrt(HambergPKPD._omega_V1),
                generator=random_gen,
                randomized=random_parameters),
            'V2': FeatureGenerator.numerical(
                name='V2',
                mean=math.log(HambergPKPD._V2),
                stdev=math.sqrt(HambergPKPD._omega_V2),
                generator=random_gen,
                randomized=random_parameters),
        }

        concentrations = []
        for i in range(n):
            h = HambergPKPD(
                cache_size=days, randomized=random_pkpd)  # type: ignore
            features = {key: value() for key, value in feature_gen_set.items()}
            vkor = features['VKORC1'].value
            if vkor == 'G/G':
                v = math.log(HambergPKPD._EC_50_GG)
            elif vkor in ('G/A', 'A/G'):
                v = math.log(HambergPKPD._EC_50_GA)
            else:  # 'A/A'
                v = math.log(HambergPKPD._EC_50_AA)

            features['EC_50'] = FeatureGenerator.numerical(
                name='EC_50',
                mean=v,
                stdev=HambergPKPD._omega_EC_50,
                generator=random_gen,
                randomized=random_parameters)()

            h.setup(**features)
            INR_values = h.run(
                dose={0: 10.0},
                measurement_days=measurement_days if INR else [])['INR']
            if not INR:
                INR_values = [0.0] * len(measurement_days)
            # sns.lineplot(y=h._total_cs, x=range(len(h._total_cs)))
            concentrations.extend([
                (i, t * 24,
                 h._total_cs[int(t * 24)]
                 * h._err(int(t * 24), False),  # type: ignore
                 INR_values[j], (random.random() - 0.5) * 2)
                for j, t in enumerate(measurement_days)])
        data = pd.DataFrame(
            concentrations, columns=['run', 't', 'Cs', 'INR', 'x_jitter'])

        return data


if __name__ == "__main__":
    unittest.main()
