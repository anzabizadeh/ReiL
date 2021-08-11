# -*- coding: utf-8 -*-
'''
PatientWarfarinRavvaz class
===========================

A warfarin patient class with features and parameters of Ravvaz et al. 2016.

Features included in this model are:
* age
* weight
* height
* gender
* race
* tobaco
* amiodarone
* fluvastatin
* CYP2C9
* VKORC1
* MTT_1
* MTT_2
* cyp_1_1
* V1
* V2
* EC_50
'''
import math
from typing import Any, Literal

from reil.datatypes.feature import FeatureGenerator
from reil.healthcare.patient import Patient
from reil.healthcare.mathematical_models import HealthMathModel
from reil.healthcare.mathematical_models.hamberg_pkpd import HambergPKPD
from reil.utils.functions import (random_categorical,
                                  random_lognormal_truncated,
                                  random_normal_truncated)

# pre-computing to gain some speed boost!
log_EC_50_GG = math.log(HambergPKPD._EC_50_GG)
log_EC_50_GA = math.log(HambergPKPD._EC_50_GA)
log_EC_50_AA = math.log(HambergPKPD._EC_50_AA)
sqrt_omega_EC_50 = math.sqrt(HambergPKPD._omega_EC_50)


class PatientWarfarinRavvaz(Patient):
    def __init__(self,
                 model: HealthMathModel,
                 randomized: bool = True,
                 allow_missing_genotypes: bool = True,
                 **feature_values: Any) -> None:
        '''
        Parameters
        ----------
        model:
            A `HealthMathModel` to be used to model patient's behavior.

        randomized:
            Whether patient characteristics and model parameters should be
            generated randomly or deterministically.

        feature_values:
            Keyword arguments by which some of the `features` of the patient
            can be determined. For example, if "age" is one of the features,
            age=40.0 will set the initial age to 40.0.
        '''
        self.feature_gen_set = {
            'age': FeatureGenerator.numerical(
                name='age',  # (years) Aurora population
                lower=18.0, upper=150.0, mean=67.30, stdev=13.43,
                generator=random_normal_truncated,
                randomized=randomized),
            'weight': FeatureGenerator.numerical(
                name='weight',  # (lb) Aurora population
                lower=70.0, upper=500.0, mean=199.24, stdev=54.71,
                generator=random_normal_truncated,
                randomized=randomized),
            'height': FeatureGenerator.numerical(
                name='height',  # (in) Aurora population
                lower=45.0, upper=85.0, mean=66.78, stdev=4.31,
                generator=random_normal_truncated,
                randomized=randomized),
            'gender': FeatureGenerator.categorical(
                name='gender',  # Aurora population
                categories=('Female', 'Male'),
                probabilities=(0.5314, 0.4686),
                generator=random_categorical,
                randomized=randomized),
            'race': FeatureGenerator.categorical(
                name='race',  # Aurora Avatar Population
                categories=('White', 'Black', 'Asian',
                            'American Indian', 'Pacific Islander'),
                probabilities=(0.9522, 0.0419, 0.0040, 0.0018, 1e-4),
                generator=random_categorical,
                randomized=randomized),
            'tobaco': FeatureGenerator.categorical(
                name='tobaco',  # Aurora Avatar Population
                categories=('No', 'Yes'),
                probabilities=(0.9067, 0.0933),
                generator=random_categorical,
                randomized=randomized),
            'amiodarone': FeatureGenerator.categorical(
                name='amiodarone',  # Aurora Avatar Population
                categories=('No', 'Yes'),
                probabilities=(0.8849, 0.1151),
                generator=random_categorical,
                randomized=randomized),
            'fluvastatin': FeatureGenerator.categorical(
                name='fluvastatin',  # Aurora Avatar Population
                categories=('No', 'Yes'),
                probabilities=(0.9998, 0.0002),
                generator=random_categorical,
                randomized=randomized),
            'CYP2C9': FeatureGenerator.categorical(
                name='CYP2C9',  # Aurora Avatar Population
                categories=('*1/*1', '*1/*2', '*1/*3',
                            '*2/*2', '*2/*3', '*3/*3'),
                probabilities=(0.6739, 0.1486, 0.0925, 0.0651, 0.0197, 2e-4),
                generator=random_categorical,
                randomized=randomized,
                allow_missing=allow_missing_genotypes),
            'VKORC1': FeatureGenerator.categorical(
                name='VKORC1',  # Aurora Avatar Population
                categories=('G/G', 'G/A', 'A/A'),
                probabilities=(0.3837, 0.4418, 0.1745),
                generator=random_categorical,
                randomized=randomized,
                allow_missing=allow_missing_genotypes),

            'MTT_1': FeatureGenerator.numerical(
                name='MTT_1',  # (hours) Hamberg PK/PD
                mean=math.log(HambergPKPD._MTT_1),
                stdev=math.sqrt(HambergPKPD._omega_MTT_1),
                generator=random_lognormal_truncated,
                randomized=randomized),
            'MTT_2': FeatureGenerator.numerical(
                name='MTT_2',  # (hours) Hamberg PK/PD
                # Hamberg et al. (2007) - Table 4
                mean=math.log(HambergPKPD._MTT_2),
                stdev=math.sqrt(HambergPKPD._omega_MTT_2),
                generator=random_lognormal_truncated,
                randomized=randomized),
            'CL_S_cyp_1_1': FeatureGenerator.numerical(
                name='CL_S_cyp_1_1',  # (l/h) Hamberg PK/PD
                mean=math.log(HambergPKPD._CL_s_1_1),
                stdev=math.sqrt(HambergPKPD._omega_CL_s),
                generator=random_lognormal_truncated,
                randomized=randomized),
            'V1': FeatureGenerator.numerical(
                name='V1',  # (L) Volume in central compartment
                mean=math.log(HambergPKPD._V1),
                stdev=math.sqrt(HambergPKPD._omega_V1),
                generator=random_lognormal_truncated,
                randomized=randomized),
            'V2': FeatureGenerator.numerical(
                name='V2',  # (L) volume in peripheral compartment
                mean=math.log(HambergPKPD._V2),
                stdev=math.sqrt(HambergPKPD._omega_V2),
                generator=random_lognormal_truncated,
                randomized=randomized),

            # 'EC_50': FeatureGenerator.numerical(
            #     name='EC_50',  # (mg/L) Hamberg PK/PD
            #     stdev=math.sqrt(0.409),
            #     generator=random_lognormal_truncated,
            #     randomized=randomized),

            # 'sensitivity': FeatureGenerator.categorical(
            #     name='sensitivity',
            #     categories=('normal', 'sensitive', 'highly sensitive'),
            #     generator=lambda f: f.value,
            #     randomized=randomized)
        }

        self._sensitivity_gen = FeatureGenerator.categorical(
            name='sensitivity',
            categories=('normal', 'sensitive', 'highly sensitive'))

        # Since EC_50 is not set (it depends on other features),
        # super().__init__() fails to setup the model.
        # I catch it, generate EC_50 and set up the model.
        self._randomized = randomized
        try:
            super().__init__(model, **feature_values)
        except KeyError:
            self._generate_EC_50()
            self._generate_sensitivity()
            self._model.setup(**self.feature_set)

    def generate(self) -> None:
        self.feature_gen_set.pop('EC_50')
        super().generate()

        self._generate_EC_50()
        self._generate_sensitivity()
        self._model.setup(**self.feature_set)

    def _generate_EC_50(self) -> None:
        vkorc1 = self.feature_set['VKORC1'].value
        if vkorc1 == 'G/G':
            mean = log_EC_50_GG
        elif vkorc1 in ('G/A', 'A/G'):
            mean = log_EC_50_GA
        else:  # 'A/A'
            mean = log_EC_50_AA

        self.feature_gen_set['EC_50'] = FeatureGenerator.numerical(
            name='EC_50',  # (mg/L) Hamberg PK/PD
            mean=mean,
            stdev=sqrt_omega_EC_50,
            generator=random_lognormal_truncated,
            randomized=self._randomized)

        self.feature_set['EC_50'] = self.feature_gen_set['EC_50']()

    def _generate_sensitivity(self):
        combo = (str(self.feature_set['CYP2C9'].value) +
                 str(self.feature_set['VKORC1'].value))

        if combo in ('*1/*1G/G', '*1/*2G/G', '*1/*1G/A'):
            s = 'normal'
        elif combo in ('*1/*2G/A', '*1/*3G/A', '*2/*2G/A',
                       '*2/*3G/G', '*1/*3G/G', '*2/*2G/G',
                       '*1/*2A/A', '*1/*1A/A'):
            s = 'sensitive'
        elif combo in ('*3/*3G/G',
                       '*3/*3G/A', '*2/*3G/A',
                       '*3/*3A/A', '*2/*3A/A', '*2/*2A/A', '*1/*3A/A'):
            s = 'highly sensitive'
        else:
            raise ValueError(
                f'Unknown CYP2C9 and VKORC1 combination: {combo}.')

        self.feature_set['sensitivity'] = self._sensitivity_gen(s)
