# -*- coding: utf-8 -*-
'''
WarfarinPatientRavvaz class
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
from typing import Any

from reil.datatypes import FeatureGenerator
from reil.subjects.healthcare import Patient
from reil.subjects.healthcare.mathematical_models import HealthMathModel
from reil.utils.functions import (random_categorical, random_truncated_lnorm,
                                  random_truncated_normal)


# pre-computing to gain some speed boost!
log_4_61 = math.log(4.61)
log_3_02 = math.log(3.02)
log_2_20 = math.log(2.20)
sqrt_0_409 = math.sqrt(0.409)


class WarfarinPatientRavvaz(Patient):
    def __init__(self,
                 model: HealthMathModel,
                 randomized: bool = True,
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
                name='age',  # Aurora population
                lower=18.0, upper=100.0, mean=67.30, stdev=13.43,
                generator=random_truncated_normal,
                randomized=randomized),
            'weight': FeatureGenerator.numerical(
                name='weight',  # lb  - Aurora population
                lower=70.0, upper=500.0, mean=199.24, stdev=54.71,
                generator=random_truncated_normal,
                randomized=randomized),
            'height': FeatureGenerator.numerical(
                name='height',  # in - Aurora population
                lower=45.0, upper=85.0, mean=66.78, stdev=4.31,
                generator=random_truncated_normal,
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
                randomized=randomized),
            'VKORC1': FeatureGenerator.categorical(
                name='VKORC1',  # Aurora Avatar Population
                categories=('G/G', 'G/A', 'A/A'),
                probabilities=(0.3837, 0.4418, 0.1745),
                generator=random_categorical,
                randomized=randomized),

            'MTT_1': FeatureGenerator.numerical(
                name='MTT_1',  # Hamberg PK/PD
                mean=math.log(11.6), stdev=math.sqrt(0.141),
                generator=random_truncated_lnorm,
                randomized=randomized),
            'MTT_2': FeatureGenerator.numerical(
                name='MTT_2',  # Hamberg PK/PD
                mean=math.log(120.0), stdev=math.sqrt(1.02),
                generator=random_truncated_lnorm,
                randomized=randomized),
            'cyp_1_1': FeatureGenerator.numerical(
                name='cyp_1_1',  # Hamberg PK/PD
                mean=math.log(0.314), stdev=math.sqrt(0.31),
                generator=random_truncated_lnorm,
                randomized=randomized),
            'V1': FeatureGenerator.numerical(
                name='V1',  # Hamberg PK/PD
                mean=math.log(13.8), stdev=math.sqrt(0.262),
                generator=random_truncated_lnorm,
                randomized=randomized),
            'V2': FeatureGenerator.numerical(
                name='V2',  # Hamberg PK/PD
                mean=math.log(6.59), stdev=math.sqrt(0.991),
                generator=random_truncated_lnorm,
                randomized=randomized),

            # 'EC_50': FeatureGenerator.numerical(
            #     name='EC_50',  # Hamberg PK/PD
            #     stdev=math.sqrt(0.409),
            #     generator=random_truncated_lnorm,
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
        if self.feature_set['VKORC1'].value == 'G/G':
            mean = log_4_61
        elif self.feature_set['VKORC1'].value in ('G/A', 'A/G'):
            mean = log_3_02
        else:  # if self.feature_set['VKORC1'].value == 'A/A':
            mean = log_2_20

        self.feature_gen_set['EC_50'] = FeatureGenerator.numerical(
            name='EC_50',  # Hamberg PK/PD
            mean=mean,
            stdev=sqrt_0_409,
            generator=random_truncated_lnorm,
            randomized=self._randomized)

        self.feature_set['EC_50'] = self.feature_gen_set['EC_50']()

    def _generate_sensitivity(self):
        combo = (self.feature_set['CYP2C9'].value +
                 self.feature_set['VKORC1'].value)

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
