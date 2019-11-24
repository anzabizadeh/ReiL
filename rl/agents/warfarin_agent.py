# -*- coding: utf-8 -*-
'''
WarfarinAgent class
=================

An agent for warfarin modeling based on the doses define in Ravvaz et al (2017)

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''


import os
from dill import HIGHEST_PROTOCOL, dump, load
from random import choice, random
from time import time

from math import exp, log, sqrt

import numpy as np

from .agent import Agent
from ..rldata import RLData

class WarfarinAgent(Agent):
    '''
    An agent for warfarin modeling based on the doses define in Ravvaz et al (2017).

    Constructor Arguments
    ---------------------

        Note: This class doesn't have any data_collectors.
    
    Methods
    -------
        act: return an action based on the given state.
        learn: learn using either history or action, reward, and state.
        reset: Not Implemented Yet!
    '''

    def __init__(self, **kwargs):
        '''
        Initialize a warfarin agent.
        '''
        super().__init__(**kwargs)
        super().set_defaults(study_arm='', method='aurora', day=1, retest_day=2, red_flag=False, skip_dose=0, dose=0, weekly_dose=0)

        super().set_params(**kwargs)
        self.data_collector.available_statistics = {}
        self.data_collector.active_statistics = []

        # The following code is just to suppress debugger's undefined variable errors!
        # These can safely be deleted, since all the attributes are defined using set_params!
        if False:
            self._day = 1
            self._method = 'aurora'
            self._study_arm = ''

    def learn(self, **kwargs):
        '''
        Learn based on history.

        Note: Since this agent implements fixed policies, it does not learn.
        '''
        pass

    def act(self, state, **kwargs):
        '''
        return the best action for a given state.

        Arguments
        ---------
            state: the state for which an action is chosen.
        '''
        if self._study_arm.lower() in ['aaa', 'ravvaz aaa', 'ravvaz_aaa']:
            self._method = 'Aurora'
        elif self._study_arm.lower() in ['caa', 'ravvaz caa', 'ravvaz_caa']:
            if state.value.day <= 2:
                self._method = 'IWPC Clinical'
            else:
                self._method = 'Aurora'
        elif self._study_arm.lower() in ['pgaa', 'ravvaz pgaa', 'ravvaz_pgaa']:
            if state.value.day <= 2:
                self._method = 'IWPC PG'
            else:
                self._method = 'Aurora'
        elif self._study_arm.lower() in ['pgpgi', 'ravvaz pgpgi', 'ravvaz_pgpgi']:
            raise NotImplementedError('This arm contains Intermountain dosing algorithm that has not been implemented.')
        elif self._study_arm.lower() in ['pgpga', 'ravvaz pgpga', 'ravvaz_pgpga']:
            if state.value.day <= 3:
                self._method = 'Modified IWPC PG'
            elif state.value.day <= 5:
                self._method = 'Lenzini PG'
            else:
                self._method = 'Aurora'

        r = RLData(self._action(self._method, state), lower=0.0, upper=15.0)
        return r

    def _action(self, method, state):
        patient = state.value

        if patient.day <= 1:
            self.reset()
            self._retest_day = patient.day
        elif self._retest_day > patient.day and not self._red_flag:
            if self._skip_dose > 0:
                self._skip_dose -= 1
                return 0.0
            else:
                return self._dose

        if method.lower() == 'aurora':
            if self._red_flag:
                if self._retest_day > patient.day:
                    return 0.0
                elif patient.INRs[-1] > 3.0:
                    self._retest_day = patient.day + 2
                    return 0.0
                else:
                    self._red_flag = False
                    self._dose = self._dose * 0.85
                    self._retest_day = patient.day + 7
                    return self._dose

            if patient.day <= 2:
                self._dose = 10.0 if patient.age < 65.0 else 5.0
            elif patient.day <= 4:
                day_2_INR = patient.INRs[-1] if patient.day == 3 else patient.INRs[-2]
                if day_2_INR >= 2.0:
                    self._dose = 5.0
                elif day_2_INR < 1.50:
                    self._dose = self._dose * 1.15
                    self._retest_day = patient.day + 7
                elif day_2_INR < 1.80:
                    self._dose = self._dose * 1.10
                    self._retest_day = patient.day + 7
                else:
                    self._dose = self._dose * 1.075
                    self._retest_day = patient.day + 7
            else:
                current_INR = patient.INRs[-1]
                if current_INR < 1.50:
                    self._dose = self._dose * 1.15
                    self._retest_day = patient.day + 7
                elif current_INR < 1.80:
                    self._dose = self._dose * 1.10
                    self._retest_day = patient.day + 7
                elif current_INR < 2.00:
                    self._dose = self._dose * 1.075
                    self._retest_day = patient.day + 7
                elif current_INR <= 3.00:
                    self._retest_day = patient.day + 28
                elif current_INR < 3.40:
                    self._dose = self._dose * 0.925
                    self._retest_day = patient.day + 7
                elif current_INR < 4.00:
                    self._dose = self._dose * 0.9
                    self._retest_day = patient.day + 7
                elif current_INR <= 5.00:
                    self._skip_dose = 2
                    self._dose = self._dose * 0.875
                    self._retest_day = patient.day + 7
                else:
                    self._red_flag = True
                    self._retest_day = patient.day + 2
                    return 0.0

            if self._skip_dose == 0:
                return self._dose
            else:
                return 0.0

        elif method.lower() in ['iwpc_clinical', 'iwpc', 'iwpc clinical']:
            if self._weekly_dose == 0:
                self._weekly_dose = (4.0376
                            - 0.2546 * patient.age / 10
                            + 0.0118 * patient.height * 2.54  # in to cm
                            + 0.0134 * patient.weight * 0.454  # lb to kg
                            - 0.6752 * (patient.race == 'Asian')
                            + 0.4060 * (patient.race == 'Black')
                            + 0.0443 * (patient.race not in  ['Asian', 'Black'])
                            + 1.2799 * 0  # Enzyme inducer status (Fluvastatin is reductant not an inducer!)
                            - 0.5695 * (patient.amiodarone == 'Yes')) ** 2
                self._dose = self._weekly_dose / 7

            if patient.day <= 3:
                return self._dose

        elif method.lower() in ['iwpc_pg', 'iwpc pg']:
            if self._weekly_dose == 0:
                self._weekly_dose = (5.6044
                            - 0.2614 * patient.age / 10
                            + 0.0087 * patient.height * 2.54  # in to cm
                            + 0.0128 * patient.weight * 0.454  # lb to kg
                            - 0.8677 * (patient.VKORC1 == 'G/A')
                            - 1.6974 * (patient.VKORC1 == 'A/A')
                            - 0.4854 * (patient.VKORC1 not in  ['G/A', 'A/A', 'G/G'])
                            - 0.5211 * (patient.CYP2C9 == '*1/*2')
                            - 0.9357 * (patient.CYP2C9 == '*1/*3')
                            - 1.0616 * (patient.CYP2C9 == '*2/*2')
                            - 1.9206 * (patient.CYP2C9 == '*2/*3')
                            - 2.3312 * (patient.CYP2C9 == '*3/*3')
                            - 0.2188 * (patient.CYP2C9 not in ['*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', '*1/*1'])
                            - 0.1092 * (patient.race == 'Asian')
                            + 0.2760 * (patient.race == 'Black')
                            + 1.0320 * (patient.race not in  ['Asian', 'Black'])
                            + 1.1816 * 0  # Enzyme inducer status (Fluvastatin is reductant not an inducer!)
                            - 0.5503 * (patient.amiodarone == 'Yes')) ** 2
                self._dose = self._weekly_dose / 7

            if patient.day <= 3:
                return self._dose

        elif method.lower() in ['modified_iwpc_pg', 'modified iwpc pg']:
            if self._weekly_dose == 0:
                self._weekly_dose = (5.6044
                            - 0.2614 * patient.age / 10
                            + 0.0087 * patient.height * 2.54  # in to cm
                            + 0.0128 * patient.weight * 0.454  # lb to kg
                            - 0.8677 * (patient.VKORC1 == 'G/A')
                            - 1.6974 * (patient.VKORC1 == 'A/A')
                            - 0.5211 * (patient.CYP2C9 == '*1/*2')
                            - 0.9357 * (patient.CYP2C9 == '*1/*3')
                            - 1.0616 * (patient.CYP2C9 == '*2/*2')
                            - 1.9206 * (patient.CYP2C9 == '*2/*3')
                            - 2.3312 * (patient.CYP2C9 == '*3/*3')
                            - 0.5503 * (patient.amiodarone == 'Yes')) ** 2

            k = {'*1/*1': 0.0189,
                 '*1/*2': 0.0158,
                 '*1/*3': 0.0132,
                 '*2/*2': 0.0130,
                 '*2/*3': 0.0090,
                 '*3/*3': 0.0075
                 }
            LD3 = self._weekly_dose / ((1 - exp(-24*k[patient.CYP2C9])) * (1 + exp(-24*k[patient.CYP2C9]) + exp(-48*k[patient.CYP2C9])))
            if patient.day == 1:
                self._dose = 1.5 * LD3 - 0.5 * self._weekly_dose
            elif patient.day == 2:
                self._dose = LD3
            else:
                self._dose = 0.5 * LD3 + 0.5 * self._weekly_dose

        elif method.lower() in ['lenzini_pg', 'lenzini pg'] and patient.day in (4, 5):
            self._dose = exp(3.10894
                            - 0.00767 * patient.age
                            - 0.51611 * log(patient.INRs[-1])
                            - 0.23032 * (patient.VKORC1 == 'G/A')
                            - 0.14745 * (patient.CYP2C9 in ['*1/*2', '*2/*2'])
                            - 0.30770 * (patient.CYP2C9 in ['*1/*3', '*2/*3', '*3/*3'])
                            + 0.24597 * sqrt(patient.height * 2.54 * patient.weight * 0.454 / 3600)  # BSA
                            + 0.26729 * 2.5  # target INR
                            - 0.10350 * (patient.amiodarone == 'Yes')
                            + 0.01690 * patient.Doses[-2]
                            + 0.02018 * patient.Doses[-3]
                            + 0.01065 * patient.Doses[-4]
                            ) / 7

        return self._dose

    def reset(self):
        self._retest_day = 1
        self._red_flag = False
        self._skip_dose = 0
        self._dose = 0
        self._weekly_dose = 0

    # def load(self, **kwargs):
    #     '''
    #     Load an object from a file.

    #     Arguments
    #     ---------
    #         filename: the name of the file to be loaded.

    #     Note: tensorflow part is saved in filename.tf folder

    #     Raises ValueError if the filename is not specified.
    #     '''
    #     Agent.load(self, **kwargs)
    #     tf.reset_default_graph()
    #     self._model = keras.models.load_model(kwargs.get(
    #         'path', self._path) + '/' + kwargs['filename'] + '.tf/' + kwargs['filename'])
    #     self._tensorboard = keras.callbacks.TensorBoard(
    #         log_dir=self._tensorboard_path)

    # def save(self, **kwargs):
    #     '''
    #     Save the object to a file.

    #     Arguments
    #     ---------
    #         filename: the name of the file to be saved.

    #     Note: tensorflow part should be in filename.tf folder

    #     Raises ValueError if the filename is not specified.
    #     '''

    #     pickle_data = tuple(key for key in self.__dict__ if key not in [
    #                         '_model', '_tensorboard', 'data_collector'])
    #     path, filename = Agent.save(self, **kwargs, data=pickle_data)
    #     try:
    #         self._model.save(kwargs.get('path', self._path) + '/' +
    #                          kwargs['filename'] + '.tf/' + kwargs['filename'])
    #     except OSError:
    #         os.makedirs(kwargs.get('path', self._path) +
    #                     '/' + kwargs['filename'] + '.tf/')
    #         self._model.save(kwargs.get('path', self._path) + '/' +
    #                          kwargs['filename'] + '.tf/' + kwargs['filename'])
    #     return path, filename

    def _report(self, **kwargs):
        '''
        generate and return the requested report.

        Arguments
        ---------
            statistic: the list of items to report.

        Note: this function is not implemented!
        '''
        raise NotImplementedError

    def __repr__(self):
        return 'WarfarinAgent'

if __name__ == "__main__":
    from rl.subjects import WarfarinModel_v4
    w = WarfarinModel_v4(age=87, CYP2C9='*1/*1', VKORC1G='G/G', extended_state=True)
    a = WarfarinAgent()
    INRs = [1, 1.1, 1.3, 1.3, 1.5, 1.3, 1.5, 1.6, 1.7, 1.7, 1.8, 1.8, 1.9, 1.9, 1.9, 1.8, 1.9, 1.9, 1.8, 1.8, 1.9, 1.9, 2, 1.8, 2, 2.1, 2, 2, 2.1, 1.9, 2.1, 2.1, 2.2, 1.9, 2, 1.9, 1.9, 2, 2, 2, 2, 1.9, 1.8, 1.9, 2.1, 1.9, 2, 2, 2.1, 2, 2, 2, 2.1, 2.2, 1.9, 2, 2.1, 2.1, 1.9, 1.9, 1.8, 2.1, 2, 2.1, 2, 2, 2.1, 2, 2, 2.1, 2.1, 2.1, 2, 2.1, 1.9, 1.8, 2, 2.1, 2, 2.1, 2.1, 2, 2, 2.3, 1.9, 2.1, 2, 1.8, 2.1]
    for i in INRs:
        action = a.act(w.state)
        w._INR.append(i)
        w._INR.popleft()
        w._day += 1
        print(action.value[0], w.state.value.INRs[-1])