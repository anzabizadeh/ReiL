# -*- coding: utf-8 -*-
'''
WarfarinAgent class
=================

An agent for warfarin modeling based on the doses define in Ravvaz et al (2017)

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from logging import WARNING
from math import exp, log, sqrt
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from rl import rlbase, rldata, agents


class WarfarinAgent(agents.Agent):
    '''
    An agent for warfarin modeling based on the doses defined in Ravvaz et al (2017).

    Constructor Arguments
    ---------------------

        Note: This class doesn't have any data_collectors.
    
    Methods
    -------
        act: return an action based on the given state.
        learn: learn using either history or action, reward, and state.
        reset: Resets the dosing algorithm to day 1.
    '''

    def __init__(self, study_arm: str = 'AAA',
                 name: str = 'warfarin_agent',
                 version: float = 0.5,
                 path: str = '.',
                 logger_name: str = __name__,
                 logger_level: int = WARNING,
                 logger_filename: Optional[str] = None):
        '''
        Initialize a warfarin agent.

        Arguments:
        \n  study_arm: one of available study arms: AAA, CAA, PGAA, PGPGI, PGPGA
        '''
        super().__init__(name=name,
                         version=version,
                         path=path,
                         logger_name=logger_name,
                         logger_level=logger_level,
                         logger_filename=logger_filename)

        self._study_arm = study_arm
        self._method: Callable[[Dict[str, Any]], float] = lambda x: -1  # type: ignore
        self._day = 1
        self._retest_day: int = 2
        self._skip_dose = 0
        self._dose: float = 0.0
        self._weekly_dose: float = 0.0
        self._red_flag = False  # used in aurora maintenance algorithm
        self._lenzini_on_day_4 = False  # used in lenzini adjustment algorithm
        self._number_of_stable_days = 0  # used in aurora maintenance algorithm
        self._visited_range = 0.0  # used in intermountain maintenance algorithm

        self.data_collector.available_statistics = {}
        self.data_collector.active_statistics = []

    def learn(self, history: List[Dict[str, Any]],
        observation: Optional[rlbase.Observation] = None) -> None:
        '''
        Learn based on history.

        Note: Since this agent implements fixed policies, it does not learn.
        '''
        pass

    def act(self,
        state: rldata.RLData,
        actions: Optional[Sequence[rldata.RLData]] = None,
        episode: Optional[int] = 0) -> rldata.RLData:
        '''
        return the best action for a given state.

        Arguments
        ---------
            state: the state for which an action is chosen.
        '''
        study_arm = self._study_arm.lower()
        patient: Dict[str, Any] = state.value
        patient['day'] += 1
        today = patient['day']

        if study_arm in ['aaa', 'ravvaz aaa', 'ravvaz_aaa']:
            self._method = self._aurora

        elif study_arm in ['caa', 'ravvaz caa', 'ravvaz_caa']:
            if today <= 2:
                self._method = self._iwpc_clinical
            else:
                self._method = self._aurora

        elif study_arm in ['pgaa', 'ravvaz pgaa', 'ravvaz_pgaa']:
            if today <= 2:
                self._method = self._iwpc_pg
            else:
                self._method = self._aurora

        elif study_arm in ['pgpgi', 'ravvaz pgpgi', 'ravvaz_pgpgi']:
            if today <= 3:
                self._method = self._modified_iwpc_pg
            elif today <= 5:
                self._method = self._lenzini_pg
            else:
                self._method = self._intermountain

        elif study_arm in ['pgpga', 'ravvaz pgpga', 'ravvaz_pgpga']:
            if today <= 3:
                self._method = self._modified_iwpc_pg
            elif today <= 5:
                self._method = self._lenzini_pg
            else:
                self._method = self._aurora

        dose = self._precheck(patient)
        if dose == -1:
            # print('* ', end='')
            dose = round(self._method(patient), 9)

        # Cuts out the dose if it is >15.0
        return rldata.RLData({'dose': {'value': min(dose, 15.0), 'lower': 0.0, 'upper': 15.0}})

    def _precheck(self, patient: Dict[str, Any]) -> float:
        v = -1
        if patient['day'] <= 1:
            self.reset()
            self._retest_day = patient['day']
        elif self._retest_day > patient['day'] and not self._red_flag:
            if self._skip_dose > 0:
                self._skip_dose -= 1
                v = 0.0
            else:
                v = self._dose
            
        return v

    def _aurora(self, patient: Dict[str, Any]) -> float:
        if self._red_flag:
            if self._retest_day > patient['day']:
                return_value = 0.0
            elif patient['INRs'][-1] > 3.0:
                self._retest_day = patient['day'] + 2
                return_value = 0.0
            else:
                self._red_flag = False
                self._retest_day = patient['day'] + 7
                return_value = self._dose
            return return_value

        next_test = 2
        if patient['day'] == 1:
            self._dose = 10.0 if patient['age'] < 65.0 else 5.0
            # self._retest_day = 3
        elif patient['day'] <= 4:
            day_2_INR = patient['INRs'][-1] if patient['day'] == 3 else patient['INRs'][-2]
            if day_2_INR >= 2.0:
                self._dose = 5.0
                if day_2_INR <= 3.0:
                    self._early_therapeutic = True

                self._number_of_stable_days, next_test = self._aurora_retesting_table(patient['INRs'][-1], self._number_of_stable_days, self._early_therapeutic)
                # self._retest_day = patient['day'] + next_test
            else:
                self._dose, next_test, _, _ = self._aurora_dosing_table(day_2_INR, self._dose)
                # self._retest_day = patient['day'] + next_test
                # self._retest_day = 5
        else:
            self._number_of_stable_days, next_test = self._aurora_retesting_table(patient['INRs'][-1], self._number_of_stable_days, self._early_therapeutic)

        if next_test == -1:
            self._early_therapeutic = False
            self._number_of_stable_days = 0
            self._dose, next_test, self._skip_dose, self._red_flag = self._aurora_dosing_table(patient['INRs'][-1], self._dose)

        self._retest_day = patient['day'] + next_test

        return self._dose if self._skip_dose == 0 else 0.0

    def _aurora_dosing_table(self, current_INR: float, dose: float) -> Tuple[float, int, int, bool]:
        skip_dose = 0
        red_flag = False
        if current_INR < 1.50:
            dose = dose * 1.15
            next_test = 7
        elif current_INR < 1.80:
            dose = dose * 1.10
            next_test = 7
        elif current_INR < 2.00:
            dose = dose * 1.075
            next_test = 7
        elif current_INR <= 3.00:
            next_test = 28
        elif current_INR < 3.40:
            dose = dose * 0.925
            next_test = 7
        elif current_INR < 4.00:
            dose = dose * 0.9
            next_test = 7
        elif current_INR <= 5.00:
            skip_dose = 1  # 2
            dose = dose * 0.875
            next_test = 7
        else:
            red_flag = True
            next_test = 2
            dose = dose * 0.85
        
        return dose, next_test, skip_dose, red_flag

    def _aurora_retesting_table(self, current_INR: float, number_of_stable_days: int, early_therapeutic: bool = False) -> Tuple[int, int]:
        # next_test = {0: 1, 1: 1, 2: 5, 7: 7, 14: 14, 28: 28}
        # NOTE: When patient gets into the range early on (before the maintenance period), we have 1: 2,
        # but when patient is in maintenance, we have 1: 6. I track the former in the main _aurora method.

        if early_therapeutic:
            next_test = {0: 1, 1: 2, 3: 4, 7: 6, 13: 6, 19: 13, 32: 27}
            max_gap = 32
        else:
            next_test = {0: 1, 1: 6, 7: 6, 13: 13, 26: 27}
            max_gap = 26
        if 2.0 <= current_INR <= 3.0:
            # number_of_stable_days = min(number_of_stable_days + next_test[number_of_stable_days], 28)
            number_of_stable_days = min(number_of_stable_days + next_test[number_of_stable_days], max_gap)
        else:
            return -1, -1

        return number_of_stable_days, next_test[number_of_stable_days]

    def _iwpc_clinical(self, patient: Dict[str, Any]) -> float:  # only the initial dose (day <= 3)
        if self._weekly_dose == 0:
            self._weekly_dose = (4.0376
                        - 0.2546 * (patient['age'] // 10)
                        + 0.0118 * patient['height'] * 2.54  # in to cm
                        + 0.0134 * patient['weight'] * 0.454  # lb to kg
                        - 0.6752 * (patient['race'] == 'Asian')
                        + 0.4060 * (patient['race'] == 'Black')
                        # + 0.0443 * (patient['race'] not in  [...])  # missing or mixed race
                        + 1.2799 * 0  # Enzyme inducer status (Fluvastatin is reductant not an inducer!)
                        - 0.5695 * (patient['amiodarone'] == 'Yes')) ** 2
            self._dose = self._weekly_dose / 7

        if patient['day'] > 3:
            self._logger.warning('_iwpc_clinical is called on a day > 3.')
    
        return self._dose

    def _iwpc_pg(self, patient: Dict[str, Any]) -> float:  # only the initial dose (day <= 3)
        if self._weekly_dose == 0:
            self._weekly_dose = (5.6044
                        - 0.2614 * (patient['age'] // 10)  # Based on Ravvaz (EU-PACT (page 18) uses year, Ravvaz (page 10 of annex) uses decades!
                        + 0.0087 * patient['height'] * 2.54  # in to cm
                        + 0.0128 * patient['weight'] * 0.454  # lb to kg
                        - 0.8677 * (patient['VKORC1'] == 'G/A')
                        - 1.6974 * (patient['VKORC1'] == 'A/A')
                        - 0.4854 * int(patient['VKORC1'] not in  ['G/A', 'A/A', 'G/G'])  # Not in EU-PACT ?!!
                        - 0.5211 * (patient['CYP2C9'] == '*1/*2')
                        - 0.9357 * (patient['CYP2C9'] == '*1/*3')
                        - 1.0616 * (patient['CYP2C9'] == '*2/*2')
                        - 1.9206 * (patient['CYP2C9'] == '*2/*3')
                        - 2.3312 * (patient['CYP2C9'] == '*3/*3')
                        - 0.2188 * int(patient['CYP2C9'] not in ['*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', '*1/*1'])  # Not in EU-PACT
                        - 0.1092 * (patient['race'] == 'Asian')  # Not in EU-PACT
                        - 0.2760 * (patient['race'] == 'Black')  # Not in EU-PACT
                        # - 1.0320 * (patient['race'] not in  [...])  # missing or mixed race - Not in EU-PACT
                        + 1.1816 * 0  # Enzyme inducer status (Fluvastatin is reductant not an inducer!) (comment by Ravvaz)
                        - 0.5503 * (patient['amiodarone'] == 'Yes')) ** 2
            self._dose = self._weekly_dose / 7

        if patient['day'] > 3:
            self._logger.warning('_iwpc_pg is called on a day > 3.')

        return self._dose

    def _modified_iwpc_pg(self, patient: Dict[str, Any]) -> float:
        if self._weekly_dose == 0:
            self._weekly_dose = (5.6044
                        - 0.2614 * (patient['age'] / 10)  # Based on Ravvaz (EU-PACT (page 18) uses year, Ravvaz (page 10 of annex) uses decades!
                        + 0.0087 * patient['height'] * 2.54  # in to cm
                        + 0.0128 * patient['weight'] * 0.454  # lb to kg
                        - 0.8677 * (patient['VKORC1'] == 'G/A')
                        - 1.6974 * (patient['VKORC1'] == 'A/A')
                        - 0.5211 * (patient['CYP2C9'] == '*1/*2')
                        - 0.9357 * (patient['CYP2C9'] == '*1/*3')
                        - 1.0616 * (patient['CYP2C9'] == '*2/*2')
                        - 1.9206 * (patient['CYP2C9'] == '*2/*3')
                        - 2.3312 * (patient['CYP2C9'] == '*3/*3')
                        - 0.5503 * (patient['amiodarone'] == 'Yes')) ** 2

        k = {'*1/*1': 0.0189,
                '*1/*2': 0.0158,
                '*1/*3': 0.0132,
                '*2/*2': 0.0130,
                '*2/*3': 0.0090,
                '*3/*3': 0.0075
                }
        LD3 = self._weekly_dose / ((1 - exp(-24*k[patient['CYP2C9']])) * (1 + exp(-24*k[patient['CYP2C9']]) + exp(-48*k[patient['CYP2C9']])))
        # The following dose calculation is based on EU-PACT report page 19
        # Ravvaz uses the same formula, but uses weekly dose. However, EU-PACT explicitly mentions "predicted daily dose (D)" 
        if patient['day'] == 1:
            self._dose = (1.5 * LD3 - 0.5 * self._weekly_dose) / 7
        elif patient['day'] == 2:
            self._dose = LD3 / 7
        elif patient['day'] == 3:
            self._dose = (0.5 * LD3 + 0.5 * self._weekly_dose) / 7
        else:
            self._logger.warning('_modified_iwpc_pg is called on a day > 3.')

        return self._dose

    def _lenzini_pg(self, patient: Dict[str, Any]) -> float:
        if self._lenzini_on_day_4:
            self._lenzini_on_day_4 = False
            return self._dose

        if patient['day'] == 4:
            self._lenzini_on_day_4 = True

        self._dose = exp(3.10894 
                        - 0.00767 * patient['age']
                        - 0.51611 * log(patient['INRs'][-1])  # Natural log
                        - 0.23032 * (1 * (patient['VKORC1'] == 'G/A')  # Heterozygous
                                   + 2 * (patient['VKORC1'] == 'A/A'))  # Homozygous
                        - 0.14745 * (1 * (patient['CYP2C9'] in ['*1/*2', '*2/*3'])  # Heterozygous
                                   + 2 * (patient['CYP2C9'] == '*2/*2'))  # Homozygous
                        - 0.30770 * (1 * (patient['CYP2C9'] in ['*1/*3', '*2/*3'])  # Heterozygous
                                   + 2 * (patient['CYP2C9'] == '*3/*3'))  # Homozygous
                        + 0.24597 * sqrt(patient['height'] * 2.54 * patient['weight'] * 0.454 / 3600)  # BSA
                        + 0.26729 * 2.5  # target INR
                        - 0.10350 * (patient['amiodarone'] == 'Yes')
                        + 0.01690 * patient['Doses'][-2]
                        + 0.02018 * patient['Doses'][-3]
                        + 0.01065 * patient['Doses'][-4]  # available if INR is measured on day 5
                        ) / 7

        return self._dose

    def _intermountain(self, patient: Dict[str, Any]) -> float:  # only maintenance dose (day >= 6)
        current_INR = patient['INRs'][-1]
        # if patient['day'] < 8:
        #     pass
        if self._red_flag:
            if self._retest_day > patient['day']:
                return_value = 0.0
            elif current_INR > 3.0:
                self._retest_day = patient['day'] + 2
                return_value = 0.0
            else:
                self._red_flag = False
                self._retest_day = patient['day'] + 7  # second dosing interval (14 days) happens automatically
                return_value = self._dose

        else:
            self._dose, next_test, self._skip_dose, self._red_flag, today_dose = \
                self._intermountain_dosing_table(current_INR, self._dose)

            if patient['day'] <= 8 and 2.0 <= current_INR <= 3.0:
                next_test = 14

            self._retest_day = patient['day'] + next_test

            if today_dose is not None:
                return_value = today_dose
            elif self._skip_dose == 0:
                return_value = self._dose
            else:
                return_value = 0.0

        return return_value

    def _intermountain_dosing_table(self,
        current_INR: float, dose: float)-> Tuple[float, int, int, bool, Union[float, None]]:

        skip_dose: int = 0
        today_dose: Union[float, None] = None
        red_flag: bool = False
        if current_INR < 1.60:
            today_dose = dose  # this should be average 5-7 for day 8
            dose = dose * 1.10
            next_test = 5 if self._visited_range != 1.60 else 14
            self._visited_range = 1.60
        elif current_INR < 1.80:
            today_dose = dose * 0.5 # this should be average 5-7 for day 8
            dose = dose * 1.05
            next_test = 7 if self._visited_range != 1.80 else 14
            self._visited_range = 1.80
        elif current_INR < 2.00:
            dose = dose * 1.05
            next_test = 28  # for day<8, I change next_test to 14 in _intermountain()
        elif current_INR <= 3.00:
            next_test = 14
        elif current_INR < 3.40:
            dose = dose * 0.95
            next_test = 14
        elif current_INR < 5.00:
            today_dose = dose * 0.5 if current_INR < 4 else 0.0
            dose = dose * 0.90
            next_test = 7 if self._visited_range != 5.00 else 14
            self._visited_range = 5.00
        else:
            red_flag = True
            skip_dose = 1
            next_test = 2
            dose = dose * 0.85

        return dose, next_test, skip_dose, red_flag, today_dose

    def _action(self, method: str, state: rldata.RLData) -> float:
        '''
        This method will be depricated. Now each dosing protocol is a separate function.
        '''
        
        self._logger.warning('_action is a depricated function. Use dedicated dosing protocol functions instead.')

        patient = state.value

        if patient['day'] <= 1:
            self.reset()
            self._retest_day = patient['day']
        elif self._retest_day > patient['day'] and not self._red_flag:
            if self._skip_dose > 0:
                self._skip_dose -= 1
                return 0.0
            else:
                return self._dose

        if method.lower() == 'aurora':
            if self._red_flag:
                if self._retest_day > patient['day']:
                    return 0.0
                elif patient['INRs'][-1] > 3.0:
                    self._retest_day = patient['day'] + 2
                    return 0.0
                else:
                    self._red_flag = False
                    self._retest_day = patient['day'] + 7
                    return self._dose

            if patient['day'] <= 2:
                self._dose = 10.0 if patient['age'] < 65.0 else 5.0
            elif patient['day'] <= 4:
                day_2_INR = patient['INRs'][-1] if patient['day'] == 3 else patient['INRs'][-2]
                if day_2_INR >= 2.0:
                    self._dose = 5.0
                else:
                    self._dose, _, _, _ = self._aurora_dosing_table(day_2_INR, self._dose)
                    self._retest_day = 5
            else:
                self._number_of_stable_days, next_test = self._aurora_retesting_table(patient['INRs'][-1], self._number_of_stable_days)
                if next_test == -1:
                    self._number_of_stable_days = 0
                    self._dose, next_test, self._skip_dose, self._red_flag = self._aurora_dosing_table(patient['INRs'][-1], self._dose)
                
                self._retest_day = patient['day'] + next_test

            if self._skip_dose == 0:
                return self._dose
            else:
                return 0.0

        elif method.lower() in ('iwpc_clinical', 'iwpc', 'iwpc clinical'):
            if self._weekly_dose == 0:
                self._weekly_dose = (4.0376
                            - 0.2546 * patient['age'] / 10
                            + 0.0118 * patient['height'] * 2.54  # in to cm
                            + 0.0134 * patient['weight'] * 0.454  # lb to kg
                            - 0.6752 * (patient['race'] == 'Asian')
                            + 0.4060 * (patient['race'] == 'Black')
                            + 0.0443 * int(patient['race'] not in  ['Asian', 'Black'])
                            + 1.2799 * 0  # Enzyme inducer status (Fluvastatin is reductant not an inducer!)
                            - 0.5695 * (patient['amiodarone'] == 'Yes')) ** 2
                self._dose = self._weekly_dose / 7

            if patient['day'] <= 3:
                return self._dose

        elif method.lower() in ('iwpc_pg', 'iwpc pg'):
            if self._weekly_dose == 0:
                self._weekly_dose = (5.6044
                            - 0.02614 * patient['age'] / 10  # Based on EU-PACT report page 18 (Ravvaz has a typo!)
                            + 0.0087 * patient['height'] * 2.54  # in to cm
                            + 0.0128 * patient['weight'] * 0.454  # lb to kg
                            - 0.8677 * (patient['VKORC1'] == 'G/A')
                            - 1.6974 * (patient['VKORC1'] == 'A/A')
                            - 0.4854 * int(patient['VKORC1'] not in  ['G/A', 'A/A', 'G/G'])  # Not in EU-PACT ?!!
                            - 0.5211 * (patient['CYP2C9'] == '*1/*2')
                            - 0.9357 * (patient['CYP2C9'] == '*1/*3')
                            - 1.0616 * (patient['CYP2C9'] == '*2/*2')
                            - 1.9206 * (patient['CYP2C9'] == '*2/*3')
                            - 2.3312 * (patient['CYP2C9'] == '*3/*3')
                            - 0.2188 * int(patient['CYP2C9'] not in ['*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', '*1/*1'])  # Not in EU-PACT
                            - 0.1092 * (patient['race'] == 'Asian')  # Not in EU-PACT
                            - 0.2760 * (patient['race'] == 'Black')  # Not in EU-PACT
                            - 1.0320 * int(patient['race'] not in  ['Asian', 'Black'])  # Not in EU-PACT
                            + 1.1816 * 0  # Enzyme inducer status (Fluvastatin is reductant not an inducer!)
                            - 0.5503 * (patient['amiodarone'] == 'Yes')) ** 2
                self._dose = self._weekly_dose / 7

            if patient['day'] <= 3:
                return self._dose

        elif method.lower() in ('modified_iwpc_pg', 'modified iwpc pg'):
            if self._weekly_dose == 0:
                self._weekly_dose = (5.6044
                            - 0.02614 * patient['age'] / 10  # Based on EU-PACT report page 18 (Ravvaz has a typo!)
                            + 0.0087 * patient['height'] * 2.54  # in to cm
                            + 0.0128 * patient['weight'] * 0.454  # lb to kg
                            - 0.8677 * (patient['VKORC1'] == 'G/A')
                            - 1.6974 * (patient['VKORC1'] == 'A/A')
                            - 0.5211 * (patient['CYP2C9'] == '*1/*2')
                            - 0.9357 * (patient['CYP2C9'] == '*1/*3')
                            - 1.0616 * (patient['CYP2C9'] == '*2/*2')
                            - 1.9206 * (patient['CYP2C9'] == '*2/*3')
                            - 2.3312 * (patient['CYP2C9'] == '*3/*3')
                            - 0.5503 * (patient['amiodarone'] == 'Yes')) ** 2

            k = {'*1/*1': 0.0189,
                 '*1/*2': 0.0158,
                 '*1/*3': 0.0132,
                 '*2/*2': 0.0130,
                 '*2/*3': 0.0090,
                 '*3/*3': 0.0075
                 }
            LD3 = self._weekly_dose / ((1 - exp(-24*k[patient['CYP2C9']])) * (1 + exp(-24*k[patient['CYP2C9']]) + exp(-48*k[patient['CYP2C9']])))
            # The following dose calculation is based on EU-PACT report page 19
            # Ravvaz uses the same formula, but uses weekly dose. However, EU-PACT explicitly mentions "predicted daily dose (D)" 
            if patient['day'] == 1:
                self._dose = 1.5 * LD3 - 0.5 * self._weekly_dose / 7
            elif patient['day'] == 2:
                self._dose = LD3
            else:
                self._dose = 0.5 * LD3 + 0.5 * self._weekly_dose / 7

        elif method.lower() in ('lenzini_pg', 'lenzini pg') and patient['day'] in (4, 5):
            if self._lenzini_on_day_4:
                self._lenzini_on_day_4 = False
                return self._dose

            if patient['day'] == 4:
                self._lenzini_on_day_4 = True

            self._dose = exp(3.10894
                            - 0.00767 * patient['age']
                            - 0.51611 * log(patient['INRs'][-1])
                            - 0.23032 * (patient['VKORC1'] == 'G/A')
                            - 0.14745 * int(patient['CYP2C9'] in ['*1/*2', '*2/*2'])
                            - 0.30770 * int(patient['CYP2C9'] in ['*1/*3', '*2/*3', '*3/*3'])
                            + 0.24597 * sqrt(patient['height'] * 2.54 * patient['weight'] * 0.454 / 3600)  # BSA
                            + 0.26729 * 2.5  # target INR
                            - 0.10350 * (patient['amiodarone'] == 'Yes')
                            + 0.01690 * patient['Doses'][-2]
                            + 0.02018 * patient['Doses'][-3]
                            + 0.01065 * patient['Doses'][-4]  # available if INR is measured on day 5
                            ) / 7

        return self._dose

    def reset(self) -> None:
        self._retest_day = 1
        self._red_flag = False
        self._lenzini_on_day_4 = False
        self._skip_dose = 0
        self._dose = 0
        self._weekly_dose = 0
        self._number_of_stable_days = 0
        self._early_therapeutic = False

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

    def __repr__(self) -> str:
        try:
            return f'WarfarinAgent: arm: {self._study_arm} day: {self._day}'
        except NameError:
            return 'WarfarinAgent'
