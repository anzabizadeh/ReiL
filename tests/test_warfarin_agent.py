import re
import unittest
from pathlib import Path

import pandas as pd
from reil.agents import WarfarinAgent
from reil.subjects import Warfarin


class testWarfarinAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dataset_path = 'C:/Users/Sadjad/OneDrive - University of Iowa/2. Current Research/Anticoagulation/dateset'
        cls._patient_profiles_filename = './UIOWA_KR_clinicalavatars10k_ageUB.csv'
        cls._patient_profiles = pd.read_csv(Path(cls._dataset_path, cls._patient_profiles_filename), index_col='ID')
        cls._files_in_arm_end = 2  # 41
        cls._study_arms = {'aaa': './arm1/ahc_avatars_ahc_algorithm',
                    'caa': './arm2/ahc_avatars_clinitialIWPC_ahc_ahc_algorithm',
                    'pgaa': './arm3/ahc_avatars_pginitialIWPC_ahc_ahc_algorithm',
                    'pgpgi': './arm4/ahc_avatars_eupactInitial_eupactAlteration_Intermt_algorithm',
                    'pgpga': './arm5/ahc_avatars_eupactInitial_eupactAlteration_ahc_algorithm'}

        cls._rounding_precision = {'aaa': 9,
                    'caa': 5,
                    'pgaa': 2,
                    'pgpgi': 9,
                    'pgpga': 9}

        cls._exceptions_IDs = {'aaa': [1000038, 1000055, 1000086, 1000111, 1000143, 1000149, 1000171, 1000207, 1000211, 1000219],
                    'caa': [],
                    'pgaa': [],
                    'pgpgi': [],
                    'pgpga': []}


    def test_aaa(self):
        self._run_test('aaa')

    def test_caa(self):
        self._run_test('caa')

    def test_pgaa(self):
        self._run_test('pgaa')

    def test_pgpgi(self):
        self._run_test('pgpgi')

    def test_pgpga(self):
        self._run_test('pgpga')

    def _run_test(self, arm: str) -> None:
        output_files = [f'{self._study_arms[arm]}_{i}.txt' for i in range(1, self._files_in_arm_end)]
        # agent_name = ''
        raised = False

        for filename in output_files:
            df = pd.read_csv(Path(self._dataset_path, filename), delimiter='|')
            # agent_name = re.match('\S+ahc_avatars_(\w+)_algorithm_\d+.txt$', filename).group(1)
            counter_indexes = list(int(re.findall(r'\d+$' ,col)[0]) for col in df.filter(regex=r'ID\.(\d+)$').columns)
            counter_start = min(counter_indexes)
            counter_end = max(counter_indexes) + 1
            ID_list = dict((counter, df.filter(regex=f'ID\.{counter}$').iat[0, 0])
                           for counter in range(counter_start, counter_end))

            for counter in range(counter_start, counter_end):
                patient_info = self._patient_profiles.loc[ID_list[counter]]  # , ['AGE', 'CYP2C9', 'VKORC1']]
                dose_info = df.filter(regex=f'(INR|Dose|Check)\.{counter}$').rename(columns=lambda x: re.findall(r'^\w+' ,x)[0])

                # print(f'Patient ID: {patient_info.name}')

                w = Warfarin(characteristics={'age': patient_info.AGE, 'weight': patient_info.WEIGHT, 'height': patient_info.HEIGHT,
                                                      'CYP2C9': patient_info.CYP2C9, 'VKORC1': patient_info.VKORC1,
                                                      'gender': patient_info.GENDER,
                                                      'race': {'White': 'White',
                                                               'Black or African American': 'Black',
                                                               'American Indian or Alaskan Native': 'American Indian',
                                                               'Asian': 'Asian',
                                                               'Native Hawaiian/Other Pacific Islander': 'Pacific Islander'
                                                              }[patient_info.RACE],
                                                      'tobaco': {'N': 'No', 'Y': 'Yes'}[patient_info.SMOKER],
                                                      'amiodarone': {'N': 'No', 'Y': 'Yes'}[patient_info.AMI],
                                                      'fluvastatin': {'N': 'No', 'Y': 'Yes'}[patient_info.FLUVASTATIN]},
                                     patient_selection='fixed',
                                     ex_protocol_current={ 'state': 'extended','possible_actions': 'standard','take_effect': 'no_reward' },
                                     dose_history_length=5)
                a = WarfarinAgent(study_arm=arm)
                w._decision_points_INR_history[0] = 1.0  # dose_info['INR'][0]
                for i in range(dose_info.shape[0]):  # (1, dose_info.shape[0]):
                    action = a.act(w.state)
                    # if dose_info['INR'][i] == 2 and dose_info['INR'][i-1] < 2.0:
                    #     w._INR_history.append(dose_info['INR'][i] - 1e-3)
                    # elif dose_info['INR'][i] == 3 and dose_info['INR'][i-1] > 3.0:
                    #     w._INR_history.append(dose_info['INR'][i] + 1e-3)
                    # else:
                    #     w._INR_history.append(dose_info['INR'][i])
                    w._decision_points_dose_history[w._decision_points_index] = dose_info['Dose'][i]
                    w._decision_points_index += 1
                    w._decision_points_INR_history[w._decision_points_index] = dose_info['INR'][i]
                    w._day += 1
                    # print(f'INR: {w._INR_history[-2]}\t Action from file: {dose_info["Dose"][i-1]}\tAction: {action[0]}')
                    # self.assertEqual(action[0], dose_info['Dose'][i-1], msg=f'Patient ID: {patient_info.name}\tday: {w._day}\tAction from file: {dose_info["Dose"][i-1]}\tAction: {action[0]}')
                    # print(f'day: {w._day}\tINR: {w._INR_history[-2]}\tAction from file: {dose_info["Dose"][i]}\tAction: {action[0]}')
                    try:
                        self.assertEqual(round(action[0].value, self._rounding_precision[arm]), round(dose_info['Dose'][i], self._rounding_precision[arm]),
                            msg=f'\nPatient ID: {patient_info.name}\tday: {w._day}')
                    except AssertionError as e:
                        if abs(round(action[0].value, self._rounding_precision[arm]) - round(dose_info['Dose'][i], self._rounding_precision[arm])) > 2e-2 \
                            or dose_info['INR'][i] >= 5.0:
                            print(action[0].value, dose_info['Dose'][i], dose_info['INR'][i-1], list(dose_info['Dose'][:i]), list(patient_info[:-3]))
                            # print(str(e))
                            # if dose_info['Check'][i-1] > 0 or dose_info['Check'][i+1] > 0:
                            #     a._dose = dose_info["Dose"][i]
                            # else:
                            raised = True
                            break

        if raised:
            raise AssertionError


if __name__ == "__main__":
    unittest.main()
