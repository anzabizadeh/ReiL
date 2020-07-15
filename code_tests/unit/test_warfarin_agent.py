import re
import unittest

import pandas as pd
from pathlib import Path

from tqdm import tqdm

from rl.subjects import WarfarinModel_v5
from rl.agents import WarfarinAgent


class testWarfarinAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dataset_path = 'C:/Users/Sadjad/OneDrive - University of Iowa/2. Current Research/Anticoagulation/dateset'
        cls._patient_profiles_filename = './UIOWA_KR_clinicalavatars10k_ageUB.csv'
        cls._patient_profiles = pd.read_csv(Path(cls._dataset_path, cls._patient_profiles_filename), index_col='ID')
        cls._files_in_arm_end = 2  # 41
        cls._output_files = [f'./arm1/ahc_avatars_ahc_algorithm_{i}.txt' for i in range(1, cls._files_in_arm_end)] + \
                    [f'./arm2/ahc_avatars_clinitialIWPC_ahc_ahc_algorithm_{i}.txt' for i in range(1, cls._files_in_arm_end)] + \
                    [f'./arm3/ahc_avatars_pginitialIWPC_ahc_ahc_algorithm_{i}.txt' for i in range(1, cls._files_in_arm_end)] + \
                    [f'./arm4/ahc_avatars_eupactInitial_eupactAlteration_Intermt_algorithm_{i}.txt' for i in range(1, cls._files_in_arm_end)] + \
                    [f'./arm5/ahc_avatars_eupactInitial_eupactAlteration_ahc_algorithm_{i}.txt' for i in range(1, cls._files_in_arm_end)]

    def test_aaa(self):
        output_files = [f'./arm1/ahc_avatars_ahc_algorithm_{i}.txt' for i in range(1, self._files_in_arm_end)]
        # agent_name = ''

        for filename in tqdm(output_files):
            df = pd.read_csv(Path(self._dataset_path, filename), delimiter='|')
            # agent_name = re.match('\S+ahc_avatars_(\w+)_algorithm_\d+.txt$', filename).group(1)
            counter_indexes = list(int(re.findall('\d+$' ,col)[0]) for col in df.filter(regex=f'ID\.(\d+)$').columns)
            counter_start = min(counter_indexes)
            counter_end = max(counter_indexes) + 1
            ID_list = dict((counter, df.filter(regex=f'ID\.{counter}$').iat[0, 0]) for counter in range(counter_start, counter_end))

            for counter in range(counter_start, counter_end):
                patient_info = self._patient_profiles.loc[ID_list[counter]]  # , ['AGE', 'CYP2C9', 'VKORC1']]
                dose_info = df.filter(regex=f'(INR|Dose)\.{counter}$').rename(columns=lambda x: re.findall('^\w+' ,x)[0])

                print(patient_info)

                w = WarfarinModel_v5(characteristics={'age': patient_info.AGE, 'weight': patient_info.WEIGHT, 'height': patient_info.HEIGHT,
                                                      'CYP2C9': patient_info.CYP2C9, 'VKORC1': patient_info.VKORC1,
                                                      'gender': patient_info.GENDER, 'race': patient_info.RACE,
                                                      'tobaco': {'N': 'No', 'Y': 'Yes'}[patient_info.SMOKER],
                                                      'amiodarone': {'N': 'No', 'Y': 'Yes'}[patient_info.AMI],
                                                      'fluvastatin': {'N': 'No', 'Y': 'Yes'}[patient_info.FLUVASTATIN]},
                                     patient_selection='fixed',
                                     ex_protocol_current={ 'state': 'extended','possible_actions': 'standard','take_effect': 'no_reward' })
                a = WarfarinAgent(study_arm='AAA')
                w._INR_history[-1] = dose_info['INR'][0]
                w._day += 1  # BUG: WarfarinAgent starts from day=1, WarfarinModel_v5 starts from day=0.
                for i in range(1, dose_info.shape[0]):
                    action = a.act(w.state)
                    w._INR_history.append(dose_info['INR'][i])
                    w._INR_history.popleft()
                    w._day += 1
                    print(f'INR: {w._INR_history[-2]}\t Action from file: {dose_info["Dose"][i-1]}\tAction: {action[0]}')
                    self.assertEqual(action[0], dose_info['Dose'][i-1])

        # self.assertEqual(rl_value._value, data)
        # self.assertEqual(rl_value.value, data)


if __name__ == "__main__":
    unittest.main()