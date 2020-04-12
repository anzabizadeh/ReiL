import re
from pathlib import Path

import numpy as np
import pandas as pd
from rl.rldata import RLData
from rl.stats import WarfarinStats
from rl.subjects import WarfarinModel_v5, WarfarinModelFixedInterval
from tqdm import tqdm

dataset_path = 'C:/Users/Sadjad/OneDrive - University of Iowa/2. Current Research/Anticoagulation/dateset'
patient_profiles_filename = './UIOWA_KR_clinicalavatars10k_ageUB.csv'
patient_profiles = pd.read_csv(Path(dataset_path, patient_profiles_filename), index_col='ID')


for patient_id, patient_info in tqdm(patient_profiles.iterrows()):
    characteristics = {'age': patient_info['AGE'],
                       'weight': patient_info['WEIGHT'],
                       'height': patient_info['HEIGHT'],
                       'gender': 'Male' if patient_info['GENDER'] == 'M' else 'Female',
                       'race': patient_info['RACE'],
                       'tobaco': 'Yes' if patient_info['SMOKER'] == 'Y' else 'No',
                       'amiodarone': 'Yes' if patient_info['AMI'] == 'Y' else 'No',
                       'fluvastatin': 'Yes' if patient_info['FLUVASTATIN'] == 'Y' else 'No',
                       'CYP2C9': patient_info['CYP2C9'],
                       'VKORC1': patient_info['VKORC1']
                       }
    patient_1 = WarfarinModelFixedInterval(characteristics=characteristics,
                                         patient_selection='')
    patient_1.save(path='./ravvaz_test_set', filename=str(patient_id))

    patient_2 = WarfarinModel_v5(characteristics=characteristics,
                                patient_selection='',
                                ex_protocol_current={'state': 'extended', 'take_effect': 'no_reward'})
    patient_2.save(path='./ravvaz_test_set_v5', filename=str(patient_id))
