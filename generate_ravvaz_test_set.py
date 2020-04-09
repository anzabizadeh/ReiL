# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
import numpy as np
import re
from pathlib import Path
from rl.rldata import RLData
from rl.stats import WarfarinStats
from tqdm import tqdm
from rl.subjects import WarfarinModelFixedInterval

# %%
dataset_path = 'C:/Users/Sadjad/OneDrive - University of Iowa/2. Current Research/Anticoagulation/dateset'
patient_profiles_filename = './UIOWA_KR_clinicalavatars10k_ageUB.csv'
patient_profiles = pd.read_csv(Path(dataset_path, patient_profiles_filename), index_col='ID')

# %%
for patient_id, patient_info in patient_profiles.iterrows():
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
    patient = WarfarinModelFixedInterval(characteristics=characteristics, patient_selection='')
    patient.save(path='./ravvaz_test_set', filename=str(patient_id))

# # %%
# for filename in tqdm(output_files):
#     df = pd.read_csv(Path(dataset_path, filename), delimiter='|')
#     agent_name_old = agent_name
#     agent_name = re.match('\S+ahc_avatars_(\w+)_algorithm_\d+.txt$', filename).group(1)
#     counter_indexes = list(int(re.findall('\d+$' ,col)[0]) for col in df.filter(regex=f'ID\.(\d+)$').columns)  #.iat[0, 0]
#     counter_start = min(counter_indexes)
#     counter_end = max(counter_indexes) + 1
#     ID_list = dict((counter, df.filter(regex=f'ID\.{counter}$').iat[0, 0]) for counter in range(counter_start, counter_end))

#     for counter in range(counter_start, counter_end):
#         patient_info = patient_profiles.loc[ID_list[counter], ['AGE', 'CYP2C9', 'VKORC1']]
#         dose_info = df.filter(regex=f'(INR|Dose)\.{counter}$').rename(columns=lambda x: re.findall('^\w+' ,x)[0])
#         dose_info['INR'] = dose_info['INR'].shift(periods=-1)
#         stat_subject = stats(stats_to_collect, patient_info=patient_info, dose_response_df=dose_info)
#         if stat_subject != {}:
#             try:
#                 stats_temp[(agent_name, 'ravvaz')].append(stat_subject)
#             except KeyError:
#                 stats_temp[(agent_name, 'ravvaz')] = [stat_subject]

#         if agent_name != agent_name_old:
#             result = warf_stats.aggregate(agent_stats=None, subject_stats=stats_temp[(agent_name, 'ravvaz')])
#             try:
#                 stats_final[(agent_name, 'ravvaz')].append(result)
#             except KeyError:
#                 stats_final[(agent_name, 'ravvaz')] = [result]

# # %%
# with open(f'./ravvaz_10000.txt', 'a+') as f:
#     for k1, v1 in stats_final.items():
#         for l in v1:
#             for k2, v2 in l.items():
#                 for row in range(v2.shape[0]):
#                     for col in range(v2.shape[1]):
#                         # print(f'{k1[0]}\t{k1[1]}\t{k2}\t{v2.columns[col]}\t{v2.index[row][0]}\t{v2.index[row][1]}\t{v2.iat[row, col]}')
#                         f.write(f'{k1[0]}\t{k1[1]}\t{k2}\t{v2.columns[col]}\t{v2.index[row][0]}\t{v2.index[row][1]}\t{v2.iat[row, col]}\n')
