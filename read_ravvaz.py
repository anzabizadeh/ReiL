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


# %%
dataset_path = 'C:/Users/Sadjad/OneDrive - University of Iowa/2. Current Research/Anticoagulation/dateset'
patient_profiles_filename = './UIOWA_KR_clinicalavatars10k_ageUB.csv'
files_in_arm_end = 41
output_files = [f'./arm1/ahc_avatars_ahc_algorithm_{i}.txt' for i in range(1, files_in_arm_end)] + \
               [f'./arm2/ahc_avatars_clinitialIWPC_ahc_ahc_algorithm_{i}.txt' for i in range(1, files_in_arm_end)] + \
               [f'./arm3/ahc_avatars_pginitialIWPC_ahc_ahc_algorithm_{i}.txt' for i in range(1, files_in_arm_end)] + \
               [f'./arm4/ahc_avatars_eupactInitial_eupactAlteration_Intermt_algorithm_{i}.txt' for i in range(1, files_in_arm_end)] + \
               [f'./arm5/ahc_avatars_eupactInitial_eupactAlteration_ahc_algorithm_{i}.txt' for i in range(1, files_in_arm_end)]


# %%
stats_to_collect = ['TTR', 'dose_change', 'delta_dose']
aggregators = ['min', 'max', 'mean','std', 'median']
warf_stats = WarfarinStats(active_stats=stats_to_collect,
                            aggregators=aggregators,
                            groupby=['sensitivity', 'age>65'])


# %%
def stats(stats_list, patient_info, dose_response_df):
    if isinstance(stats_list, str):
        stats_list = [stats_list]
    results = {}
    for s in stats_list:
        if s == 'TTR':
            INRs = dose_response_df.INR
            temp = sum((1 if 2.0<=INRi<=3.0 else 0 for INRi in INRs)) / len(INRs)
        elif s == 'dose_change':
            temp = np.sum(np.abs(np.diff(dose_response_df.Dose))>0)
        elif s == 'delta_dose':
            temp = np.sum(np.abs(np.diff(dose_response_df.Dose)))

        results[s] = temp
    results['ID'] = RLData({'age': patient_info['AGE'],
                            'CYP2C9': patient_info['CYP2C9'],
                            'VKORC1': patient_info['VKORC1']},
                            lower={'age': 18},
                            upper={'age': 100},
                            categories={'CYP2C9': ('*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3'),
                                        'VKORC1': ('G/G', 'G/A', 'A/A')}, lazy_evaluation=True)

    return results


# %%
patient_profiles = pd.read_csv(Path(dataset_path, patient_profiles_filename), index_col='ID')


# %%
agent_name = ''
stats_temp = {}
stats_final = {}

for filename in tqdm(output_files):
    df = pd.read_csv(Path(dataset_path, filename), delimiter='|')
    agent_name_old = agent_name
    agent_name = re.match('\S+ahc_avatars_(\w+)_algorithm_\d+.txt$', filename).group(1)
    counter_indexes = list(int(re.findall('\d+$' ,col)[0]) for col in df.filter(regex=f'ID\.(\d+)$').columns)  #.iat[0, 0]
    counter_start = min(counter_indexes)
    counter_end = max(counter_indexes) + 1
    ID_list = dict((counter, df.filter(regex=f'ID\.{counter}$').iat[0, 0]) for counter in range(counter_start, counter_end))

    for counter in range(counter_start, counter_end):
        patient_info = patient_profiles.loc[ID_list[counter], ['AGE', 'CYP2C9', 'VKORC1']]
        dose_info = df.filter(regex=f'(INR|Dose)\.{counter}$').rename(columns=lambda x: re.findall('^\w+' ,x)[0])
        dose_info['INR'] = dose_info['INR'].shift(periods=-1)
        stat_subject = stats(stats_to_collect, patient_info=patient_info, dose_response_df=dose_info)
        if stat_subject != {}:
            try:
                stats_temp[(agent_name, 'ravvaz')].append(stat_subject)
            except KeyError:
                stats_temp[(agent_name, 'ravvaz')] = [stat_subject]

        if agent_name != agent_name_old:
            result = warf_stats.aggregate(agent_stats=None, subject_stats=stats_temp[(agent_name, 'ravvaz')])
            try:
                stats_final[(agent_name, 'ravvaz')].append(result)
            except KeyError:
                stats_final[(agent_name, 'ravvaz')] = [result]

# %%
with open(f'./ravvaz_10000.txt', 'a+') as f:
    for k1, v1 in stats_final.items():
        for l in v1:
            for k2, v2 in l.items():
                for row in range(v2.shape[0]):
                    for col in range(v2.shape[1]):
                        # print(f'{k1[0]}\t{k1[1]}\t{k2}\t{v2.columns[col]}\t{v2.index[row][0]}\t{v2.index[row][1]}\t{v2.iat[row, col]}')
                        f.write(f'{k1[0]}\t{k1[1]}\t{k2}\t{v2.columns[col]}\t{v2.index[row][0]}\t{v2.index[row][1]}\t{v2.iat[row, col]}\n')
