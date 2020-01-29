import numpy as np
import pandas as pd

class WarfarinStats:
    def __init__(self, agent_stat_dict, **kwargs):
        '''
        Attributes:
        -----------
            agent_stat_dict: a dictionary that determines groupby, stats, etc. per agent_name in the form of a nested dictionary.
            Example: agent_stat_dict={'agent_1': {'stats': 'all', 'groupby': ['A/A', 'G/A', 'G/G']}}
        '''
        self._agent_stat_dict = agent_stat_dict
        for agent in self._agent_stat_dict.keys():
            if 'stats' not in self._agent_stat_dict[agent].keys():
                self._agent_stat_dict[agent]['stats'] = []
            if 'groupby' not in self._agent_stat_dict[agent].keys():
                self._agent_stat_dict[agent]['groupby'] = []
            if self._agent_stat_dict[agent]['stats'] == 'all':
                self._agent_stat_dict[agent]['stats'] = ['TTR', 'TTR>0.65', 'dose_change', 'count', 'INR', 'INR_percent_dose_change']

    def stats_func(self, agent_name, history):
        def sensitivity(row):
            combo = row['CYP2C9'] + row['VKORC1']
            return (int(combo in ('*1/*1G/G', '*1/*2G/G', '*1/*1G/A')) * 1 +
                    int(combo in ('*1/*2G/A', '*1/*3G/A', '*2/*2G/A',
                                  '*2/*3G/G', '*1/*3G/G', '*2/*2G/G',
                                  '*1/*2A/A', '*1/*1A/A')) * 2 +
                    int(combo in ('*3/*3G/G',
                                  '*3/*3G/A', '*2/*3G/A',
                                  '*3/*3A/A', '*2/*3A/A', '*2/*2A/A', '*1/*3A/A')) * 4)

        groupby = self._agent_stat_dict[agent_name]['groupby']
        df_from_history = pd.DataFrame(history)
        df_from_history['interval'] = df_from_history.apply(lambda row: row['state']['Intervals'][-1], axis=1)
        df_from_history['age'] = df_from_history.apply(lambda row: row['state']['age'][0], axis=1)
        df_from_history['CYP2C9'] = df_from_history.apply(lambda row: row['state']['CYP2C9'][0], axis=1)
        df_from_history['VKORC1'] = df_from_history.apply(lambda row: row['state']['VKORC1'][0], axis=1)
        df_from_history['dose_current'] = df_from_history.apply(lambda row: row['action'][0], axis=1)
        df_from_history['INR_current'] = df_from_history.apply(lambda row: row['state']['INRs'][-1], axis=1)

        df_temp = []
        for id in df_from_history.instance_id.unique():
            temp_INR = []
            temp_action = []

            section = df_from_history[df_from_history.instance_id == id]
            section.reset_index(inplace=True)
            section['day'] = section.interval.expanding(1).sum().astype(int)

            for i, j in zip(section.day[:-1], section.day[1:]):
                s = float(section.loc[section.day == i, 'INR_current'])
                t = float(section.loc[section.day == j, 'INR_current'])
                for k in range(0, j-i):
                    temp_INR.append(s + (t-s)*k/(j-i))
                temp_action += [float(section.loc[section.day == i, 'dose_current'])] * (j-i)
            temp_INR.append(section.iloc[-1]['INR_current'])
            temp_action.append(section.iloc[-1]['dose_current'])

            section.drop(columns=['index', 'INR_current', 'dose_current', 'interval', 'state', 'day', 'action', 'reward'], inplace=True)
            df_temp.append(pd.DataFrame({'day': range(1, len(temp_INR)+1), 'INR': temp_INR, 'action': temp_action, 'previous_action': [0] + temp_action[:-1]}).join(section))
            df_temp[-1][section.columns] = df_temp[-1][section.columns].ffill()
            df_temp[-1] = df_temp[-1][list(section.columns) + ['day', 'INR', 'previous_action', 'action']]

        df = pd.concat(df_temp, axis=0)
        df.reset_index(inplace=True)
        df.drop(columns=['index'], inplace=True)

        df['delta_dose'] = df.apply(
            lambda row: abs(row['action'] - row['previous_action']), axis=1)
        df['dose_change'] = df.apply(
            lambda row: int(row['action'] != row['previous_action']), axis=1)
        df['TTR'] = df.INR.apply(
            lambda x: 1 if 2 <= x <= 3 else 0)
        df['sensitivity'] = df.apply(sensitivity, axis=1)
        df.replace({'sensitivity': {1: 'normal', 2: 'sensitive', 4: 'highly sensitive'}}, inplace=True)

        results = {}
        grouped_df = df.groupby(groupby if 'instance_id' in groupby else ['instance_id'] + groupby)
                     
        for stat in self._agent_stat_dict[agent_name]['stats']:
            if stat == 'TTR':
                temp = grouped_df['TTR'].mean().groupby(groupby)
            elif stat[:4] == 'TTR>':
                temp = grouped_df['TTR'].mean().apply(lambda x: int(x > float(stat[4:]))).groupby(groupby)
            elif stat == 'dose_change':
                temp = grouped_df['dose_change'].mean().groupby(groupby)
            elif stat == 'count':
                temp = grouped_df['dose_change'].count().groupby(groupby)
            elif stat == 'delta_dose':
                temp = grouped_df['delta_dose'].mean().groupby(groupby)
            elif stat == 'INR':
                temp = grouped_df['INR']
            else:
                continue

            stat_temp = pd.DataFrame([
                temp.mean().rename(f'{stat}_mean'),
                temp.std().rename(f'{stat}_stdev')])
            # elif stat == 'INR_percent_dose_change':
            #     temp = df.groupby(['INR'] + groupby if 'instance_id' in groupby else ['instance_id'] + groupby)
            #     stat_temp = (temp['delta_dose'].sum() /
            #                     (temp['action'].sum()
            #                     - temp['delta_dose'].sum())).rename(stat)

            results[stat] = stat_temp

        return results
