import numpy as np
import pandas as pd
from ..stats import Stats

class WarfarinStats(Stats):
    def __init__(self, active_stats='all', groupby=[], aggregators=['mean', 'std'], **kwargs):
        super().__init__(active_stats=active_stats,
            groupby=groupby, aggregators=aggregators,
            all_stats=['TTR', 'TTR>0.65', 'dose_change', 'count', 'INR', 'INR_percent_dose_change'],
            **kwargs)

    def from_history(self, name, history):
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
        df['sensitivity'] = df.apply(self._sensitivity, axis=1)
        df.replace({'sensitivity': {1: 'normal', 2: 'sensitive', 4: 'highly sensitive'}}, inplace=True)

        results = {}
        grouped_df = df.groupby(self._groupby if 'instance_id' in self._groupby else ['instance_id'] + self._groupby)
                     
        for stat in self._active_stats:
            if stat == 'TTR':
                temp = grouped_df['TTR'].mean().groupby(self._groupby)
            elif stat[:4] == 'TTR>':
                temp = grouped_df['TTR'].mean().apply(lambda x: int(x > float(stat[4:]))).groupby(self._groupby)
            elif stat == 'dose_change':
                temp = grouped_df['dose_change'].mean().groupby(self._groupby)
            elif stat == 'count':
                temp = grouped_df['dose_change'].count().groupby(self._groupby)
            elif stat == 'delta_dose':
                temp = grouped_df['delta_dose'].mean().groupby(self._groupby)
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

    def aggregate(self, agent_stats=None, subject_stats=None):
        df = pd.DataFrame.from_dict(subject_stats)
        df['age'] = df.apply(lambda row: row['ID']['age'][-1], axis=1)
        df['CYP2C9'] = df.apply(lambda row: row['ID']['CYP2C9'][-1], axis=1)
        df['VKORC1'] = df.apply(lambda row: row['ID']['VKORC1'][-1], axis=1)
        df['sensitivity'] = df.apply(self._sensitivity, axis=1)
        df.replace({'sensitivity': {1: 'normal', 2: 'sensitive', 4: 'highly sensitive'}}, inplace=True)

        results = {}
        grouped_df = df.groupby(self._groupby)

        for stat in self._active_stats:
            if stat == 'TTR':
                temp = grouped_df['TTR']
            # elif stat[:4] == 'TTR>':
            #     temp = grouped_df['TTR'].mean().apply(lambda x: int(x > float(stat[4:]))).groupby(self._groupby)
            elif stat == 'dose_change':
                temp = grouped_df['dose_change']
            # elif stat == 'count':
            #     temp = grouped_df['dose_change'].count().groupby(self._groupby)
            elif stat == 'delta_dose':
                temp = grouped_df['delta_dose']
            # elif stat == 'INR':
            #     temp = grouped_df['INR']
            else:
                continue

            stat_temp = temp.agg([(f'{stat}_{func}', func) for func in self._aggregators])
                # pd.DataFrame([
                # temp.max().rename(f'{stat}_max'),
                # temp.min().rename(f'{stat}_min'),
                # temp.mean().rename(f'{stat}_mean'),
                # temp.std().rename(f'{stat}_stdev')])
            # elif stat == 'INR_percent_dose_change':
            #     temp = df.groupby(['INR'] + groupby if 'instance_id' in groupby else ['instance_id'] + groupby)
            #     stat_temp = (temp['delta_dose'].sum() /
            #                     (temp['action'].sum()
            #                     - temp['delta_dose'].sum())).rename(stat)

            results[stat] = stat_temp

        return results

    def _sensitivity(self, row):
        combo = row['CYP2C9'] + row['VKORC1']
        return (int(combo in ('*1/*1G/G', '*1/*2G/G', '*1/*1G/A')) * 1 +
                int(combo in ('*1/*2G/A', '*1/*3G/A', '*2/*2G/A',
                                '*2/*3G/G', '*1/*3G/G', '*2/*2G/G',
                                '*1/*2A/A', '*1/*1A/A')) * 2 +
                int(combo in ('*3/*3G/G',
                                '*3/*3G/A', '*2/*3G/A',
                                '*3/*3A/A', '*2/*3A/A', '*2/*2A/A', '*1/*3A/A')) * 4)
