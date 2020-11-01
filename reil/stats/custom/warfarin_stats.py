import operator as op
import re
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from reil import rlbase, stats

conditionals = {'<=': op.le,
                '>=': op.ge,
                '==': op.eq,
                '!=': op.ne,
                '<': op.lt,
                '>': op.gt}


def sensitivity(row):
    combo = row.ID.value['CYP2C9'] + row.ID.value['VKORC1']
    return (int(combo in ('*1/*1G/G', '*1/*2G/G', '*1/*1G/A')) * 1 +
            int(combo in ('*1/*2G/A', '*1/*3G/A', '*2/*2G/A',
                          '*2/*3G/G', '*1/*3G/G', '*2/*2G/G',
                          '*1/*2A/A', '*1/*1A/A')) * 2 +
            int(combo in ('*3/*3G/G',
                          '*3/*3G/A', '*2/*3G/A',
                          '*3/*3A/A', '*2/*3A/A', '*2/*2A/A', '*1/*3A/A')) * 4)


class WarfarinStats(stats.Stats):
    def __init__(self,
                 active_stats: Union[Sequence[str], str] = 'all',
                 groupby: Optional[List[str]] = None,
                 aggregators: Optional[List[str]] = ['mean', 'std']):
        super().__init__(active_stats=active_stats,
                         groupby=groupby,
                         aggregators=aggregators,
                         all_stats=('TTR', 'TTR>0.65', 'dose_change', 'count',
                                    'INR', 'INR_percent_dose_change'))

    def from_history(self, history: rlbase.History) -> Dict[str, pd.DataFrame]:
        df_from_history = pd.DataFrame(history)
        df_from_history['interval'] = df_from_history.apply(
            lambda row: row['state']['Intervals'][-1], axis=1)
        df_from_history['age'] = df_from_history.apply(
            lambda row: row['state']['age'][0], axis=1)
        df_from_history['CYP2C9'] = df_from_history.apply(
            lambda row: row['state']['CYP2C9'][0], axis=1)
        df_from_history['VKORC1'] = df_from_history.apply(
            lambda row: row['state']['VKORC1'][0], axis=1)
        df_from_history['dose_current'] = df_from_history.apply(
            lambda row: row['action'][0], axis=1)
        df_from_history['INR_current'] = df_from_history.apply(
            lambda row: row['state']['INRs'][-1], axis=1)

        df_temp = []
        for id in df_from_history.instance_id.unique():
            temp_INR = []
            temp_action = []

            section = df_from_history[df_from_history.instance_id == id]
            section.reset_index(inplace=True)
            section['day'] = section.interval.expanding(1).sum().astype(int)

            for i, j in zip(section.day[:-1], section.day[1:]):
                s = float(section.loc[section.day == i, 'INR_current'])  # type: ignore
                t = float(section.loc[section.day == j, 'INR_current'])  # type: ignore
                for k in range(0, j-i):
                    temp_INR.append(s + (t-s)*k/(j-i))
                temp_action += [
                    float(section.loc[section.day == i,
                                      'dose_current'])] * (j-i)  # type: ignore
            temp_INR.append(section.iloc[-1]['INR_current'])
            temp_action.append(section.iloc[-1]['dose_current'])

            section.drop(columns=['index', 'INR_current', 'dose_current',
                                  'interval', 'state', 'day', 'action', 'reward'],
                                  inplace=True)
            df_temp.append(pd.DataFrame({'day': range(1, len(temp_INR)+1),
                                         'INR': temp_INR,
                                         'action': temp_action,
                                         'previous_action': [0] + temp_action[:-1]}
                                       ).join(section))
            df_temp[-1][section.columns] = df_temp[-1][section.columns].ffill()
            df_temp[-1] = df_temp[-1][list(section.columns) +
                                      ['day', 'INR', 'previous_action', 'action']]

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
        df.replace({'sensitivity': {1: 'normal', 2: 'sensitive',
                                    4: 'highly sensitive'}}, inplace=True)

        results = {}
        temp_group_by = [] if self._groupby is None else self._groupby
        grouped_df = df.groupby(temp_group_by if 'instance_id' in temp_group_by
                                else ['instance_id'] + temp_group_by)

        for stat in self._active_stats:
            if stat == 'TTR':
                temp = grouped_df['TTR'].mean().groupby(self._groupby)
            elif stat[:4] == 'TTR>':
                temp = grouped_df['TTR'].mean().apply(
                    lambda x: int(x > float(stat[4:]))).groupby(self._groupby)
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
                temp.mean().rename(f'{stat}_mean'),  # type: ignore
                temp.std().rename(f'{stat}_stdev')])  # type: ignore
 
            results[stat] = stat_temp

        return results

    def aggregate(self,
                  subject_stats: Dict[str, Any],
                  agent_stats: None = None) -> Dict[str, pd.DataFrame]:

        df = pd.DataFrame.from_dict(subject_stats)

        if 'age' in self._groupby:
            df['age'] = df['ID'].apply(lambda row: row['age'][-1])
        if 'CYP2C9' in self._groupby:
            df['CYP2C9'] = df['ID'].apply(lambda row: row['CYP2C9'][-1])
        if 'VKORC1' in self._groupby:
            df['VKORC1'] = df['ID'].apply(lambda row: row['VKORC1'][-1])
        if 'sensitivity' in self._groupby:
            df['sensitivity'] = df.apply(sensitivity, axis=1)
            df.replace({'sensitivity': {1: 'normal', 2: 'sensitive',
                                        4: 'highly sensitive'}}, inplace=True)

        if 'ID' in self._groupby:
            # since type(ID)=rldata and cannot be used in groupby, we replace it with row index.
            df['ID'] = df.index

        results = {}
        for g in self._groupby:
            g_split = re.split('(!=|<=|>=|==|<|>)', g)
            try:
                # use the first item in the corresponding column to convert the right-hand side into proper datatype
                rhs = type(df[g_split[0]].iat[0])(g_split[2])
                df[g] = df[g_split[0]].apply(
                    lambda row: conditionals[g_split[1]](row, rhs))
            except IndexError:
                pass

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

            stat_temp = temp.agg([(func, func) for func in self._aggregators])

            results[stat] = stat_temp

        return results
