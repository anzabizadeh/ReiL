# %%
import csv
import json
import pathlib
from os import listdir
from os.path import isfile, join
from random import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from dill import HIGHEST_PROTOCOL, dump, load
from matplotlib import patches
from rl.agents import WarfarinClusterAgent
from rl.environments import Environment
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

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
                self._agent_stat_dict[agent].update('stats', ['TTR', 'TTR>0.65', 'dose_change', 'count', 'INR', 'INR_percent_dose_change'])

        self._dose_history = kwargs.get('dose_history', 10)
        self._INR_history = kwargs.get('INR_history', 10)
        self._extended_state = kwargs.get('extended_state', False)
        if self._extended_state:
            self._column_names = ['age', 'weight', 'height', 'female', 'male', 'White', 'Black', 'Asian', 'American Indian', 'Pacific Islander',
            'tobaco_no', 'tobaco_yes', 'amiodarone_no', 'amiodarone_yes', 'fluvastatin_no', 'fluvastatin_yes',
            '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A']
        else:
            self._column_names = ['age', '*1/*1', '*1/*2', '*1/*3',
                '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A']
        self._df_for_stats = {}
        # self._X_embedded = {}
        # self._df_for_segmentation = {}
        # self._gmm = {}
        # self._gmm_results = {}

    def load_experiment_data(self, **kwargs):
        experiment_path = kwargs.get('experiment_path', self._experiment_path)
        print(experiment_path)
        self._dose_history = kwargs.get('dose_history', 10)
        self._INR_history = kwargs.get('INR_history', 10)
        if kwargs.get('reload', False):
            self._files_loaded = []
            self._all_data = pd.DataFrame()
            self._df_for_stats = {}  # to calculate TTR and other possible measures
            self._df_for_segmentation = {}  # a point is defined as a 90-day dose

        def sensitivity(row):
            if row['G/G'] * (row['*1/*1'] + row['*1/*2']) == 1:
                return 'normal'
            elif row['G/A'] * row['*1/*1'] == 1:
                return 'normal'
            elif row['G/A'] * (row['*1/*2'] + row['*1/*3'] + row['*2/*2']) == 1:
                return 'sensitive'
            elif row['G/G'] * (row['*1/*3'] + row['*2/*2'] + row['*2/*3']) == 1:
                return 'sensitive'
            elif row['A/A'] * (row['*1/*1'] + row['*1/*2']) == 1:
                return 'sensitive'
            elif row['G/G'] * row['*3/*3'] == 1:
                return 'highly sensitive'
            elif row['G/A'] * (row['*2/*3'] + row['*3/*3']) == 1:
                return 'highly sensitive'
            elif row['A/A'] * (row['*1/*3'] + row['*2/*2'] + row['*2/*3'] + row['*3/*3']):
                return 'highly sensitive'

        for f in listdir(experiment_path):
            full_filename = join(experiment_path, f)
            if isfile(full_filename) and full_filename not in self._files_loaded:
                agent, subject = f[:-4].split(sep='@', )
                print(agent, subject)
                df_from_file = pd.read_pickle(full_filename)[
                    ['state', 'action']]

                patient_info = df_from_file.iloc[0].state.normalize().as_list()
                if len(patient_info) == 1:
                    patient_info = list(
                        *df_from_file.iloc[0].state.normalize().value)

                try:
                    self._df_for_segmentation[agent].loc[len(self._df_for_segmentation[agent])] = \
                        patient_info[:10] + [float(a.value)
                                             for a in df_from_file.action]
                except KeyError:
                    self._df_for_segmentation[agent] = pd.DataFrame(dict(zip(self._column_names
                                                                             + [f'dose {i:02}' for i in range(90)],
                                                                             patient_info[:10] + [float(a.value) for a in df_from_file.action])),
                                                                    index=[0])

                INR_max = df_from_file.loc[0, 'state'].upper.loc['INRs']
                dose_max = df_from_file.loc[0, 'state'].upper.loc['Doses']
                df_state_action = df_from_file.apply(lambda row: [z for y in [x if hasattr(x, '__iter__') and not isinstance(
                    x, str) else [x] for x in row.state.normalize().value] for z in y] + row.action.as_list(), axis=1, result_type='expand')
                df_state_action.columns = self._column_names + [f'dose-{self._dose_history-i:02}' for i in range(self._dose_history)] + [f'INR-{self._INR_history-i:02}' for i in range(self._INR_history)] + ['INR_current', 'action']
                df_state_action['delta_dose']=df_state_action.apply(
                    lambda row: row['action'] - row['dose-01'] * dose_max, axis=1)
                df_state_action['dose_change'] = df_state_action.apply(
                    lambda row: int(row['action'] != row['dose-01']*dose_max), axis=1)
                df_state_action.INR_current = df_state_action.INR_current.apply(
                    lambda x: x * INR_max)
                df_state_action['TTR'] = df_state_action.INR_current.apply(
                    lambda x: 1 if 2 <= x <= 3 else 0)
                df_state_action['instance_id'] = subject
                df_state_action['agent'] = agent
                df_state_action['sensitivity'] = sensitivity(df_state_action.iloc[0, :])  # df_state_action.apply(lambda row: sensitivity(row), axis=1)
                df_state_action['VKORC1'] = df_state_action.loc[[0], ['A/A', 'G/A', 'G/G']].idxmax(axis=1)[0] # .apply(lambda row: row[['A/A', 'G/A', 'G/G']].idxmax(), axis=1)
                df_state_action['CYP2C9'] = df_state_action.loc[[0], ['*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3']].idxmax(axis=1)[0]  # .apply(lambda row: row[['*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3']].idxmax(), axis=1)

                df_state_action.drop([f'dose-{dose_history-i:02}' for i in range(dose_history)] + [
                                     f'INR-{INR_history-i:02}' for i in range(INR_history)], inplace=True, axis=1)
                # temp_list_stats[counter] = df_state_action
                try:
                    self._df_for_stats[agent] = pd.concat(
                        [self._df_for_stats[agent], df_state_action], ignore_index = True, sort = False)
                except KeyError:
                    self._df_for_stats[agent]=df_state_action

                self._files_loaded.append(full_filename)

        # self._df_for_segmentation = pd.concat([self._df_for_segmentation] + temp_list_segmentation)
        # self._df_for_stats = pd.concat([self._df_for_stats] + temp_list_stats)

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
        temp_df = pd.DataFrame(history)

        # df_state_action = df_from_file.apply(lambda row: [z for y in [x if hasattr(x, '__iter__') and not isinstance(
        #     x, str) else [x] for x in row.state.normalize().value] for z in y] + row.action.as_list(), axis=1, result_type='expand')
        temp_df['age'] = temp_df.apply(lambda row: row['state']['age'][0], axis=1)
        temp_df['CYP2C9'] = temp_df.apply(lambda row: row['state']['CYP2C9'][0], axis=1)
        temp_df['VKORC1'] = temp_df.apply(lambda row: row['state']['VKORC1'][0], axis=1)
        temp_df['Doses'] = temp_df.apply(lambda row: row['state']['Doses'], axis=1)
        temp_df['INRs'] = temp_df.apply(lambda row: row['state']['INRs'], axis=1)
        temp_df['Intervals'] = temp_df.apply(lambda row: row['state']['Intervals'], axis=1)
        temp_df['INR_current'] = temp_df.apply(lambda row: row['state']['INRs'][-1], axis=1)
        temp_df['delta_dose'] = temp_df.apply(
            lambda row: row['action'][0] - row['state']['Doses'][-1], axis=1)
        temp_df['dose_change'] = temp_df.apply(
            lambda row: row['action'][0] - row['state']['Doses'][-1], axis=1)
        temp_df['TTR'] = temp_df.INR_current.apply(
            lambda x: 1 if 2 <= x <= 3 else 0)
        temp_df['sensitivity'] = temp_df.apply(sensitivity, axis=1)
        temp_df.replace({'sensitivity': {1: 'normal', 2: 'sensitive', 4: 'highly sensitive'}}, inplace=True)
        
        results = {}
        for stat in self._agent_stat_dict[agent_name]['stats']:
            if stat == 'TTR':
                stat_temp = (temp_df.groupby(groupby).sum()['TTR'] /
                                temp_df.groupby(groupby).count()['TTR'])
            elif stat[:4] == 'TTR>':
                stat_temp_temp = (temp_df.groupby(groupby + ['instance_id']).sum()['TTR'] / 
                                    temp_df.groupby(groupby + ['instance_id']).count()['TTR']) > float(stat[4:])  # .reset_index(level='instance_id', drop='instance_id')
                stat_temp = (stat_temp_temp.groupby(groupby).sum() / stat_temp_temp.groupby(groupby).count()).rename(stat)
            elif stat == 'dose_change':
                stat_temp = (temp_df.groupby(groupby).sum()['dose_change'] /
                                temp_df.groupby(groupby).count()['dose_change']).rename(stat)
            elif stat == 'count':
                stat_temp = temp_df.groupby(groupby).count()['dose_change'].rename(stat)
            elif stat == 'INR':
                if 'instance_id' not in groupby:
                    groupby_temp = ['instance_id'] + groupby
                else:
                    groupby_temp = groupby
                stat_temp = (temp_df.groupby(groupby_temp).sum()['INR_current'] /
                                temp_df.groupby(groupby_temp).count()['INR_current']).rename(stat)
            elif stat == 'INR_percent_dose_change':
                if 'instance_id' not in groupby:
                    groupby_temp = ['instance_id', 'INR_current'] + groupby
                else:
                    groupby_temp = ['INR_current'] + groupby
                stat_temp = (temp_df.groupby(groupby_temp).sum()['delta_dose'] /
                                (temp_df.groupby(groupby_temp).sum()['action']
                                - temp_df.groupby(groupby_temp).sum()['delta_dose'])).rename(stat)

            results[stat] = stat_temp
        
        return results

    def stats(self, items='all', stats='all',
              groupby=['A/A', 'G/A', 'G/G',
                       '*1/*1', '*1/*2', '*1/*3',
                       '*2/*2', '*2/*3', '*3/*3'],
              filename=None, drop_days=[0]):
        if items == 'all':
            items = self._df_for_segmentation.keys()
        if stats == 'all':
            stats = ['TTR', 'TTR>0.65', 'dose_change', 'count', 'INR', 'INR_percent_dose_change']

        groupby = ['agent'] + groupby

        results = {}
        for agent in items:
            records_to_keep = [i+j for j in range(0, self._df_for_stats[agent].shape[0], 90) for i in range(90) if i not in drop_days]
            temp_df = self._df_for_stats[agent].loc[records_to_keep]
            for stat in stats:
                if stat == 'TTR':
                    stat_temp = (temp_df.groupby(groupby).sum()['TTR'] /
                                 temp_df.groupby(groupby).count()['TTR'])
                elif stat[:4] == 'TTR>':
                    stat_temp_temp = (temp_df.groupby(groupby + ['instance_id']).sum()['TTR'] / 
                                       temp_df.groupby(groupby + ['instance_id']).count()['TTR']) > float(stat[4:])  # .reset_index(level='instance_id', drop='instance_id')
                    stat_temp = (stat_temp_temp.groupby(groupby).sum() / stat_temp_temp.groupby(groupby).count()).rename(stat)
                elif stat == 'dose_change':
                    stat_temp = (temp_df.groupby(groupby).sum()['dose_change'] /
                                 temp_df.groupby(groupby).count()['dose_change']).rename(stat)
                elif stat == 'count':
                    stat_temp = temp_df.groupby(groupby).count()['dose_change'].rename(stat)
                elif stat == 'INR':
                    if 'instance_id' not in groupby:
                        groupby_temp = ['instance_id'] + groupby
                    else:
                        groupby_temp = groupby
                    stat_temp = (temp_df.groupby(groupby_temp).sum()['INR_current'] /
                                 temp_df.groupby(groupby_temp).count()['INR_current']).rename(stat)
                elif stat == 'INR_percent_dose_change':
                    if 'instance_id' not in groupby:
                        groupby_temp = ['instance_id', 'INR_current'] + groupby
                    else:
                        groupby_temp = ['INR_current'] + groupby
                    stat_temp = (temp_df.groupby(groupby_temp).sum()['delta_dose'] /
                                 (temp_df.groupby(groupby_temp).sum()['action']
                                 - temp_df.groupby(groupby_temp).sum()['delta_dose'])).rename(stat)

                results[(agent, stat)] = stat_temp

                if filename is not None:
                    with open('_'.join((self._results_path + filename, stat.replace('>',''), '.csv')), 'a+') as f:
                        stat_temp.to_csv(f, header=True)

        return results
