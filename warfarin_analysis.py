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


# %%
class Analyzer:
    def __init__(self, experiment_path, **kwargs):
        self._experiment_path = experiment_path
        self._results_path = kwargs.get(
            'results_path', experiment_path + r'\results')
        self._aggregated_data_path = kwargs.get(
            'aggregated_data_path', experiment_path + r'\aggregated')
        self._dose_history = kwargs.get('dose_history', 10)
        self._INR_history = kwargs.get('INR_history', 10)
        # self._cluster_filename = kwargs.get('cluster_filename', None)
        # self._cluster_count = kwargs.get('cluster_count', 3)
        self._extended_state = kwargs.get('extended_state', False)
        if self._extended_state:
            self._column_names = ['age', 'weight', 'height', 'female', 'male', 'White', 'Black', 'Asian', 'American Indian', 'Pacific Islander',
            'tobaco_no', 'tobaco_yes', 'amiodarone_no', 'amiodarone_yes', 'fluvastatin_no', 'fluvastatin_yes',
            '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A']
        else:
            self._column_names = ['age', '*1/*1', '*1/*2', '*1/*3',
                '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A']
        self._X_embedded = {}
        self._files_loaded = []
        self._df_for_stats = {}
        self._df_for_segmentation = {}
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
                df_state_action.columns = self._column_names + [f'dose-{dose_history-i:02}') for i in range(dose_history)] + [f'INR-{INR_history-i:02}' for i in range(INR_history)] + ['INR_current', 'action']
                df_state_action['delta_dose']=df_state_action.apply(
                    lambda row: row['action'] - row['dose-01'] * dose_max, axis=1)
                df_state_action['dose_change'] = df_state_action.apply(
                    lambda row: int(row['action'] != row['dose-01']*dose_max), axis=1)
                df_state_action.INR_current = df_state_action.INR_current.apply(
                    lambda x: x * INR_max)
                df_state_action['TTR'] = df_state_action.INR_current.apply(
                    lambda x: 1 if 2 <= x <= 3 else 0)
                df_state_action['patient'] = subject
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

    def assign_cluster_label(self, cluster_filename, items = 'all', **kwargs):
        self._clustering_agent=WarfarinClusterAgent(
            cluster_filename = cluster_filename, **kwargs)
        if items == 'all':
            items=self._df_for_stats.keys()

        for agent in items:
            self._df_for_stats[agent]['cluster']=self._df_for_stats[agent].apply(
                lambda row: self._clustering_agent._assign_to_cluster(age=row['age'], CYP2C9=row['CYP2C9'], VKORC1=row['VKORC1']), axis=1)

        print('done')

    def TSNE(self, items='all'):
        if items == 'all':
            items = self._df_for_segmentation.keys()

        for agent in items:
            self._X_embedded[agent] = TSNE(n_components=2).fit_transform(
                self._df_for_segmentation[agent])

    def plot(self, items = 'all', plots = 'all', show = True):
        if self._X_embedded == {}:
            self.TSNE(items)

        if items == 'all':
            items=self._df_for_segmentation.keys()
        if plots == 'all':
            plots=['age', 'CYP2C9', 'VKORC1', 'sensitivity',
                     'mixture_model']

        fig, axs = plt.subplots(len(plots), len(
            items), figsize=(5*len(items), 5*len(plots)))
        if np.ndim(axs) == 1:
            axs = np.expand_dims(axs, axis=1)

        for index_agent, agent in enumerate(items):
            for index_plot, plot in enumerate(plots):
                if plot == 'age':
                    colors = self._df_for_segmentation[agent]['age']
                    color_map = 'Greens'
                elif plot == 'CYP2C9':
                    colors = list(zip(self._df_for_segmentation[agent]['G/G'],
                                      self._df_for_segmentation[agent]['G/A'],
                                      self._df_for_segmentation[agent]['A/A']))
                    color_map = 'jet'
                elif plot == 'VKORC1':
                    colors= self._df_for_segmentation[agent]['*1/*1'] * 2 +
                        self._df_for_segmentation[agent]['*1/*2'] * 4 +
                        self._df_for_segmentation[agent]['*1/*3'] * 6 +
                        self._df_for_segmentation[agent]['*2/*2'] * 8 +
                        self._df_for_segmentation[agent]['*2/*3'] * 10 +
                        self._df_for_segmentation[agent]['*3/*3'] * 12
                    color_map = 'tab10'
                elif plot == 'sensitivity':
                    colors= 1 * self._df_for_segmentation[agent]['G/G'] * (self._df_for_segmentation[agent]['*1/*1'] + self._df_for_segmentation[agent]['*1/*2']) +
                        1 * self._df_for_segmentation[agent]['G/A'] * self._df_for_segmentation[agent]['*1/*1'] +
                        2 * self._df_for_segmentation[agent]['G/G'] * (self._df_for_segmentation[agent]['*1/*3'] + self._df_for_segmentation[agent]['*2/*2'] + self._df_for_segmentation[agent]['*2/*3']) +
                        2 * self._df_for_segmentation[agent]['G/A'] * (self._df_for_segmentation[agent]['*1/*2'] + self._df_for_segmentation[agent]['*1/*3'] + self._df_for_segmentation[agent]['*2/*2']) +
                        2 * self._df_for_segmentation[agent]['A/A'] * (self._df_for_segmentation[agent]['*1/*1'] + self._df_for_segmentation[agent]['*1/*2']) +
                        3 * self._df_for_segmentation[agent]['G/G'] * self._df_for_segmentation[agent]['*3/*3'] +
                        3 * self._df_for_segmentation[agent]['G/A'] * (self._df_for_segmentation[agent]['*2/*3'] + self._df_for_segmentation[agent]['*3/*3']) +
                        3 * self._df_for_segmentation[agent]['A/A'] * (self._df_for_segmentation[agent]['*1/*3'] + self._df_for_segmentation[agent]
                                                                       ['*2/*2'] + self._df_for_segmentation[agent]['*2/*3'] + self._df_for_segmentation[agent]['*3/*3'])
                    color_map='coolwarm'
                elif plot == 'mixture_model':
                    colors=self._gmm_results.get(
                        agent, self.GMM(items=[agent]))
                    color_map='jet'
                    # axs[index_plot, index_agent].add_patch(patches.Ellipse())

                axs[index_plot, index_agent].scatter(self._X_embedded[agent][:, 0],
                                                     self._X_embedded[agent][:, 1],
                                                     c=colors, cmap=color_map)
                axs[index_plot, index_agent].set_title(agent + ': ' + plot)

        if show:
            plt.show()

        return fig

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
                    stat_temp_temp = (temp_df.groupby(groupby + ['patient']).sum()['TTR'] / 
                                       temp_df.groupby(groupby + ['patient']).count()['TTR']) > float(stat[4:])  # .reset_index(level='patient', drop='patient')
                    stat_temp = (stat_temp_temp.groupby(groupby).sum() / stat_temp_temp.groupby(groupby).count()).rename(stat)
                elif stat == 'dose_change':
                    stat_temp = (temp_df.groupby(groupby).sum()['dose_change'] /
                                 temp_df.groupby(groupby).count()['dose_change']).rename(stat)
                elif stat == 'count':
                    stat_temp = temp_df.groupby(groupby).count()['dose_change'].rename(stat)
                elif stat == 'INR':
                    if 'patient' not in groupby:
                        groupby_temp = ['patient'] + groupby
                    else:
                        groupby_temp = groupby
                    stat_temp = (temp_df.groupby(groupby_temp).sum()['INR_current'] /
                                 temp_df.groupby(groupby_temp).count()['INR_current']).rename(stat)
                elif stat == 'INR_percent_dose_change':
                    if 'patient' not in groupby:
                        groupby_temp = ['patient', 'INR_current'] + groupby
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

    def load(self, filename=None, path='.'):
        '''
        Load an object from a file.

        Arguments
        ---------
            filename: the name of the file to be loaded.
            path: the path of the file to be loaded. (Default='.')

        Raises ValueError if the filename is not specified.
        '''
        if filename is None:
            raise ValueError('name of the input file not specified.')

        with open(path + '/' + filename + '.pkl', 'rb') as f:
            try:
                data = load(f)
            except EOFError:
                raise RuntimeError('Corrupted data file: '+filename)
            for key, value in data.items():
                self.__dict__[key] = value

    def save(self, filename=None, path='.'):
        '''
        Save the object to a file.

        Arguments
        ---------
            filename: the name of the object (Default=self._name)
            path: the path of the file to be loaded. (Default='.')
        '''
        if filename is None:
            raise ValueError('name of the output file not specified.')

        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open(path + '/' + filename + '.pkl', 'wb+') as f:
            dump(self.__dict__, f, HIGHEST_PROTOCOL)

        return path, filename


# %%
# experiment_path = r'./ravvaz_outputs'
# filename = 'collected_data_4_6_16_with_4_clusters'
filename = 'collected_data'
experiment_path = r'./outputs'
results_path = experiment_path + r'/results_INR_dose_change/'
dose_history = 10
INR_history = 10

analysis = Analyzer(experiment_path=experiment_path,
                    results_path=results_path,
                    dose_history=dose_history,
                    INR_history=INR_history)


try:
    analysis.load(filename=filename)
    analysis._results_path = experiment_path + r'/results_INR_dose_change/'
except:
    analysis.load_experiment_data()
    analysis.assign_cluster_label('Weka output (4 clusters).csv')
    analysis.save(filename='collected_data_4_6_16_with_4_clusters')

analysis.stats(stats=['INR', 'delta_dose', 'INR_percent_dose_change'],
               groupby=['sensitivity', 'cluster'],
               items=['WCA04_00_0.0'],
               filename='4_6_16_with_4_clusters',
               drop_days=range(7))
