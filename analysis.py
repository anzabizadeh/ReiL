#%%
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
from mpl_toolkits.mplot3d import Axes3D
from rl.environments import Environment
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


#%%
class Analyzer:
    def __init__(self, **kwargs):
        self._experiment_path = kwargs.get('experiment_path', r'.\W')
        self._result_path = kwargs.get('result_path', experiment_path + r'\results')
        self._aggregated_data_path = kwargs.get('aggregated_data_path', experiment_path + r'\aggregated')
        self._dose_history = kwargs.get('dose_history', 9)
        self._INR_history = kwargs.get('INR_history', 9)
        self._X_embedded = {}
        self._files_loaded = []
        self._df_for_stats = {}
        self._df_for_segmentation = {}

    def load_experiment_data(self, **kwargs):
        result_path = kwargs.get('result_path', self._result_path)
        self._dose_history = kwargs.get('dose_history', 9)
        self._INR_history = kwargs.get('INR_history', 9)
        if kwargs.get('reload', False):
            self._files_loaded = []
            self._df_for_stats = {}  # to calculate TTR and other possible measures
            self._df_for_segmentation = {}  # a point is defined as a 90-day dose

        for f in listdir(result_path):
            full_filename = join(result_path, f)
            if isfile(full_filename) and full_filename not in self._files_loaded:
                agent, subject = f[:-4].split(sep='@', )
                print(agent, subject)
                # agent, subject = f[:-4].split(sep='_')
                df_from_file = pd.read_pickle(full_filename)[['state', 'action']]

                patient_info = df_from_file.iloc[0].state.normalize().as_list()
                if len(patient_info) == 1:
                    patient_info = list(*df_from_file.iloc[0].state.normalize().value)
                try:
                    self._df_for_segmentation[agent].loc[len(self._df_for_segmentation[agent])] = \
                        patient_info[:10] + [float(a.value) for a in df_from_file.action]
                    # df_for_segmentation[agent].append(np.concatenate([df_from_file.iloc[0].state.normalize().as_nparray()[:, :10],
                    #         np.array(tuple(float(a.value) for a in df_from_file.action), ndmin=2)], axis=1), ignore_index=True)
                except KeyError:
                    self._df_for_segmentation[agent] = pd.DataFrame(dict(zip(['age', '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A']
                                        + ['dose {:02}'.format(i) for i in range(90)],
                                        patient_info[:10] + [float(a.value) for a in df_from_file.action])),
                                        index=[0])
                    # df_for_segmentation[agent] = pd.DataFrame(df_from_file.iloc[0].state.normalize().as_list()[:10]
                    #     + [float(a.value) for a in df_from_file.action])
                    # df_for_segmentation[agent] = pd.DataFrame(np.concatenate([df_from_file.iloc[0].state.normalize().as_nparray()[:, :10],
                    #     np.array(tuple(float(a.value) for a in df_from_file.action), ndmin=2)], axis=1))
                    # df_for_segmentation[agent].columns = ['age', '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A'] + ['dose {:02}'.format(i) for i in range(90)]

                # df_state_action = pd.DataFrame(np.concatenate((df_from_file.iloc[0].state.normalize().as_nparray(),
                #         np.array(tuple(float(a.value) for a in df_from_file.action), ndmin=2)), axis=1))
                # df_state_action.columns = ['age', '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A'] + ['dose {:02}'.format(i) for i in range(90)]
                INR_factor = df_from_file.loc[0, 'state'].upper.loc['INRs']
                df_state_action = df_from_file.apply(lambda row: [z for y in [x if hasattr(x, '__iter__') and not isinstance(x, str) else [x] for x in row.state.normalize().value] for z in y] + row.action.as_list(), axis=1, result_type='expand')
                df_state_action.columns = ['age', '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A'] + ['dose-{:02}'.format(dose_history-i) for i in range(dose_history)] + ['INR-{:02}'.format(INR_history-i) for i in range(INR_history)] + ['INR_current', 'action']
                # df_state_action.drop(['dose-{:02}'.format(dose_history-i) for i in range(dose_history)] + ['INR-{:02}'.format(INR_history-i) for i in range(INR_history)] + ['INR_current'], inplace=True, axis=1)
                df_state_action['dose change'] = df_state_action.apply(lambda row: int(row['action']/15.0 != row['dose-01']), axis=1)
                df_state_action.INR_current = df_state_action.INR_current.apply(lambda x: x * INR_factor)
                df_state_action['TTR'] = df_state_action.INR_current.apply(lambda x: 1 if 2<=x<=3 else 0)
                df_state_action['patient'] = subject

                df_state_action.drop(['dose-{:02}'.format(dose_history-i) for i in range(dose_history)] + ['INR-{:02}'.format(INR_history-i) for i in range(INR_history)], inplace=True, axis=1)
                try:
                    self._df_for_stats[agent] = pd.concat([self._df_for_stats[agent], df_state_action], ignore_index=True)
                except KeyError:
                    self._df_for_stats[agent] = df_state_action

                self._files_loaded.append(full_filename)

    def TSNE(self, items='all'):
        if items == 'all':
            items = self._df_for_segmentation.keys()

        self._X_embedded = {}
        for agent in items:
            self._X_embedded[agent] = TSNE(n_components=2).fit_transform(self._df_for_segmentation[agent])

    def plot(self, items='all', plots='all', show=True):
        if self._X_embedded == {}:
            self.TSNE(items)

        if items == 'all':
            items = self._df_for_segmentation.keys()
        if plots == 'all':
            plots = ['age', 'CYP2C9', 'VKORC1', 'sensitivity']

        fig, axs = plt.subplots(len(plots), len(items), figsize=(5*len(items), 5*len(plots)))
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
                    colors = self._df_for_segmentation[agent]['*1/*1'] * 2 + \
                        self._df_for_segmentation[agent]['*1/*2'] * 4 + \
                        self._df_for_segmentation[agent]['*1/*3'] * 6 + \
                        self._df_for_segmentation[agent]['*2/*2'] * 8 + \
                        self._df_for_segmentation[agent]['*2/*3'] * 10 + \
                        self._df_for_segmentation[agent]['*3/*3'] * 12
                    color_map = 'tab10'
                elif plot == 'sensitivity':
                    colors = 1 * self._df_for_segmentation[agent]['G/G'] * (self._df_for_segmentation[agent]['*1/*1'] + self._df_for_segmentation[agent]['*1/*2']) + \
                            1 * self._df_for_segmentation[agent]['G/A'] * self._df_for_segmentation[agent]['*1/*1'] + \
                            2 * self._df_for_segmentation[agent]['G/G'] * (self._df_for_segmentation[agent]['*1/*3'] + self._df_for_segmentation[agent]['*2/*2'] + self._df_for_segmentation[agent]['*2/*3']) + \
                            2 * self._df_for_segmentation[agent]['G/A'] * (self._df_for_segmentation[agent]['*1/*2'] + self._df_for_segmentation[agent]['*1/*3'] + self._df_for_segmentation[agent]['*2/*2']) + \
                            2 * self._df_for_segmentation[agent]['A/A'] * (self._df_for_segmentation[agent]['*1/*1'] + self._df_for_segmentation[agent]['*1/*2']) + \
                            3 * self._df_for_segmentation[agent]['G/G'] * self._df_for_segmentation[agent]['*3/*3'] + \
                            3 * self._df_for_segmentation[agent]['G/A'] * (self._df_for_segmentation[agent]['*2/*3'] + self._df_for_segmentation[agent]['*3/*3']) + \
                            3 * self._df_for_segmentation[agent]['A/A'] * (self._df_for_segmentation[agent]['*1/*3'] + self._df_for_segmentation[agent]['*2/*2'] + self._df_for_segmentation[agent]['*2/*3'] + self._df_for_segmentation[agent]['*3/*3'])
                    color_map = 'coolwarm'

                axs[index_plot, index_agent].scatter(self._X_embedded[agent][:, 0],
                                                     self._X_embedded[agent][:, 1],
                                                     c=colors, cmap=color_map)
                axs[index_plot, index_agent].set_title(agent + ': ' + plot)      

        if show:
            plt.show()
        
        return fig

    def stats(self, items='all', stats='all', groupby=['A/A', 'G/A', 'G/G', '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3']):
        if items == 'all':
            items = self._df_for_segmentation.keys()
        if stats == 'all':
            stats = ['TTR', 'dose change']

        results = {}
        for agent in items:
            for stat in stats:
                if stat == 'TTR':
                    stat_temp = (self._df_for_stats[agent].groupby(groupby).sum()/ \
                                 self._df_for_stats[agent].groupby(groupby).count())['TTR']
                elif stat == 'dose change':
                    stat_temp = (self._df_for_stats[agent].groupby(groupby).sum()/ \
                                 self._df_for_stats[agent].groupby(groupby).count())['dose change']

                results[(agent, stat)] = stat_temp

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
            raise ValueError('name of the output file not specified.')

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

#%%
experiment_path = r'.\Warfarin_v4_default_state'
result_path = experiment_path + r'\results'
dose_history = 9
INR_history = 9

analysis = Analyzer(experiment_path=experiment_path,
                    result_path=result_path,
                    dose_history=dose_history,
                    INR_history=INR_history)

#%%
analysis.load_experiment_data()
analysis.save(filename='collected_data')

#%%
analysis = Analyzer(experiment_path=experiment_path,
                    result_path=result_path,
                    dose_history=dose_history,
                    INR_history=INR_history)

analysis.load(filename='collected_data')
#%%
analysis.plot()

#%%
analysis.stats()

# #%%
# experiment_path = r'.\W'
# result_path = experiment_path + r'\results'
# dose_history = 9
# INR_history = 9

# #%%
# df_for_stats = {}  # to calculate TTR and other possible measures
# df_for_segmentation = {}  # a point is defined as a 90-day dose
# for f in listdir(result_path):
#     full_filename = join(result_path, f)
#     if isfile(full_filename):
#         agent, subject = f[:-4].split(sep='@', )
#         print(agent, subject)
#         # agent, subject = f[:-4].split(sep='_')
#         df_from_file = pd.read_pickle(full_filename)[['state', 'action']]

#         patient_info = df_from_file.iloc[0].state.normalize().as_list()
#         if len(patient_info) == 1:
#             patient_info = list(*df_from_file.iloc[0].state.normalize().value)
#         try:
#             df_for_segmentation[agent].loc[len(df_for_segmentation[agent])] = \
#                 patient_info[:10] + [float(a.value) for a in df_from_file.action]
#             # df_for_segmentation[agent].append(np.concatenate([df_from_file.iloc[0].state.normalize().as_nparray()[:, :10],
#             #         np.array(tuple(float(a.value) for a in df_from_file.action), ndmin=2)], axis=1), ignore_index=True)
#         except KeyError:
#             df_for_segmentation[agent] = pd.DataFrame(dict(zip(['age', '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A']
#                                 + ['dose {:02}'.format(i) for i in range(90)],
#                                 patient_info[:10] + [float(a.value) for a in df_from_file.action])),
#                                 index=[0])
#             # df_for_segmentation[agent] = pd.DataFrame(df_from_file.iloc[0].state.normalize().as_list()[:10]
#             #     + [float(a.value) for a in df_from_file.action])
#             # df_for_segmentation[agent] = pd.DataFrame(np.concatenate([df_from_file.iloc[0].state.normalize().as_nparray()[:, :10],
#             #     np.array(tuple(float(a.value) for a in df_from_file.action), ndmin=2)], axis=1))
#             # df_for_segmentation[agent].columns = ['age', '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A'] + ['dose {:02}'.format(i) for i in range(90)]

#         # df_state_action = pd.DataFrame(np.concatenate((df_from_file.iloc[0].state.normalize().as_nparray(),
#         #         np.array(tuple(float(a.value) for a in df_from_file.action), ndmin=2)), axis=1))
#         # df_state_action.columns = ['age', '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A'] + ['dose {:02}'.format(i) for i in range(90)]
#         INR_factor = df_from_file.loc[0, 'state'].upper.loc['INRs']
#         df_state_action = df_from_file.apply(lambda row: [z for y in [x if hasattr(x, '__iter__') and not isinstance(x, str) else [x] for x in row.state.normalize().value] for z in y] + row.action.as_list(), axis=1, result_type='expand')
#         df_state_action.columns = ['age', '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A'] + ['dose-{:02}'.format(dose_history-i) for i in range(dose_history)] + ['INR-{:02}'.format(INR_history-i) for i in range(INR_history)] + ['INR_current', 'action']
#         # df_state_action.drop(['dose-{:02}'.format(dose_history-i) for i in range(dose_history)] + ['INR-{:02}'.format(INR_history-i) for i in range(INR_history)] + ['INR_current'], inplace=True, axis=1)
#         df_state_action.drop(['dose-{:02}'.format(dose_history-i) for i in range(dose_history)] + ['INR-{:02}'.format(INR_history-i) for i in range(INR_history)], inplace=True, axis=1)
#         df_state_action.INR_current = df_state_action.INR_current.apply(lambda x: x * INR_factor)
#         df_state_action['TTR'] = df_state_action.INR_current.apply(lambda x: 1 if 2<=x<=3 else 0)
#         try:
#             df_for_stats[agent] = pd.concat([df_for_stats[agent], df_state_action], ignore_index=True)
#         except KeyError:
#             df_for_stats[agent] = df_state_action


# #%%
# # Plot doses
# for i in df.index:
#     plt.plot(df.loc[i, 'dose 0': 'dose 89'])

# plt.show()


# #%%
# # stats
# df_for_stats['0.60_2_day'].groupby(['A/A', 'G/A', 'G/G', '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3']).count()/90

# #%%
# # Visualization of protocols
# k_means = {}
# X_embedded = {}
# plot_index = 0
# fig = plt.figure()

# for agent, df in df_for_segmentation.items():
#     plot_index += 1
#     try:
#         X_embedded[agent] = TSNE(n_components=2).fit_transform(df)
#         plt.subplot(len(df_for_segmentation), 1, plot_index) # plt.figure(figsize=(10, 10))
#         # ax = fig.axis(projection='2d')  # Axes3D(fig)
#         # colors = dict((i, (random(), random(), random(), 0.7)) for i in pd.unique(Y))
#         plt.scatter(X_embedded[agent][:, 0],
#                     X_embedded[agent][:, 1])
#                     # X_embedded[:, 2],
#                     # c=df_for_segmentation['0.50_4_day'].age,
#                     # label=y_kmeans)
#     except ValueError:
#         pass

# plt.show()

#     # kmeans = KMeans(n_clusters=5)
#     # kmeans.fit(X)
#     # y_kmeans = kmeans.predict(X)


# #%%
# # Different plots:

# # age
# for plot_index, agent in enumerate(df_for_segmentation):
#     plt.subplot(len(df_for_segmentation), 1, plot_index + 1) # plt.figure(figsize=(10, 10))
#     plt.scatter(X_embedded[agent][:, 0], X_embedded[agent][:, 1], c=df_for_segmentation[agent]['age'], cmap='Greens')
#     # fig.suptitle('G/G->Red \n G/A -> Green \n A/A -> Blue')
# plt.show()

# # G/G, G/A, A/A
# for plot_index, agent in enumerate(df_for_segmentation):
#     plt.subplot(len(df_for_segmentation), 1, plot_index + 1) # plt.figure(figsize=(10, 10))
#     plt.scatter(X_embedded[agent][:, 0], X_embedded[agent][:, 1], c=list(zip(df_for_segmentation[agent]['G/G'], df_for_segmentation[agent]['G/A'], df_for_segmentation[agent]['A/A'])))
#     # fig.suptitle('G/G->Red \n G/A -> Green \n A/A -> Blue')
# plt.show()


# # *1/*1, *1/*2, *1/*3, *2/*3, *3/*3
# for plot_index, agent in enumerate(df_for_segmentation):
#     plt.subplot(len(df_for_segmentation), 1, plot_index + 1) # plt.figure(figsize=(10, 10))
#     colors = df_for_segmentation[agent]['*1/*1'] * 2 + \
#              df_for_segmentation[agent]['*1/*2'] * 4 + \
#              df_for_segmentation[agent]['*1/*3'] * 6 + \
#              df_for_segmentation[agent]['*2/*2'] * 8 + \
#              df_for_segmentation[agent]['*2/*3'] * 10 + \
#              df_for_segmentation[agent]['*3/*3'] * 12

#     plt.scatter(X_embedded[agent][:, 0], X_embedded[agent][:, 1], c=colors, cmap='tab10')
# plt.show()


# # Normal, Sensitive, Highly Sensitive
# for plot_index, agent in enumerate(df_for_segmentation):
#     plt.subplot(len(df_for_segmentation), 1, plot_index + 1) # plt.figure(figsize=(10, 10))
#     colors = 1 * df_for_segmentation[agent]['G/G'] * (df_for_segmentation[agent]['*1/*1'] + df_for_segmentation[agent]['*1/*2']) + \
#              1 * df_for_segmentation[agent]['G/A'] * df_for_segmentation[agent]['*1/*1'] + \
#              2 * df_for_segmentation[agent]['G/G'] * (df_for_segmentation[agent]['*1/*3'] + df_for_segmentation[agent]['*2/*2'] + df_for_segmentation[agent]['*2/*3']) + \
#              2 * df_for_segmentation[agent]['G/A'] * (df_for_segmentation[agent]['*1/*2'] + df_for_segmentation[agent]['*1/*3'] + df_for_segmentation[agent]['*2/*2']) + \
#              2 * df_for_segmentation[agent]['A/A'] * (df_for_segmentation[agent]['*1/*1'] + df_for_segmentation[agent]['*1/*2']) + \
#              3 * df_for_segmentation[agent]['G/G'] * df_for_segmentation[agent]['*3/*3'] + \
#              3 * df_for_segmentation[agent]['G/A'] * (df_for_segmentation[agent]['*2/*3'] + df_for_segmentation[agent]['*3/*3']) + \
#              3 * df_for_segmentation[agent]['A/A'] * (df_for_segmentation[agent]['*1/*3'] + df_for_segmentation[agent]['*2/*2'] + df_for_segmentation[agent]['*2/*3'] + df_for_segmentation[agent]['*3/*3'])

#     plt.scatter(X_embedded[agent][:, 0], X_embedded[agent][:, 1], c=colors, cmap='coolwarm')
# plt.show()

# #%%
# print('Extracting Xs and Ys.')

# X = {}
# Y = {}
# for agent, df in temp.items():
#     X[agent] = df[:10000].apply(lambda row: [z for y in [x if hasattr(x, '__iter__') and not isinstance(x, str) else [x] for x in row.state.normalize().value] for z in y ] + row.action.as_list(), axis=1, result_type='expand')
#     Y[agent] = df[:10000].apply(lambda row: row.action.as_list()[0], axis=1)
#     X[agent].columns = ['age', '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A'] + ['dose-{:02}'.format(dose_history-i) for i in range(dose_history)] + ['INR-{:02}'.format(INR_history-i) for i in range(INR_history)] + ['INR_current', 'action']
#     Y[agent].columns = ['action']


# ##%%
# #temp_df = pd.concat([pd.read_pickle(join(result_path, f))[['state', 'action']] for f in listdir(result_path) if isfile(join(result_path, f))], ignore_index=True)

# # #%%
# # print('Extracting Xs and Ys.')
# # X = temp_df[:10000].apply(lambda row: [z for y in [x if hasattr(x, '__iter__') and not isinstance(x, str) else [x] for x in row.state.normalize().value] for z in y ] + row.action.as_list(), axis=1, result_type='expand')
# # Y = temp_df[:10000].apply(lambda row: row.action.as_list()[0], axis=1)

# # X.columns = ['age', '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A'] + ['dose-{:02}'.format(dose_history-i) for i in range(dose_history)] + ['INR-{:02}'.format(INR_history-i) for i in range(INR_history)] + ['INR_current', 'action']
# # Y.columns = ['action']


# kmeans = KMeans(n_clusters=5)
# kmeans.fit(X)
# y_kmeans = kmeans.predict(X)


# #%%
# X_embedded = TSNE(n_components=2).fit_transform(X)


# #%%
# plt.savefig('t-sne of kNN 5 of states - 10000.png')


# #%%
# get_ipython().run_line_magic('matplotlib', 'notebook')

# fig = plt.figure(figsize=(10, 10))
# ax = fig.gca(projection='3d')  # Axes3D(fig)
# colors = dict((i, (random(), random(), random(), 0.7)) for i in pd.unique(Y))
# scatter = ax.scatter(X_embedded[:, 0],
#                      X_embedded[:, 1],
#                      X_embedded[:, 2],
#                      c=y_kmeans,
#                      label=y_kmeans)
# ax.legend()
# plt.show()
# #plt.savefig('t-sne of states - 10000.png')
















# #%%
# generation_cycles = 1
# patient_count = 10
# dose_history = 9
# INR_history = 9
# file_names = ['./sampled_policies_0.20.pkl',
#               './sampled_policies_0.40.pkl',
#               './sampled_policies_0.80.pkl']
# RL_filenames = ['WARF_00_00_XX_d_90_dose_9_INR_9_T__dose_change_coef_0.20_DQN_(20,20)_g_0.8_e_func_lr_0.02_buff_900_clr_F_btch_50_vld_0.3',
#                 'WARF_00_00_XX_d_90_dose_9_INR_9_T__dose_change_coef_0.40_DQN_(20,20)_g_0.8_e_func_lr_0.02_buff_900_clr_F_btch_50_vld_0.3',
#                 'WARF_00_00_XX_d_90_dose_9_INR_9_T__dose_change_coef_0.80_DQN_(20,20)_g_0.8_e_func_lr_0.02_buff_900_clr_F_btch_50_vld_0.3']


# #%%
# tf.reset_default_graph()

# #%%
# env = []
# agent = []
# subject = []

# for f in RL_filenames:
#     print(f)
#     env.append(Environment(filename=f))
#     agent.append(env[-1]._agent['protocol'])
#     subject.append(env[-1]._subject['W'])

# #%% [markdown]
# # ## Create state-action Dataset 

# #%%
# temp_df = []
# for f in range(len(RL_filenames)):
#     print('Reading previous records.')
#     try:
#         temp_df.append(pd.read_pickle(file_name[f]))
#     except FileNotFoundError:
#         print('No previous records found.')

#     for i in range(generation_cycles):
#         print('\n\ncycle {:03}\n'.format(i))
#         print('Generating new records.')
#         new_records = pd.concat([env[f].trajectory()['protocol'][['state', 'action']] for i in range(patient_count)], ignore_index=True)

#         try:
#             temp_df = pd.concat([temp_df, new_records], ignore_index=True)
#         except FileNotFoundError:
#             temp_df = new_records

#         print('Saving.')
#         temp_df.to_pickle(file_name[f])

# #%% [markdown]
# # ## Read the dataset and extract

# #%%
# temp_df = pd.read_pickle(file_name)


# #%%
# print('Extracting Xs and Ys.')
# X = temp_df[:10000].apply(lambda row: [z for y in [x if hasattr(x, '__iter__') and not isinstance(x, str) else [x] for x in row.state.normalize().value] for z in y ] + row.action.as_list(), axis=1, result_type='expand')
# Y = temp_df[:10000].apply(lambda row: row.action.as_list()[0], axis=1)

# X.columns = ['age', '*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'G/G', 'G/A', 'A/A'] + ['dose-{:02}'.format(dose_history-i) for i in range(dose_history)] + ['INR-{:02}'.format(INR_history-i) for i in range(INR_history)] + ['INR_current', 'action']
# Y.columns = ['action']

# #%% [markdown]
# # ## Clustering and its plot


# kmeans = KMeans(n_clusters=5)
# kmeans.fit(X)
# y_kmeans = kmeans.predict(X)


# #%%
# X_embedded = TSNE(n_components=3).fit_transform(X)


# #%%
# plt.savefig('t-sne of kNN 5 of states - 10000.png')


# #%%
# get_ipython().run_line_magic('matplotlib', 'notebook')

# fig = plt.figure(figsize=(10, 10))
# ax = fig.gca(projection='3d')  # Axes3D(fig)
# colors = dict((i, (random(), random(), random(), 0.7)) for i in pd.unique(Y))
# scatter = ax.scatter(X_embedded[:, 0],
#                      X_embedded[:, 1],
#                      X_embedded[:, 2],
#                      c=y_kmeans,
#                      label=y_kmeans)
# ax.legend()
# plt.show()
# #plt.savefig('t-sne of states - 10000.png')

# #%% [markdown]
# # ## Plots per dose level

# #%%
# get_ipython().run_line_magic('matplotlib', 'notebook')

# fig = plt.figure(figsize=(16, 16))
# ax = fig.gca(projection='3d')  # Axes3D(fig)
# colors = dict((i, (random(), random(), random(), 0.7)) for i in pd.unique(Y))
# for label in sorted(pd.unique(Y)):
#     scatter = ax.scatter(X_embedded[Y[Y==label].index, 0],
#                          X_embedded[Y[Y==label].index, 1],
#                          X_embedded[Y[Y==label].index, 2],
#                          c=[colors[label]],
#                          label=label)
# ax.legend()
# plt.show()
# plt.savefig('t-sne of states - 10000.png')
#     #plt.savefig('t-sne 3d of states - 10000 - {:02.1f}.png'.format(label))
#     #plt.cla()


# #%%
# plt.savefig('t-sne of states.png')

# #%% [markdown]
# # ## TTR

# #%%
# TTRs = {}
# counts = {}
# temp = temp_df
# for i in range(0, temp_df.shape[0], 90):
#     print('{:02}\tpatient: Age: {}, CYP2C9: {}, VKORC1: {}'.format(
#       i,
#       temp.state[i].value['Age'],
#       temp.state[i].value['CYP2C9'],
#       temp.state[i].value['VKORC1'])
#      )

#     try:
#         TTR = sum(1 if 2.0<=temp.state[i+j].value.INRs[-1]<=3.0 else 0 for j in range(90)) / 90
#         TTRs[(temp.state[i].value['Age'],
#                  temp.state[i].value['CYP2C9'],
#                  temp.state[i].value['VKORC1'])].append(TTR)

#     except KeyError:
#         TTRs[(temp.state[i].value['Age'],
#                  temp.state[i].value['CYP2C9'],
#                  temp.state[i].value['VKORC1'])] = [TTR]

# print('TTR\npatient\t\t\tcount\tmin\taverage\tmax')
# for patient, r in TTRs.items():
#     print('{}\t{:3}\t{:2.2%}\t{:2.2%}\t{:2.2%}'.format(patient, len(r), min(r), sum(r)/len(r), max(r)))


# #%%
# TTRs = {}
# counts = {}
# temp = temp_df
# for i in range(temp_df.shape[0]):
# #    print('{:02}\tpatient: Age: {}, CYP2C9: {}, VKORC1: {}'.format(
#  #     i,
#   #    temp.state[i].value['Age'],
#    #   temp.state[i].value['CYP2C9'],
#     #  temp.state[i].value['VKORC1'])
#      #)
#     try:
#         TTRs[(temp.state[i].value['Age'],
#                  temp.state[i].value['CYP2C9'],
#                  temp.state[i].value['VKORC1'])] += 1 if 2.0 <= temp_df.state[i].value.INRs[-1] <= 3.0 else 0
#         counts[(temp.state[i].value['Age'],
#                  temp.state[i].value['CYP2C9'],
#                  temp.state[i].value['VKORC1'])] += 1
#     except KeyError:
#         TTRs[(temp.state[i].value['Age'],
#                  temp.state[i].value['CYP2C9'],
#                  temp.state[i].value['VKORC1'])] = 1 if 2.0 <= temp_df.state[i].value.INRs[-1] <= 3.0 else 0
#         counts[(temp.state[i].value['Age'],
#                  temp.state[i].value['CYP2C9'],
#                  temp.state[i].value['VKORC1'])] = 1

# print('TTR\npatient\t\t\tcount\tTTR')
# for patient, r in TTRs.items():
#     print('{}\t{:3}\t{:3.2%}'.format(patient, counts[patient], r/counts[patient]))  # len(r), min(r), sum(r)/len(r), max(r)))


# #%%
# print('TTR\npatient\t\t\tcount\tTTR')
# for patient, r in TTRs.items():
#     print('{}\t{:3}\t{:3.2%}'.format(patient, counts[patient]/90, r/counts[patient]))

# #%% [markdown]
# # ## One Patient

# #%%
# temp = env.trajectory()['protocol']
# print('patient: Age: {}, CYP2C9: {}, VKORC1: {}'.format(
#       temp.state[0].value['Age'],
#       temp.state[0].value['CYP2C9'],
#       temp.state[0].value['VKORC1'])
#      )
# print('day\tINR\taction\treward')
# TTR = 0
# for t in temp.iterrows():
#     print('{}\t{:3.1f}\t{:5.1f}\t{:3.2f}'.format(t[0],
#                                               t[1].state.value['INRs'][-1],
#                                               t[1].action.value['value'],
#                                               t[1].reward))
#     if 2 <= t[1].state.value['INRs'][-1] <= 3:
#         TTR += 1
# print(TTR/90)


# #%%
# TTRs = {}
# actions = {}
# for i in range(50):
    
#     temp = env.trajectory()['protocol']
#     print('{:02}\tpatient: Age: {}, CYP2C9: {}, VKORC1: {}'.format(
#       i,
#       temp.state[0].value['Age'],
#       temp.state[0].value['CYP2C9'],
#       temp.state[0].value['VKORC1'])
#      )
#     try:
#         TTR = sum(1 if 2.0<=t[1].state.value['INRs'][-1]<=3.0 else 0 for t in temp.iterrows()) / len(temp)
#         ave_action = sum(t[1].action.value['value'] for t in temp.iterrows()) / len(temp)
#         TTRs[(temp.state[0].value['Age'],
#                  temp.state[0].value['CYP2C9'],
#                  temp.state[0].value['VKORC1'])].append(TTR)
#         actions[(temp.state[0].value['Age'],
#                 temp.state[0].value['CYP2C9'],
#                 temp.state[0].value['VKORC1'])].append(ave_action)
#     except KeyError:
#         TTRs[(temp.state[0].value['Age'],
#                  temp.state[0].value['CYP2C9'],
#                  temp.state[0].value['VKORC1'])] = [TTR]
#         actions[(temp.state[0].value['Age'],
#                  temp.state[0].value['CYP2C9'],
#                  temp.state[0].value['VKORC1'])] = [ave_action]

# print('TTR\npatient\t\t\tcount\tmin\taverage\tmax')
# for patient, r in TTRs.items():
#     print('{}\t{:3}\t{:2.2%}\t{:2.2%}\t{:2.2%}'.format(patient, len(r), min(r), sum(r)/len(r), max(r)))

# print('actions\npatient\t\t\tcount\tmin\taverage\tmax')
# for patient, r in actions.items():
#     print('{}\t{:3}\t{:3.2}\t{:3.2}\t{:3.2}'.format(patient, len(r), min(r), sum(r)/len(r), max(r)))



# j_results = {}
# for p, v in results.items():
#     j_results[', '.join((p[2], p[1], str(p[0])))] = v

# with open("10000patients.json", "w") as write_file:
#     json.dump(j_results, write_file)
    


# #%%
# all_records = []
# for p, value in results.items():
#     all_records.extend(list((p, v) for v in value))
    
# for r in all_records:
#     print(r[0], r[1])


# csv.writer()


# #%%
# for patient in sorted(results, key=lambda s: s[2]+s[1]+str(s[0])):
#     print('\t'.join((patient[2], patient[1], str(patient[0]),
#                      '{: 3}'.format(len(results[patient])),
#                      '{:3.2}'.format(min(results[patient])),
#                      '{:3.2}'.format(sum(results[patient])/len(results[patient])),
#                      '{:3.2}'.format(max(results[patient])))))


#%%
