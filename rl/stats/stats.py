import numpy as np
import pandas as pd

class Stats:
    # def __init__(self, name_stat_dict, all_stats=[], **kwargs):
    #     '''
    #     Attributes:
    #     -----------
    #         name_stat_dict: a dictionary that assigns a name to groupby, stats, etc. in the form of a nested dictionary.
    #         Example: name_stat_dict={'stat01': {'stats': 'all', 'groupby': ['some column']}}
    #     '''
    #     self._all_stats = all_stats
    #     self._name_stat_dict = name_stat_dict
    #     for name in self._name_stat_dict.keys():
    #         if 'stats' not in self._name_stat_dict[name].keys():
    #             self._name_stat_dict[name]['stats'] = []
    #         if 'groupby' not in self._name_stat_dict[name].keys():
    #             self._name_stat_dict[name]['groupby'] = []
    #         if self._name_stat_dict[name]['stats'] == 'all':
    #             self._name_stat_dict[name]['stats'] = self._all_stats

    def __init__(self, active_stats='all', groupby=[], aggregators=[], all_stats=[], **kwargs):
        '''
        Attributes:
        -----------
            active_stats: a list of stats that should be active for calculation.
            groupby: fields by which in input to stats should be grouped.
        '''
        self._all_stats = all_stats
        self._active_stats = active_stats
        self._groupby = groupby
        self._aggregators = aggregators
        if active_stats == 'all':
            self._active_stats = self._all_stats
        else:
            self._active_stats = active_stats

    def from_history(self, name, history):
        raise NotImplementedError

    def aggregate(self, agent_stats=None, subject_stats=None):
        raise NotImplementedError
