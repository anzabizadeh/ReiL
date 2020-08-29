from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd
from reil import rlbase


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

    def __init__(self,
                 all_stats: Sequence = (),
                 active_stats: Union[Sequence[str], str] = 'all',
                 groupby: Optional[Sequence] = None,
                 aggregators: Optional[Sequence] = None):
        '''
        Attributes:
        -----------
            active_stats: a list of stats that should be active for calculation.
                You can also set it to 'all' to indicate that you want all the stats in `all_stats`. 
            groupby: fields by which in input to stats should be grouped.
            aggregator: an aggregator function for groupby.
            all_stats: list of all stats.
        '''
        self._all_stats = all_stats
        self._groupby = groupby
        self._aggregators = aggregators
        if active_stats == 'all':
            self._active_stats = self._all_stats
        else:
            self._active_stats = active_stats

    def from_history(self, history: rlbase.History) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    def aggregate(self,
                  agent_stats: Optional[Dict[str, Any]] = None,
                  subject_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError
