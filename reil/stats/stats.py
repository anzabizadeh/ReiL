from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from reil import rlbase


class Stats:
    def __init__(self,
                 all_stats: Tuple[str, ...] = ('',),
                 active_stats: Union[Sequence[str], str] = 'all',
                 groupby: Optional[List[str]] = None,
                 aggregators: Optional[List[str]] = None):
        '''
        Attributes:
        -----------
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
