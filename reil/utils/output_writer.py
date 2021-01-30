from typing import Any, Dict, List, Optional, Tuple, Union
import pathlib
import pandas as pd
from reil.environments.environment import AgentSubjectTuple


class OutputWriter:
    def __init__(self,
                 filename: str,
                 path: Union[str, pathlib.Path] = '.',
                 columns: Optional[Tuple[str]] = None
                 ) -> None:

        self._csv_filename = pathlib.Path(path, f'{filename}.csv')
        self._filehandler = open(self._csv_filename, 'a+', newline='')
        if columns:
            pd.DataFrame([], columns=columns).to_csv(
                self._filehandler, header=True)
            self._need_header = False
        else:
            self._need_header = True

    def __del__(self) -> None:
        self._filehandler.close()

    def write_stats_output(
            self, stats_output: Dict[AgentSubjectTuple, pd.DataFrame]) -> None:
        '''Write stats to file.'''
        for s in stats_output.values():
            print(s)
            s.to_csv(self._filehandler, mode='a+', header=self._need_header)
            self._need_header = False
