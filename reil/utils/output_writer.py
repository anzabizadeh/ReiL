import pathlib
import time
from typing import Dict, Optional, Tuple, Union

import pandas as pd


class OutputWriter:
    def __init__(self,
                 filename: str,
                 path: Union[str, pathlib.PurePath] = '.',
                 columns: Optional[Tuple[str]] = None
                 ) -> None:

        self._path = pathlib.PurePath(path)
        self._csv_filename = filename if filename.endswith((
            '.yaml', '.yml')) else f'{filename}.csv'
        pathlib.Path(self._path).mkdir(parents=True, exist_ok=True)
        if columns:
            with open(self._path / self._csv_filename, 'a+', newline='') as f:
                pd.DataFrame([], columns=columns).to_csv(
                    f, header=True)

    def write_stats_output(
            self, stats_output: Dict[Tuple[str, str], pd.DataFrame]) -> None:
        '''Write stats to file.'''
        attempts = 0
        while attempts < 5 and not self._write_stats_output(stats_output):
            time.sleep(1)
            attempts += 1

        if attempts == 5:
            with open(self._path / f'{self._csv_filename}_temp',
                      'a+', newline='') as f:
                for s in stats_output.values():
                    print(s)
                    s.to_csv(f, mode='a+', header=False)

    def _write_stats_output(
            self, stats_output: Dict[Tuple[str, str], pd.DataFrame]) -> bool:
        '''Write stats to file.'''
        try:
            with open(self._path / self._csv_filename, 'a+', newline='') as f:
                for s in stats_output.values():
                    print(s)
                    s.to_csv(f, mode='a+', header=False)
        except (PermissionError,):
            return False

        return True
