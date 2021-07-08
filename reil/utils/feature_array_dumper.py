from __future__ import annotations

import pathlib
import time
from typing import Optional, Tuple, Union

import pandas as pd
from reil.datatypes.feature import FeatureArray


class FeatureArrayDumper:
    def __init__(
            self, filename: str,
            path: Union[str, pathlib.PurePath] = '.',
            columns: Optional[Tuple[str]] = None) -> None:
        self._path = pathlib.PurePath(path)
        self._filename = filename if filename.endswith(
            '.csv') else f'{filename}.csv'
        pathlib.Path(self._path).mkdir(parents=True, exist_ok=True)
        if columns:
            with open(self._path / self._filename, 'a+', newline='') as f:
                pd.DataFrame([], columns=columns).to_csv(
                    f, header=True)

    def dump(self, component: FeatureArray) -> None:
        '''Write stats to file.'''
        attempts = 0
        while attempts < 5 and not self._dump(
                component, self._filename, self._path):
            time.sleep(1)
            attempts += 1

        if attempts == 5:
            self._dump(component, f'{self._filename}_temp', self._path)

    @staticmethod
    def _dump(
            component: FeatureArray,
            filename: str, path: pathlib.PurePath) -> bool:
        '''Write stats to file.'''
        raise NotImplementedError
