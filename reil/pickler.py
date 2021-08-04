from __future__ import annotations

import bz2
import logging
import time
from pathlib import Path, PurePath
from typing import Any, Callable, List, Protocol, Union

import dill as pickle


class CustomUnPickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if name == 'MockStatistic':
            from reil.datatypes.mock_statistic import MockStatistic
            return MockStatistic
        if name == 'PrimaryComponent':
            from reil.datatypes.components import State
            return State

        return super().find_class(module, name)  # type: ignore


class LowLevelPickler(Protocol):
    ext: str

    def dump(
            self, obj: Any, filename: str,
            path: Union[str, PurePath]) -> PurePath:
        raise NotImplementedError

    def load(
            self, filename: str,
            path: Union[str, PurePath]) -> Any:
        raise NotImplementedError

    def resolve_path(
            self,
            filename: str,
            path: Union[str, PurePath]
    ) -> PurePath:
        _filename = (
            filename if filename.endswith(self.ext)
            else f'{filename}.{self.ext}')

        return PurePath(Path(path, _filename).resolve())


class DefaultPickler(LowLevelPickler):
    ext: str = 'pkl'

    @staticmethod
    def _dump(
            obj: Any,
            full_path: PurePath,
            file_fn: Callable[..., Any],
            mode: str) -> PurePath:
        _path = Path(full_path)
        _path.parent.mkdir(parents=True, exist_ok=True)
        with file_fn(_path, mode) as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)  # type: ignore

        return PurePath(_path)

    @staticmethod
    def _load(
            full_path: PurePath,
            file_fn: Callable[..., Any],
            mode: str) -> Any:
        _path = Path(full_path)
        obj: Any
        err = None
        for i in range(1, 6):
            try:
                try:
                    with file_fn(_path, mode) as f:
                        obj = pickle.load(f)  # type: ignore
                except AttributeError:
                    logging.warning(
                        f'pickle failed to load {_path}. '
                        'Using CustomUnpickler.')
                    with file_fn(_path, mode) as f:
                        obj = CustomUnPickler(f).load()
            except FileNotFoundError:
                raise
            except (EOFError, OSError) as e:
                err = e
                logging.info(
                    f'Attempt {i} failed to load {_path}.')
                time.sleep(1)
            else:
                return obj

        logging.exception(
            'Corrupted or inaccessible data file: '
            f'{full_path}')
        raise RuntimeError(
            f'Corrupted or inaccessible data file: '
            f'{full_path}') from err

    def dump(
        self, obj: Any,
        filename: str, path: Union[str, PurePath]
    ) -> PurePath:
        return self._dump(
                obj=obj,
                full_path=self.resolve_path(filename, path),
                file_fn=open, mode='wb+')

    def load(
            self, filename: str,
            path: Union[str, PurePath]) -> Any:
        return self._load(
            full_path=self.resolve_path(filename, path),
            file_fn=open, mode='rb')


class ZippedPickler(DefaultPickler):
    ext: str = 'pbz2'

    def dump(
        self, obj: Any,
        filename: str, path: Union[str, PurePath]
    ) -> PurePath:
        return self._dump(
            obj=obj, full_path=self.resolve_path(filename, path),
            file_fn=bz2.BZ2File, mode='w')

    def load(
            self, filename: str,
            path: Union[str, PurePath]) -> Any:
        return self._load(
            full_path=self.resolve_path(filename, path),
            file_fn=bz2.BZ2File, mode='r')


class PicklerManager:
    def __init__(
            self,
            low_level_picklers: List[LowLevelPickler]) -> None:
        self._low_level_picklers = {p.ext: p for p in low_level_picklers}

    def get(self, ext: str) -> LowLevelPickler:
        return self._low_level_picklers[ext]


PickleMe = PicklerManager(
    low_level_picklers=[DefaultPickler(), ZippedPickler()])
