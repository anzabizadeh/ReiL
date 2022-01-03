from __future__ import annotations

import pathlib
import random
from typing import Any, Dict, List, Optional, Union

from reil import reilbase
from reil.learners.learning_rate_schedulers import (ConstantLearningRate,
                                                    LearningRateScheduler)
from tensorflow import keras


class SerializeTF:
    def __init__(self, temp_path: Union[str, pathlib.PurePath] = '.') -> None:
        self._temp_path = (
            pathlib.PurePath(temp_path) /
            '{n:06}'.format(n=random.randint(1, 1000000)))

    def dump(self, model: keras.Model) -> Dict[str, List[Any]]:
        path = pathlib.Path(self._temp_path)
        model.save(path)  # type: ignore
        result = self.traverse(path)
        self.__remove_dir(path)
        path.rmdir()

        return result

    def load(self, data: Dict[str, List[Any]]) -> keras.Model:
        path = pathlib.Path(self._temp_path)
        self.generate(path, data)
        sub_folder = next(iter(data))

        model = keras.models.load_model(path / sub_folder)  # type: ignore
        self.__remove_dir(path)
        path.rmdir()

        return model  # type: ignore

    @staticmethod
    def traverse(root: pathlib.Path) -> Dict[str, List[Any]]:
        result: Dict[str, List[Any]] = {root.name: []}
        for child in root.iterdir():
            if child.is_dir():
                result[root.name].append(SerializeTF.traverse(child))
            else:
                with open(child, 'rb') as f:
                    data = f.read()
                result[root.name].append({child.name: data})

        return result

    @staticmethod
    def generate(
            root: pathlib.Path, data: Dict[str, List[Any]]) -> None:
        for name, sub in data.items():
            if isinstance(sub, bytes):
                with open(root / name, 'wb+') as f:
                    f.write(sub)
            else:
                (root / name).mkdir(parents=True, exist_ok=True)
                for s in sub:
                    SerializeTF.generate(root / name, s)

    @staticmethod
    def __remove_dir(root: pathlib.Path) -> None:
        for child in root.iterdir():
            if child.is_dir():
                SerializeTF.__remove_dir(child)
                child.rmdir()
            else:
                child.unlink()


class TF2IOMixin(reilbase.ReilBase):
    def __init__(self):
        self._callbacks: List[Any]
        self._learning_rate: LearningRateScheduler
        self._tensorboard_path: Optional[pathlib.PurePath]
        self._ann_ready: bool

    def save(
            self,
            filename: Optional[str] = None,
            path: Optional[Union[str, pathlib.PurePath]] = None
    ) -> pathlib.PurePath:
        '''
        Extends `ReilBase.save` to handle `TF` objects.

        Arguments
        ---------
        filename:
            the name of the file to be saved.

        path:
            the path in which the file should be saved.

        data_to_save:
            This argument is only present for signature consistency. It has
            no effect on save.

        Returns
        -------
        :
            a `Path` object to the location of the saved file and its name
            as `str`
        '''
        _path = super().save(filename, path)

        try:
            self._model.save(pathlib.Path(  # type: ignore
                _path.parent, f'{_path.stem}.tf').resolve())
        except ValueError:
            self._logger.warning(
                'Model is not compiled. Skipped saving the model.')

        return _path

    def load(
            self,
            filename: str,
            path: Optional[Union[str, pathlib.PurePath]] = None) -> None:
        '''
        Extends `ReilBase.load` to handle `TF` objects.

        Arguments
        ---------
        filename:
            The name of the file to be loaded.

        path:
            Path of the location of the file.

        Raises
        ------
        ValueError:
            The filename is not specified.
        '''
        super().load(filename, path)

        _path = path or '.'
        if self._ann_ready:
            self._model = keras.models.load_model(  # type: ignore
                pathlib.Path(_path, f'{filename}.tf').resolve())
        else:
            self._model = keras.models.Sequential()

        if self._tensorboard_path is not None:
            self._tensorboard = keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_path)

        if not isinstance(self._learning_rate,
                          ConstantLearningRate):
            learning_rate_scheduler = \
                keras.callbacks.LearningRateScheduler(
                    self._learning_rate.new_rate, verbose=0)
            self._callbacks.append(learning_rate_scheduler)

    def __getstate__(self):
        state = super().__getstate__()
        if state['_ann_ready']:
            state['_serialized_model'] = SerializeTF().dump(state['_model'])

        del state['_model']
        del state['_callbacks']
        if '_tensorboard' in state:
            del state['_tensorboard']

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if state['_ann_ready']:
            self._model = SerializeTF().load(state['_serialized_model'])
            del state['_serialized_model']
        else:
            self._model = keras.models.Sequential()

        self.__dict__.update(state)

        self._callbacks: List[Any] = []
        if self._tensorboard_path is not None:
            self._tensorboard = keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_path)
            # , histogram_freq=1)  #, write_images=True)
            self._callbacks.append(self._tensorboard)

        if not isinstance(self._learning_rate,
                          ConstantLearningRate):
            learning_rate_scheduler = \
                keras.callbacks.LearningRateScheduler(
                    self._learning_rate.new_rate, verbose=0)
            self._callbacks.append(learning_rate_scheduler)
