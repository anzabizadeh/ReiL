from __future__ import annotations

import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import tensorflow as tf
from reil import reilbase
from reil.datatypes.feature import FeatureSet
from reil.learners.learning_rate_schedulers import (ConstantLearningRate,
                                                    LearningRateScheduler)
from tensorflow import keras


class SerializeTF:
    def __init__(
            self, cls: Optional[Type[keras.Model]] = None,
            temp_path: Union[str, pathlib.PurePath] = '.') -> None:
        self.cls = cls
        self._temp_path = (
            pathlib.PurePath(temp_path) /
            '{n:06}'.format(n=random.randint(1, 1000000)))

    def dump(self, model: keras.Model) -> Dict[str, List[Any]]:
        path = pathlib.Path(self._temp_path)
        try:
            model.save(path)  # type: ignore
            result = self.traverse(path)
            self.__remove_dir(path)
            path.rmdir()
        except ValueError:  # model is not compiled.
            result = model.get_config()

        return result

    def load(self, data: Dict[str, List[Any]]) -> keras.Model:
        path = pathlib.Path(self._temp_path)
        try:
            self.generate(path, data)
            sub_folder = next(iter(data))

            model = keras.models.load_model(path / sub_folder)
            self.__remove_dir(path)
            path.rmdir()
        except AttributeError:  # model not compiled.
            cls = self.cls or keras.models.Model
            model = cls.from_config(data)

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
    def __init__(
            self, models: Dict[str, Type[keras.Model]], **kwargs):
        super().__init__(**kwargs)
        self._no_model: bool
        self._models = models
        self._callbacks: List[Any]
        self._learning_rate: LearningRateScheduler
        self._tensorboard_path: Optional[pathlib.PurePath]

    @staticmethod
    def convert_to_tensor(data: Tuple[FeatureSet, ...]) -> tf.Tensor:
        return tf.convert_to_tensor([x.normalized.flattened for x in data])

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

        for model in self._models:
            try:
                self.__dict__[model].save(pathlib.Path(  # type: ignore
                    _path.parent, f'{_path.stem}_{model}.tf').resolve())
            except ValueError:
                self._logger.warning(
                    f'{model} is not compiled. Skipped saving the model.')

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
        if self._no_model:
            self._model = keras.Model()
        else:
            for model in self._models:
                self.__dict__[model] = keras.models.load_model(  # type: ignore
                    pathlib.Path(_path, f'{filename}_{model}.tf').resolve())

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
        if '_no_model' in state and state['_no_model']:
            pass
        else:
            for name, model in self._models.items():
                try:
                    state[f'_serialized_{name}'] = SerializeTF(
                        cls=model).dump(state[name])
                except ValueError:
                    state[f'_serialized_{name}'] = None

        for k in list(self._models) + ['_callbacks', '_tensorboard']:
            if k in state:
                del state[k]

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        _models = state['_models']
        if '_no_model' in state and state['_no_model']:  # for compatibility
            for name, model in _models.items():
                self.__dict__[name] = model
        elif isinstance(_models, list):  # for compatibility
            for name in _models:
                self.__dict__[name] = SerializeTF().load(
                    state[f'_serialized_{name}'])
            del state[f'_serialized_{name}']
        else:
            for name, model in _models.items():
                self.__dict__[name] = SerializeTF(
                    cls=model).load(state[f'_serialized_{name}'])
            del state[f'_serialized_{name}']

        self.__dict__.update(state)

        self._callbacks: List[Any] = []
        if self._tensorboard_path is not None:
            self._tensorboard = keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_path)
            # , histogram_freq=1)  #, write_images=True)
            self._callbacks.append(self._tensorboard)

        if '_learning_rate' in state and not isinstance(
                self._learning_rate, ConstantLearningRate):
            learning_rate_scheduler = \
                keras.callbacks.LearningRateScheduler(
                    self._learning_rate.new_rate, verbose=0)
            self._callbacks.append(learning_rate_scheduler)


class BroadcastAndConcatLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(trainable=False, dynamic=True, **kwargs)

    @staticmethod
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], name='x'),
            tf.TensorSpec(shape=[None, None], name='y')))
        # experimental_compile=True)
    def _broadcast_and_concat(x: tf.Tensor, y: tf.Tensor):
        dim_x = tf.shape(x)
        dim_y = tf.shape(y)
        if tf.equal(dim_x[0], dim_y[0]):
            return tf.concat([x, y], axis=1)
        elif tf.equal(dim_x[0], 1):
            return tf.concat(
                [tf.broadcast_to(x, (dim_y[0], dim_x[1])), y], axis=1)
        elif tf.equal(dim_y[0], 1):
            return tf.concat([
                x, tf.broadcast_to(y, (dim_x[0], dim_y[1]))], axis=1)
        else:
            return tf.convert_to_tensor([])
        # raise ValueError(
        #     'Dimensions should be of the same'
        #     ' or all but one should be of size one.')

    def call(self, x: List[tf.Tensor]):
        return self._broadcast_and_concat(x[0], x[1])

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, sum(shape[1] for shape in input_shape)])

    def count_params(self):
        return 0

    def get_config(self) -> Dict[str, Any]:
        return super().get_config()


class ArgMaxLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(trainable=False, dynamic=True, **kwargs)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None]),))
        # experimental_compile=True)
    def call(self, x: tf.Tensor):
        return tf.argmax(x)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1])

    def count_params(self):
        return 0

    def get_config(self) -> Dict[str, Any]:
        return super().get_config()


class MaxLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(trainable=False, dynamic=True, **kwargs)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, 1]),))
        # experimental_compile=True)
    def call(self, x: tf.Tensor):
        return tf.reduce_max(x)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1])

    def count_params(self):
        return 0

    def get_config(self) -> Dict[str, Any]:
        return super().get_config()
