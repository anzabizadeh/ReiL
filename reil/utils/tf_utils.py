from __future__ import annotations

import pathlib
import random
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

import tensorflow as tf
from tensorflow import Tensor, TensorShape, TensorSpec

from reil import reilbase
from reil.datatypes.feature import FeatureSet
from reil.learners.learning_rate_schedulers import (ConstantLearningRate,
                                                    LearningRateScheduler)

keras = tf.keras

EAGER_EXECUTION: bool = False
JIT_COMPILE: bool = True


def set_tf_flags(
        eager_execution: bool | None = None,
        jit_compile: bool | Literal['autoclustering'] | None = None):
    global EAGER_EXECUTION
    global JIT_COMPILE

    if eager_execution is not None:
        EAGER_EXECUTION = eager_execution
        tf.config.run_functions_eagerly(EAGER_EXECUTION)

    if jit_compile is not None:
        if jit_compile == 'autoclustering':
            JIT_COMPILE = True
            tf.config.optimizer.set_jit("autoclustering")  # "autoclustering" or False
        else:
            JIT_COMPILE = jit_compile
        if not jit_compile:
            tf.config.optimizer.set_jit(False)


@tf.function(jit_compile=JIT_COMPILE)
def entropy(logits: Tensor) -> Tensor:
    # Adopted the code from OpenAI baseline
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/distributions.py
    a0 = tf.subtract(logits, tf.reduce_max(logits, axis=-1, keepdims=True))
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p0 = tf.divide(ea0, z0)

    return tf.reduce_sum(  # type: ignore
        tf.multiply(p0, tf.subtract(tf.math.log(z0), a0)), axis=-1)


@tf.function(
    input_signature=(
        TensorSpec(shape=[None, None], dtype=tf.float32, name='logits'),
        TensorSpec(shape=[None], dtype=tf.int32, name='indices'),
        TensorSpec(shape=[], dtype=tf.int32, name='index_count'),),
    jit_compile=False  # tf.onehot requires a compile-time constant arg 1.
)
def logprobs(logits: Tensor, indices: Tensor, index_count: Tensor) -> Tensor:
    return -tf.nn.softmax_cross_entropy_with_logits(  # type: ignore
        tf.one_hot(indices, index_count), logits, name='logprobs')


class SerializeTF:
    def __init__(
            self, cls: type[keras.Model] | None = None,
            temp_path: str | pathlib.PurePath = '.') -> None:
        self.cls = cls
        self._temp_path = (
            pathlib.PurePath(temp_path) /
            '{n:06}'.format(n=random.randint(1, 1000000)))

    def dump(self, model: keras.Model) -> dict[str, list[Any]]:
        path = pathlib.Path(self._temp_path)
        try:
            model.save(path)  # type: ignore
            result = self.traverse(path)
            self.__remove_dir(path)
            path.rmdir()
        except ValueError:  # model is not compiled.
            result = model.get_config()

        return result

    def load(self, data: dict[str, list[Any]]) -> keras.Model:
        path = pathlib.Path(self._temp_path)
        try:
            self.generate(path, data)
            sub_folder = next(iter(data))

            model = keras.models.load_model(path / sub_folder)
            self.__remove_dir(path)
            path.rmdir()
        except (AttributeError, TypeError):  # model not compiled.
            cls = self.cls or keras.models.Model
            model = cls.from_config(data)  # type: ignore

        return model  # type: ignore

    @staticmethod
    def traverse(root: pathlib.Path) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = {root.name: []}
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
            root: pathlib.Path, data: dict[str, list[Any]]) -> None:
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


class TF2UtilsMixin(reilbase.ReilBase):
    def __init__(
            self, models: dict[str, type[keras.Model]], **kwargs):
        super().__init__(**kwargs)
        self._no_model: bool
        self._models = models
        self._callbacks: list[Any]
        self._learning_rate: LearningRateScheduler
        self._tensorboard_path: pathlib.PurePath | None

    @staticmethod
    def convert_to_tensor(data: tuple[FeatureSet, ...]) -> Tensor:
        return tf.convert_to_tensor([x.normalized.flattened for x in data])

    @staticmethod
    def mlp_layers(
            layer_sizes: tuple[int, ...],
            activation: str | Callable[[Tensor], Tensor] | None,
            layer_name_format: str,
            start_index: int = 1, **kwargs):
        kernel_initializer = kwargs.pop(
            'kernel_initializer') if 'kernel_initializer' in kwargs else 'zeros'
        bias_initializer = kwargs.pop(
            'bias_initializer') if 'bias_initializer' in kwargs else 'zeros'

        return [
            keras.layers.Dense(
                v, activation=activation, name=layer_name_format.format(i=i),
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                **kwargs)
            for i, v in enumerate(layer_sizes, start_index)]

    @staticmethod
    def mlp_functional(
            input_: Tensor,
            layer_sizes: tuple[int, ...],
            activation: str | Callable[[Tensor], Tensor],
            layer_name_format: str = 'layer_{i:0>2}',
            start_index: int = 1, **kwargs) -> Tensor:
        '''Build a feedforward dense network.'''
        layers = TF2UtilsMixin.mlp_layers(
            layer_sizes, activation, layer_name_format,
            start_index, **kwargs)
        x = input_
        for layer in layers:
            x = layer(x)

        return x  # type: ignore

    @staticmethod
    def mlp_functional_w_concat(
            input_: Tensor,
            layer_sizes: dict[str, tuple[int, ...]],
            activation: str | Callable[[Tensor], Tensor],
            action_per_head: tuple[int, ...],
            head_activation: str | None = None,
            layer_name_format: str = 'layer_{i:0>2}',
            output_name_format: str = 'output_{i:0>2}',
            start_index: int = 1,
            backprop_mode: Literal['separate', 'shared', 'all'] = 'all',
            normalize_before_concat: Literal['regular', 'batch', 'none'] = 'none',
            **kwargs):
        '''Build a feedforward dense network.'''
        layers_iterable = iter(layer_sizes.items())
        first_layer_name, first_layer_sizes = next(layers_iterable)
        index = layer_name_format.find('{i')
        layer_name_format_new = ''.join(
            (layer_name_format[:index], '_', first_layer_name, '_',
             layer_name_format[index:]))

        logit_heads = TF2UtilsMixin.mlp_layers(
            action_per_head, head_activation, output_name_format)
        for name, layer in zip(layer_sizes, logit_heads):
            layer._name = layer.name.replace(
                'output', f'{name}_output')[:-3]

        layers = [
            TF2UtilsMixin.mlp_functional(
                input_, first_layer_sizes, activation,
                layer_name_format_new,
                start_index, **kwargs)
        ]

        logits = [
            logit_heads[0](
                tf.stop_gradient(layers[-1]) if backprop_mode == 'separate' else layers[-1])
        ]

        for i, (layer_name_i, layer_sizes_i) in enumerate(layers_iterable, 1):
            layer_name_format_new = ''.join(
                (layer_name_format[:index], '_', layer_name_i, '_',
                 layer_name_format[index:]))

            if normalize_before_concat == 'none':
                normalized_previous_layer = layers[-1]
            elif normalize_before_concat == 'regular':
                normalized_previous_layer = tf.math.l2_normalize(layers[-1])
            elif normalize_before_concat == 'batch':
                normalized_previous_layer = tf.keras.layers.BatchNormalization(
                    name='_'.join(
                        (layer_name_format[:index], layer_name_i, 'pre_batch_normalization'))
                )(logits[-1])
            if backprop_mode == 'separate':
                normalized_previous_layer = tf.stop_gradient(normalized_previous_layer)

            temp = tf.concat([input_, normalized_previous_layer], axis=-1)

            layers.append(
                TF2UtilsMixin.mlp_functional(
                    temp, layer_sizes_i, activation, layer_name_format_new,
                    start_index, **kwargs)
            )

            logits.append(
                logit_heads[i](
                    tf.stop_gradient(layers[-1]) if backprop_mode == 'separate' else layers[-1])
            )

        return tuple(logits)

    def save(
            self,
            filename: str | None = None, path: str | pathlib.PurePath | None = None
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
            filename: str, path: str | pathlib.PurePath | None = None) -> None:
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
                log_dir=self._tensorboard_path.__str__())

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

        for k in list(self._models) + ['_callbacks']:
            if k in state:
                del state[k]

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        _models: dict[str, Any] | list[str]
        try:
            _models = state['_models']
        except KeyError:
            _models = ['model']

        if '_no_model' in state and state['_no_model']:  # for compatibility
            for name, model in _models.items():  # type: ignore
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

        self._callbacks: list[Any] = []
        if '_learning_rate' in state and not isinstance(
                self._learning_rate, ConstantLearningRate):
            learning_rate_scheduler = \
                keras.callbacks.LearningRateScheduler(
                    self._learning_rate.new_rate, verbose=0)
            self._callbacks.append(learning_rate_scheduler)


@keras.utils.register_keras_serializable(
    package='reil.utils.tf_utils')
class BroadcastAndConcatLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(trainable=False, dynamic=True, **kwargs)

    @staticmethod
    @tf.function(
        input_signature=(
            TensorSpec(shape=[None, None], name='x'),
            TensorSpec(shape=[None, None], name='y')),
    )
    def _broadcast_and_concat(x: Tensor, y: Tensor):
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

    def call(self, x: list[Tensor]):
        return self._broadcast_and_concat(x[0], x[1])

    def compute_output_shape(self, input_shape):
        return TensorShape([None, sum(shape[1] for shape in input_shape)])

    def count_params(self):
        return 0

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.pop('trainable')
        config.pop('dynamic')

        return config


@keras.utils.register_keras_serializable(
    package='reil.utils.tf_utils')
class ArgMaxLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(trainable=False, dynamic=True, **kwargs)

    @tf.function(
        input_signature=(TensorSpec(shape=[None, None]),),
        jit_compile=JIT_COMPILE
    )
    def call(self, x: Tensor):
        return tf.argmax(x)

    def compute_output_shape(self, input_shape):
        return TensorShape([1])

    def count_params(self):
        return 0

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.pop('trainable')
        config.pop('dynamic')

        return config


@keras.utils.register_keras_serializable(
    package='reil.utils.tf_utils')
class MaxLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(trainable=False, dynamic=True, **kwargs)
        super().build([])

    @tf.function(
        input_signature=(
            TensorSpec(shape=[None, 1]),),
        jit_compile=JIT_COMPILE)
    def call(self, x: Tensor):
        return tf.reduce_max(x)

    def compute_output_shape(self, input_shape):
        return TensorShape([1])

    def count_params(self):
        return 0

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.pop('trainable')
        config.pop('dynamic')

        return config


if TYPE_CHECKING:
    class MetricSerializerMixinProtocol(Protocol):
        def __init__(self, name: str, *args, **kwargs) -> None:
            ...
        name: str
        variables: list[Any]
        _unconditional_dependency_names: dict[str, Any]
else:
    class MetricSerializerMixinProtocol(Protocol):
        ...


class MetricSerializerMixin:
    def __getstate__(self: MetricSerializerMixinProtocol) -> dict[str, Any]:
        variables = {v.name: v.numpy() for v in self.variables}
        state = {
            name: variables[var.name]
            for name, var in self._unconditional_dependency_names.items()
            if isinstance(var, tf.Variable)}
        state['name'] = self.name
        return state

    def __setstate__(self: MetricSerializerMixinProtocol, state: dict[str, Any]):
        self.__init__(name=state.pop('name'))
        for name, value in state.items():
            self._unconditional_dependency_names[name].assign(value)


@keras.utils.register_keras_serializable(
    package='reil.utils.tf_utils')
class ActionRank(MetricSerializerMixin, keras.metrics.Metric):
    def __init__(self, name='action_rank', **kwargs):
        super().__init__(name=name, **kwargs)  # type: ignore
        self.cumulative_rank = self.add_weight(
            name='cumulative_rank', initializer='zeros', dtype=tf.int64)
        self.count = self.add_weight(
            name='count', initializer='zeros', dtype=tf.int32)

    @tf.function(
        input_signature=(
            TensorSpec(shape=[None], dtype=tf.int32, name='y_true'),
            TensorSpec(shape=[None, None], dtype=tf.float32, name='y_pred')),
        jit_compile=JIT_COMPILE
    )
    def update_state(self, y_true, y_pred, sample_weight=None):
        shape = tf.shape(y_pred)
        ranks = tf.math.top_k(y_pred, k=shape[1]).indices
        self.count.assign_add(shape[0])  # type: ignore
        locs = tf.equal(ranks, tf.expand_dims(y_true, axis=1))
        self.cumulative_rank.assign_add(  # type: ignore
            tf.reduce_sum(tf.gather(tf.where(locs), 1, axis=1)))

    @tf.function(jit_compile=JIT_COMPILE)
    def result(self):
        # ranks are zero-based. Add one to make it 1-based, which is more
        # intuitive.
        return tf.reduce_sum((tf.divide(
            self.cumulative_rank, tf.cast(self.count, tf.int64)), 1.0))

    def reset_states(self):
        self.cumulative_rank.assign(0)  # type: ignore
        self.count.assign(0)  # type: ignore


class MeanMetric(MetricSerializerMixin, keras.metrics.Mean):
    pass


class SparseCategoricalAccuracyMetric(
        MetricSerializerMixin, tf.keras.metrics.SparseCategoricalAccuracy):
    pass


class SummaryWriter:
    def __init__(
            self,
            tensorboard_path: str | pathlib.PurePath | None = None,
            tensorboard_filename: str | None = None):

        self._data_types = {}
        self._tensorboard_path: pathlib.PurePath | None = None
        if (tensorboard_path or tensorboard_filename) is not None:
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._tensorboard_path = pathlib.PurePath(
                tensorboard_path or './logs')
            self._tensorboard_filename: str = current_time + (
                f'-{tensorboard_filename}' or '')
            self._summary_writer = \
                tf.summary.create_file_writer(  # type: ignore
                    str(self._tensorboard_path / self._tensorboard_filename))
        else:
            self._summary_writer = tf.summary.create_noop_writer()  # type: ignore

    def set_data_types(
            self, data_types: dict[str, Literal['scalar', 'histogram']]):
        for k, v in data_types.items():
            if v == 'scalar':
                self._data_types[k] = tf.summary.scalar
            elif v == 'histogram':
                self._data_types[k] = tf.summary.histogram
            else:
                raise ValueError(f'Unsupported type: {v}.')

    def __getstate__(self):
        state = dict(
            tensorboard_filename=self._tensorboard_filename,
            tensorboard_path=self._tensorboard_path,
            data_types={k: v.__name__ for k, v in self._data_types.items()}
        )

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(
            tensorboard_filename=state.pop('tensorboard_filename'),
            tensorboard_path=state.pop('tensorboard_path'))
        self.set_data_types(state.pop('data_types'))
        # self._tensorboard_filename = state['_tensorboard_filename']
        # self._tensorboard_path = state['_tensorboard_path']

        # if self._tensorboard_path:
        #     self._summary_writer = \
        #         tf.summary.create_file_writer(  # type: ignore
        #             str(
        #                 self._tensorboard_path /
        #                 self._tensorboard_filename))
        # else:
        #     self._summary_writer = tf.summary.create_noop_writer()  # type: ignore

    def write(self, data: dict[str, float], iteration: int | None = None):
        with self._summary_writer.as_default(step=iteration):
            for name, value in data.items():
                self._data_types.get(name, tf.summary.scalar)(name, value)
