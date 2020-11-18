import pathlib
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from reil import learners, rldata
from tensorflow import keras  # type: ignore


class Dense(learners.Learner):
    def __init__(self,
                 learning_rate: learners.LearningRateScheduler,
                 validation_split: float = 0.3,
                 hidden_layer_sizes: Sequence[int] = (1,),
                 input_length: Optional[int] = None,
                 tensorboard_path: Optional[Union[str, pathlib.Path]] = None,
                 ) -> None:

        super().__init__(learning_rate=learning_rate)

        self._epoch = 0

        self._hidden_layer_sizes = hidden_layer_sizes
        self._input_length = input_length

        if not 0.0 < validation_split < 1.0:
            raise ValueError('validation split should be in (0.0, 1.0).')
        self._validation_split = validation_split

        self._tensorboard_path = tensorboard_path

        if self._input_length is not None:
            self._generate_network()
        else:
            self._graph = None

    def _generate_network(self) -> None:
        '''
        Generate a tensorflow ANN network.
        '''

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._session = tf.Session()

            self._model = keras.models.Sequential()
            self._model.add(keras.layers.Dense(
                self._hidden_layer_sizes[0], activation='relu', name='layer_01', input_shape=(self._input_length,)))
            for i, v in enumerate(self._hidden_layer_sizes[1:]):
                self._model.add(keras.layers.Dense(
                    v, activation='relu', name=f'layer_{i+2:0>2}'))

            self._model.add(keras.layers.Dense(
                1, name='output'))

            self._model.compile(optimizer=keras.optimizers.Adam(
                learning_rate=self._learning_rate.initial_lr), loss='mae')

            self._callbacks = []
            if self._tensorboard_path is not None:
                self._tensorboard_path = pathlib.Path(
                    'logs', self._tensorboard_path)
                self._tensorboard = keras.callbacks.TensorBoard(
                    log_dir=self._tensorboard_path)  # , histogram_freq=1)  #, write_images=True)
                self._callbacks.append(self._tensorboard)

            if not isinstance(self._learning_rate, learners.ConstantLearningRate):
                self._learning_rate_scheduler = keras.callbacks.LearningRateScheduler(
                    self._learning_rate.new_rate, verbose=1)
                self._callbacks.append(self._learning_rate_scheduler)

    def predict(self, X: List[rldata.RLData]) -> float:
        _X = [x.normalized.flatten() for x in X]
        if self._graph is None:
            self._input_length = len(_X[0])
            self._generate_network()

        with self._session.as_default():
            with self._graph.as_default():
                result = self._model.predict(np.array(_X))

        return result

    def learn(self, X: List[rldata.RLData], Y: List[float]) -> None:
        _X = [x.normalized.flatten() for x in X]
        if self._graph is None:
            self._input_length = len(_X[0])
            self._generate_network()

        with self._session.as_default():
            with self._graph.as_default():
                self._model.fit(np.array(_X), np.array(Y),
                                initial_epoch=self._epoch, epochs=self._epoch+1,
                                callbacks=self._callbacks,
                                validation_split=self._validation_split,
                                verbose=2)

    def reset(self) -> None:
        self._epoch += 1

    def save(self,
             filename: str,
             path: pathlib.Path) -> Tuple[pathlib.Path, str]:
        path.mkdir(parents=True, exist_ok=True)
        with self._session.as_default():
            with self._graph.as_default():
                self._model.save(path / filename)

        return path, filename

    def load(self, filename: str, path: Optional[Union[str, pathlib.Path]] = None) -> None:
        _path = path if path is not None else '.'
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._session = keras.backend.get_session()
            self._model = keras.models.load_model(
                pathlib.Path(_path, f'{filename}.tf', filename))
            self._tensorboard = keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_path)  # , histogram_freq=1)  # , write_images=True)
            self._learning_rate_scheduler = keras.callbacks.LearningRateScheduler(
                self._learning_rate_scheduler)
