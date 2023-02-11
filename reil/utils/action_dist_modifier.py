import math
from typing import Any, Final, Protocol

import tensorflow as tf
from tensorflow import Tensor, TensorSpec

from reil.utils.tf_utils import JIT_COMPILE

keras = tf.keras


class ScaleFn(Protocol):
    def __call__(self) -> float | Tensor:
        raise NotImplementedError


@keras.utils.register_keras_serializable(package='reil.utils.action_dist_modifier')
class N_over_N_plus_n(ScaleFn):
    def __init__(self, N: int) -> None:
        self._N: int = N
        self._n: int = 0

    @tf.function(jit_compile=JIT_COMPILE)
    def __call__(self) -> Tensor:
        self._n += 1
        return tf.cast(
            self._N / (self._N + self._n), tf.float32, name='N_over_N_plus_n')  # type: ignore

    def get_config(self) -> dict[str, Any]:
        return {'N': self._N, 'n': self._n}

    def from_config(self, config: dict[str, Any]):
        temp = N_over_N_plus_n(config.pop('N'))
        temp._n = config.pop('n')

        return temp

    def __getstate__(self) -> dict[str, Any]:
        return self.get_config()

    def __setstate__(self, config: dict[str, Any]) -> 'N_over_N_plus_n':
        return self.from_config(config)


class Sigmoid(ScaleFn):
    def __init__(self, c1: float, c2: float) -> None:
        self._c1, self._c2 = c1, c2
        self._exp_c1_c2 = tf.math.exp(c1 * c2)
        self._n: int = 0

    @tf.function(jit_compile=JIT_COMPILE)
    def __call__(self) -> Tensor:
        self._n += 1
        return 1. - self._exp_c1_c2 / (tf.math.exp(self._c1 * self._n) + self._exp_c1_c2)

    def get_config(self) -> dict[str, Any]:
        return {'c1': self._c1, 'c2': self._c2, 'n': self._n}

    def from_config(self, config: dict[str, Any]):
        temp = Sigmoid(config.pop('c1'), config.pop('c2'))
        temp._n = config.pop('n')

        return temp

    def __getstate__(self) -> dict[str, Any]:
        return self.get_config()

    def __setstate__(self, config: dict[str, Any]):
        return self.from_config(config)


class ActionModifier:
    def __init__(
            self, relative_action_distances: tuple[float, ...],
            scale_fn: ScaleFn):
        self._relative_action_distances: Tensor = tf.constant(
            relative_action_distances, dtype=tf.float32)
        self._scale_fn = scale_fn

    def __call__(self, action_distribution: Tensor) -> Tensor:
        raise NotImplementedError


class PointyHatActionModifier(ActionModifier):
    def __init__(
            self, relative_action_distances: tuple[float, ...],
            scale_fn: ScaleFn,
            height: tuple[float, float],
            width: tuple[float, float] | None = None):
        super().__init__(
            relative_action_distances=relative_action_distances,
            scale_fn=scale_fn)

        down, up = height
        if width:
            left, right = width
        else:
            left = min(relative_action_distances)
            right = max(relative_action_distances)

        x = self._relative_action_distances

        if left == -right:
            self._y = -down / right * tf.abs(x) + down
        else:
            greater_than_zero = tf.cast(
                tf.math.greater(x, 0.), dtype=tf.float32)
            self._y: Tensor = tf.add(
                tf.multiply(-down / right * x + down, greater_than_zero),
                tf.multiply(-down / left * x + down, 1. - greater_than_zero)
            )

        self._y = tf.tensor_scatter_nd_update(
            self._y, [[relative_action_distances.index(0.)]], [up])

    @tf.function(
        input_signature=(TensorSpec(
            shape=[None, None], dtype=tf.float32, name='action_distribution'),),
        jit_compile=JIT_COMPILE
    )
    def __call__(self, action_distribution: Tensor) -> Tensor:
        scale = self._scale_fn()
        return tf.add(action_distribution, tf.multiply(scale, self._y))


class RickerWaveletActionModifier(ActionModifier):
    '''
    Implements the one dimensional Ricker wavelet, a.k.a. the Mexican hat.
    This implementation assumes `t` to be fixed and `sigma` to change.
    '''
    fixed_part: Final = tf.divide(2., tf.sqrt(3.) * tf.math.pow(math.pi, 0.25), name='fixed_part')

    def __init__(
            self, relative_action_distances: tuple[float, ...],
            scale_fn: ScaleFn):
        '''
        Initialize the object with a fixed time tensor.

        Args:
            relative_action_distances: tuple[float]
                A tuple the relative distances of actions.

        Returns:
            None
        '''
        super().__init__(relative_action_distances, scale_fn)
        self.t: Tensor = self._relative_action_distances
        self.t2: Tensor = tf.math.pow(self.t, 2.)

    def __call__(self, action_distribution: Tensor) -> Tensor:
        '''
        Computes the Ricker wavelet function given `sigma`

        Returns:
        Tensor
            The computed value of the Ricker wavelet function
        '''
        coef: Tensor
        one_over_s2: Tensor
        scale = self._scale_fn()
        coef, one_over_s2 = self._prep(scale)
        return tf.add(action_distribution, self._f(coef, one_over_s2, self.t2))

    @staticmethod
    @tf.function(
        input_signature=(
            TensorSpec(shape=None, dtype=tf.float32, name='sigma'),
        ),
        jit_compile=JIT_COMPILE
    )
    def _prep(sigma: Tensor) -> Tensor:
        '''
        Prepare the coefficients and one over sigma squared for the Ricker wavelet function.

        Args:
            sigma (Tensor): A tensor of shape [None] representing the sigma values.

        Returns:
            Tensor: A tensor of shape [2, None] containing the coefficients and
            one over sigma squared.
        '''
        return tf.stack([
            tf.multiply(
                RickerWavelet.fixed_part,
                tf.math.divide(1., tf.sqrt(sigma), name='one_over_sqrt_s')),
            tf.math.divide(1., tf.math.pow(sigma, 2.), name='one_over_s2')
        ], axis=0, name='coef_and_one_over_s2')

    @staticmethod
    @tf.function(
        input_signature=(
            TensorSpec(shape=None, dtype=tf.float32, name='coef'),
            TensorSpec(shape=None, dtype=tf.float32, name='one_over_s2'),
            TensorSpec(shape=[None], dtype=tf.float32, name='t2'),
        ),
        jit_compile=JIT_COMPILE
    )
    def _f(coef: Tensor, one_over_s2: Tensor, t2: Tensor) -> Tensor:
        '''
        Compute the Ricker wavelet function.

        Args:
            coef (Tensor): A tensor of shape [2, None] containing the coefficients.
            one_over_s2 (Tensor): A tensor of shape [2, None] containing one over sigma squared.
            t2 (Tensor): A tensor of shape [None] representing the fixed time values squared.

        Returns:
            Tensor: A tensor of shape [None] representing the Ricker wavelet function.
        '''
        return tf.multiply(
            coef,
            tf.multiply(
                1. - tf.multiply(t2, one_over_s2, name='t2_over_s2'),
                tf.math.exp(
                    tf.multiply(-0.5, tf.multiply(t2, one_over_s2)),
                    name='exponential_part')
            ),
            name='ricker'
        )


class RickerWavelet:
    '''Implements the one dimensional Ricker wavelet, a.k.a. the Mexican hat'''
    fixed_part: Final = tf.divide(2., tf.sqrt(3.) * tf.math.pow(math.pi, 0.25))

    def __init__(self, sigma: float | Tensor) -> None:
        self.sigma: Tensor = tf.constant(sigma, dtype=tf.float32)
        self.coef: Tensor
        self.one_over_s2: Tensor
        self.coef, self.one_over_s2 = self._prep(self.sigma)

    def f(self, t: Tensor):
        return self._f(self.coef, self.one_over_s2, t)

    @staticmethod
    @tf.function(
        input_signature=(
            TensorSpec(shape=None, dtype=tf.float32, name='sigma'),
        ),
        jit_compile=JIT_COMPILE
    )
    def _prep(sigma: Tensor) -> Tensor:
        return tf.concat([  # type: ignore
            tf.multiply(
                RickerWavelet.fixed_part,
                tf.divide(1., tf.sqrt(sigma, name='sqrt_sigma'),
                          name='one_over_sqrt_sigma'),
                name='coef'),
            tf.divide(
                1., tf.math.pow(sigma, 2., name='sigma_sq'),
                name='one_over_s_2')
        ], axis=0, name='coef_and_one_over_s2')

    @staticmethod
    @tf.function(
        input_signature=(
            TensorSpec(shape=None, dtype=tf.float32, name='coef'),
            TensorSpec(shape=None, dtype=tf.float32, name='one_over_s2'),
            TensorSpec(shape=[None], dtype=tf.float32, name='t'),
        ),
        jit_compile=JIT_COMPILE
    )
    def _f(coef: Tensor, one_over_s2: Tensor, t: Tensor) -> Tensor:
        t2: Tensor = tf.math.pow(t, 2., name='t_sq')
        t2_over_s2 = tf.multiply(t2, one_over_s2)
        return tf.multiply(
            coef,
            tf.multiply(
                tf.subtract(1., t2_over_s2, name='one_minus_t2_over_s2'),
                tf.exp(-t2_over_s2 * 0.5, name='exponential_part'),
                name='ricker'
            )
        )
