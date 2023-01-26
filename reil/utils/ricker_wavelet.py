import math
from typing import Final

import tensorflow as tf
from tensorflow import Tensor, TensorSpec

from reil.utils.tf_utils import JIT_COMPILE


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
                tf.divide(1., tf.sqrt(sigma, name='sqrt_sigma'), name='one_over_sqrt_sigma'),
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


class RickerWavelet2:
    '''
    Implements the one dimensional Ricker wavelet, a.k.a. the Mexican hat.
    This implementation assumes `t` to be fixed and `sigma` to change.
    '''
    fixed_part: Final = tf.divide(2., tf.sqrt(3.) * tf.math.pow(math.pi, 0.25), name='fixed_part')

    def __init__(self, t: Tensor) -> None:
        self.t: Tensor = t
        self.t2: Tensor = tf.math.pow(t, 2.)

    def f(self, sigma: Tensor) -> Tensor:
        coef: Tensor
        one_over_s2: Tensor
        coef, one_over_s2 = self._prep(sigma)
        return self._f(coef, one_over_s2, self.t2)

    @staticmethod
    @tf.function(
        input_signature=(
            TensorSpec(shape=None, dtype=tf.float32, name='sigma'),
        ),
        jit_compile=JIT_COMPILE
    )
    def _prep(sigma: Tensor) -> Tensor:
        return tf.stack([
            tf.multiply(
                RickerWavelet.fixed_part,
                tf.divide(1., tf.sqrt(sigma), name='one_over_sqrt_s')),
            tf.divide(1., tf.math.pow(sigma, 2.), name='one_over_s2')
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
        return tf.multiply(
            coef,
            tf.multiply(
                1. - tf.multiply(t2, one_over_s2, name='t2_over_s2'),
                tf.exp(
                    tf.multiply(-0.5, tf.multiply(t2, one_over_s2)),
                    name='exponential_part')
            ),
            name='ricker'
        )
