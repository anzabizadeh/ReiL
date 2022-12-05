import math
from typing import Final, Union

import tensorflow as tf


class RickerWavelet:
    '''Implements the one dimensional Ricker wavelet, a.k.a. the Mexican hat'''
    fixed_part: Final = tf.divide(2., tf.sqrt(3.) * tf.math.pow(math.pi, 0.25))

    def __init__(self, sigma: Union[float, tf.Tensor]) -> None:
        self.sigma = tf.constant(sigma, dtype=tf.float32)
        self.coef, self.one_over_s2 = self._prep(self.sigma)

    def f(self, t: tf.Tensor):
        return self._f(self.coef, self.one_over_s2, t)

    @staticmethod
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=None, dtype=tf.float32, name='sigma'),
        )
    )
    def _prep(sigma: tf.Tensor):
        return tf.concat([
            tf.multiply(
                RickerWavelet.fixed_part, tf.divide(1., tf.sqrt(sigma))),
            tf.divide(1., tf.math.pow(sigma, 2.))
        ], axis=0)

    @staticmethod
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=None, dtype=tf.float32, name='coef'),
            tf.TensorSpec(shape=None, dtype=tf.float32, name='one_over_s2'),
            tf.TensorSpec(shape=[None], dtype=tf.float32, name='t'),
        )
    )
    def _f(coef: tf.Tensor, one_over_s2: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        t2 = tf.math.pow(t, 2.)
        return coef * (1. - t2 * one_over_s2) * tf.exp(-t2 * one_over_s2 * 0.5)


class RickerWavelet2:
    '''
    Implements the one dimensional Ricker wavelet, a.k.a. the Mexican hat.
    This implementation assumes `t` to be fixed and `sigma` to change.
    '''
    fixed_part: Final = tf.divide(2., tf.sqrt(3.) * tf.math.pow(math.pi, 0.25))

    def __init__(self, t: tf.Tensor) -> None:
        self.t = t
        self.t2 = tf.math.pow(t, 2.)

    def f(self, sigma: tf.Tensor) -> tf.Tensor:
        coef, one_over_s2 = self._prep(sigma)
        return self._f(coef, one_over_s2, self.t2)

    @staticmethod
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=None, dtype=tf.float32, name='sigma'),
        )
    )
    def _prep(sigma: tf.Tensor):
        return tf.concat([
            tf.multiply(
                RickerWavelet.fixed_part, tf.divide(1., tf.sqrt(sigma))),
            tf.divide(1., tf.math.pow(sigma, 2.))
        ], axis=0)

    @staticmethod
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=None, dtype=tf.float32, name='coef'),
            tf.TensorSpec(shape=None, dtype=tf.float32, name='one_over_s2'),
            tf.TensorSpec(shape=[None], dtype=tf.float32, name='t2'),
        )
    )
    def _f(coef: tf.Tensor, one_over_s2: tf.Tensor, t2: tf.Tensor):
        return tf.multiply(
            coef,
            tf.multiply(
                1. - tf.multiply(t2, one_over_s2),
                tf.exp(tf.multiply(-0.5, tf.multiply(t2, one_over_s2)))
            )
        )
