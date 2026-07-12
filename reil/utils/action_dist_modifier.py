import math
from typing import Any, Final, Protocol

import tensorflow as tf
from tensorflow import Tensor, TensorSpec

from reil.utils.tf_utils import JIT_COMPILE

keras = tf.keras


class ScaleFn:
    def __init__(self) -> None:
        self.last_call: float | Tensor = 0.

    def call(self) -> float | Tensor:
        raise NotImplementedError

    def __call__(self) -> float | Tensor:
        self.last_call = self.call()
        return self.last_call


class Constant(ScaleFn):
    def __init__(self, value: float) -> None:
        self._value = value

    def call(self) -> float:
        return self._value

    def get_config(self) -> dict[str, Any]:
        return {'value': self._value}

    def from_config(self, config: dict[str, Any]):
        return Constant(config.pop('value'))

    def __getstate__(self) -> dict[str, Any]:
        return self.get_config()

    def __setstate__(self, config: dict[str, Any]):
        self._value = config.pop('value')


@keras.utils.register_keras_serializable(package='reil.utils.action_dist_modifier')
class N_over_N_plus_n(ScaleFn):
    def __init__(self, N: int) -> None:
        self._N: int = N
        self._n: int = 0

    @tf.function(jit_compile=JIT_COMPILE)
    def call(self) -> Tensor:
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

    def __setstate__(self, config: dict[str, Any]):
        self._N = config.pop('N')
        self._n = config.pop('n')


class Sigmoid(ScaleFn):
    def __init__(self, steepness: float, endpoint: float) -> None:
        self._steepness, self._endpoint = steepness, endpoint
        self._n: int = 0

    def call(self) -> float:
        self._n += 1
        return 1 / (1 + math.exp(-self._steepness * (self._n - self._endpoint * 0.5)))

    def get_config(self) -> dict[str, Any]:
        return {
            'steepness': self._steepness,
            'endpoint': self._endpoint,
            'n': self._n
        }

    def from_config(self, config: dict[str, Any]):
        temp = Sigmoid(config.pop('steepness'), config.pop('endpoint'))
        temp._n = config.pop('n')

        return temp

    def __getstate__(self) -> dict[str, Any]:
        return self.get_config()

    def __setstate__(self, config: dict[str, Any]):
        self._steepness = config.pop('steepness')
        self._endpoint = config.pop('endpoint')
        self._n = config.pop('n')


class ProgressRamp(ScaleFn):
    '''A schedule that ramps the modifier scale from `start` to `end` as a
    function of *fraction-of-training-complete*, rather than saturating early.

    Unlike `Sigmoid` — whose per-decision counter `n` saturates inside the
    first training chunk — this schedule normalises the same per-call counter
    by `total_steps` (the expected total number of decisions over the whole
    run), so the scale changes *gradually and observably* across training.

        progress p = clip(n / total_steps, 0, 1)
        shape='linear': scale = start + (end - start) * p
        shape='cosine': scale = start + (end - start) * 0.5 * (1 - cos(pi*p))
                        (ease-in-out S-curve; gradual at both ends)

    `total_steps` is in the same unit as `n` (one increment per call, i.e. per
    main-phase training decision). Set it from the measured decision budget of
    a 5K/1K run. The counter is serialized so the ramp survives checkpointing
    and resumed training (matching `Sigmoid`).
    '''

    def __init__(
            self, total_steps: float, shape: str = 'cosine',
            start: float = 0.0, end: float = 1.0) -> None:
        if total_steps <= 0:
            raise ValueError('total_steps must be positive')
        if shape not in ('linear', 'cosine'):
            raise ValueError(f"shape must be 'linear' or 'cosine', got {shape!r}")
        self._total_steps = float(total_steps)
        self._shape = shape
        self._start = float(start)
        self._end = float(end)
        self._n: int = 0

    def call(self) -> float:
        self._n += 1
        p = self._n / self._total_steps
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0
        if self._shape == 'cosine':
            p = 0.5 * (1.0 - math.cos(math.pi * p))
        return self._start + (self._end - self._start) * p

    def get_config(self) -> dict[str, Any]:
        return {
            'total_steps': self._total_steps,
            'shape': self._shape,
            'start': self._start,
            'end': self._end,
            'n': self._n,
        }

    def from_config(self, config: dict[str, Any]):
        temp = ProgressRamp(
            config.pop('total_steps'), config.pop('shape'),
            config.pop('start'), config.pop('end'))
        temp._n = config.pop('n')
        return temp

    def __getstate__(self) -> dict[str, Any]:
        return self.get_config()

    def __setstate__(self, config: dict[str, Any]):
        self._total_steps = config.pop('total_steps')
        self._shape = config.pop('shape')
        self._start = config.pop('start')
        self._end = config.pop('end')
        self._n = config.pop('n')


class ActionModifier:
    def __init__(
            self, relative_action_distances: tuple[float, ...],
            scale_fn: ScaleFn, name: str = 'action_modifier'):
        self.name = name
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
            width: tuple[float, float] | None = None,
            name: str = 'pointyhat'):
        super().__init__(
            relative_action_distances=relative_action_distances,
            scale_fn=scale_fn, name=name)

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
            greater_than_zero: tf.Tensor = tf.cast(  # type: ignore
                tf.math.greater(x, 0.), dtype=tf.float32)
            self._y: Tensor = tf.add(
                tf.multiply(
                    tf.multiply(-down / right, x) + down, greater_than_zero),
                tf.multiply(
                    tf.multiply(
                        -down / left, x) + down,
                        1. - greater_than_zero)  # type: ignore
            )

        self._y = tf.tensor_scatter_nd_update(
            self._y, [[relative_action_distances.index(0.)]], [up])

    # @tf.function(
    #     input_signature=(TensorSpec(
    #         shape=[None, None], dtype=tf.float32, name='action_distribution'),),
    #     jit_compile=JIT_COMPILE
    # )
    def __call__(self, action_distribution: Tensor) -> Tensor:
        scale = self._scale_fn()
        return tf.add(
            action_distribution,
            tf.expand_dims(tf.multiply(scale, self._y), axis=0)
        )


class CombActionModifier(ActionModifier):
    '''Periodic "pointy-comb" — Action Focus toward action VALUES that are
    multiples of `period` (Paper-3 duration forging onto {7, 14, 21, 28}).

    Unlike `PointyHatActionModifier` (one hat at the 0-index no-change action),
    the bias vector `y` is built directly from the action *values* (e.g. the
    duration grid 1..28) with a per-value rule (height = (down, up)):

        y_v = up     if v is a nonzero multiple of `period`   (7, 14, 21, 28)
        y_v = down   if v > period and v is NOT a multiple     (8..13, 15..20, …)
        y_v = 0      if v < period                             (1..6 left untouched)

    So it ATTRACTS the policy onto the weekly-multiple intervals and SUPPRESSES
    the non-multiple intervals above one period, while leaving the short
    "test-soon" intervals (< period) unbiased. Applied to the actor logits as
    `logits + scale * y` (training-only, scheduled by `scale_fn`), exactly like
    the pointy hat.
    '''
    def __init__(
            self, action_values: tuple[float, ...],
            scale_fn: ScaleFn,
            period: float,
            height: tuple[float, float],
            name: str = 'comb'):
        super().__init__(
            relative_action_distances=action_values,
            scale_fn=scale_fn, name=name)
        if period <= 0:
            raise ValueError('period must be positive')
        down, up = height
        y: list[float] = []
        for v in action_values:
            vi = int(round(v))
            if vi != 0 and vi % int(round(period)) == 0:
                y.append(float(up))
            elif v > period:
                y.append(float(down))
            else:
                y.append(0.0)
        self._y: Tensor = tf.constant(y, dtype=tf.float32)

    def __call__(self, action_distribution: Tensor) -> Tensor:
        scale = self._scale_fn()
        return tf.add(
            action_distribution,
            tf.expand_dims(tf.multiply(scale, self._y), axis=0)
        )


class RickerWaveletActionModifier(ActionModifier):
    '''
    Implements the one dimensional Ricker wavelet, a.k.a. the Mexican hat.
    This implementation assumes `t` to be fixed and `sigma` to change.
    '''
    fixed_part: Final = tf.divide(
        2., tf.sqrt(3.) * tf.math.pow(math.pi, 0.25), name='fixed_part')

    def __init__(
            self, relative_action_distances: tuple[float, ...],
            scale_fn: ScaleFn, name: str = 'ricker_wavelet'):
        '''
        Initialize the object with a fixed time tensor.

        Args:
            relative_action_distances: tuple[float]
                A tuple the relative distances of actions.

        Returns:
            None
        '''
        super().__init__(relative_action_distances, scale_fn, name)

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
