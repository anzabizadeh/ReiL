import operator as op
from typing import Any, Callable, Literal

from reil.datatypes.buffers.circular_buffer import CircularBuffer


class StoppingCriteria:

    def __init__(
            self, monitor: str,
            mode: Literal['min', 'max'] = 'min',
            average_every: int = 1, min_delta: float = 0.,
            patience: int = 0, warm_up: int = 0) -> None:
        self._monitor = monitor
        self._mode = mode
        self._min_delta = abs(min_delta)
        self._best_weights = None

        if mode == 'min':
            self._cmp = op.lt
            self._best = float('inf')
            self._min_delta = -self._min_delta
        else:
            self._cmp = op.gt
            self._best = -float('inf')

        self._patience = patience
        self._wait: int = 0
        self._warm_up = warm_up
        self._call_counter: int = 0

        self._buffer = CircularBuffer(
            buffer_size=average_every, buffer_names=[monitor])

    def __call__(
            self, logs: dict[str, float],
            weights_fn: Callable[[], Any] | None = None) -> bool:
        self._call_counter += 1
        if self._call_counter < self._warm_up:
            return False

        if self._monitor not in logs:
            return False

        self._buffer.add({self._monitor: logs[self._monitor]})
        current: float = self._buffer.aggregate('mean').get(
            self._monitor, self._best)
        if self._cmp(current, self._best + self._min_delta):
            self._best = current
            self._wait = 0
            if weights_fn:
                self._best_weights = weights_fn()
            else:
                self._best_weights = None

            return False

        self._wait += 1
        if self._wait > self._patience:
            print(
                f'stopped at current value: {current}, best: {self._best}, '
                f'waited for: {self._wait}.')
            return True

        return False

    def get_best(self) -> tuple[float, Any] | None:
        return self._best, self._best_weights
