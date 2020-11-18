import csv
import pathlib
from typing import List, Optional, Tuple, Union

from reil import learners, rldata


class LookupTable(learners.Learner):
    def __init__(self,
                 learning_rate: learners.LearningRateScheduler,
                 initial_reward_estimate: float = 0.0,
                 minimum_visits: int = 0) -> None:
        self._learning_rate = learning_rate
        self._initial_reward_estimate = initial_reward_estimate
        self._minimum_visits = minimum_visits
        self._table = {}  # defaultdict is not efficient. It creates entries as soon as they are looked up.

    def predict(self, X: List[rldata.RLData]) -> List[float]:
        result = [float(self._table.get(Xi, {'Q': 0})['Q']
                   if self._table.get(Xi, {'N': 0})['N'] >= self._minimum_visits
                   else self._initial_reward_estimate)
                  for Xi in X]

        return result

    def learn(self, X: List[rldata.RLData], Y: List[float]) -> None:
        for i in range(len(X)):
            if X[i] not in self._table:
                self._table[X[i]] = {'Q': self._initial_reward_estimate, 'N': 0}

            self._table[X[i]]['Q'] += self._learning_rate.initial_lr * \
                (Y[i] - self._table[X[i]]['Q'])
            self._table[X[i]]['N'] += 1

    # def load(self, filename: str, path: Optional[Union[str, pathlib.Path]] = None) -> None:
        # temp = defaultdict(
        #     lambda: {'Q': self._initial_reward_estimate,
        #              'N': 0})
        # _path = pathlib.Path(path if path is not None else '')
        # with open(_path / f'{filename}.csv', 'r') as f:
        #     for k, v in csv.DictReader(f):
        #         temp[k] = v

    def save(self,
             filename: str,
             path: Optional[Union[str, pathlib.Path]] = None) -> Tuple[pathlib.Path, str]:
        _path = pathlib.Path(path if path is not None else '')
        with open(_path / f'{filename}.csv', 'w') as f:
            w = csv.writer(f)
            w.writerows(self._table.items())

        return _path, filename
