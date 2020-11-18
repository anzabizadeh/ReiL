import pathlib
from typing import List, Optional, Tuple, Union

from reil import rldata, learners


class Learner:
    def __init__(self,
                 learning_rate: learners.LearningRateScheduler) -> None:
        self._learning_rate = learning_rate

    @classmethod
    def from_pickle(cls, filename: str, path: Optional[Union[pathlib.Path, str]] = None):
        instance = cls(learning_rate=learners.ConstantLearningRate(0.0))
        instance.load(filename=filename, path=path)

        return instance

    def predict(self, X: List[rldata.RLData]) -> List[float]:
        raise NotImplementedError

    def learn(self, X: List[rldata.RLData], Y: List[float]) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def load(self, filename: str, path: Optional[Union[str, pathlib.Path]] = None) -> None:
        raise NotImplementedError

    def save(self,
             filename: str,
             path: Optional[Union[str, pathlib.Path]] = None) -> Tuple[pathlib.Path, str]:
        raise NotImplementedError
