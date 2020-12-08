import pathlib
from typing import Any, Generic, Optional, Tuple, TypeVar, Union

from reil import learners, reilbase
from reil.datatypes import ReilData

LabelType = TypeVar('LabelType')


class Learner(reilbase.ReilBase, Generic[LabelType]):
    '''
    The base class for all `Learners`.

    Methods
-----------
    from_pickle: loads a learner from a file.

    predict: predicts `y` for a given input list `X`.

    learn: learns using training set `X` and `y`.
    '''
    def __init__(self,
                 learning_rate: learners.LearningRateScheduler,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._learning_rate = learning_rate

    @classmethod
    def from_pickle(cls, filename: str, path: Optional[Union[pathlib.Path, str]] = None):
        instance = cls(learning_rate=learners.ConstantLearningRate(0.0))
        instance.load(filename=filename, path=path)

        return instance

    def predict(self, X: Tuple[ReilData, ...]) -> Tuple[LabelType, ...]:
        '''
        predicts `y` for a given input list `X`.

        Arguments
-----------
        X: a list of `ReilData` as inputs to the prediction model.
        '''
        raise NotImplementedError

    def learn(self, X: Tuple[ReilData, ...], Y: Tuple[LabelType, ...]) -> None:
        '''
        Learns using training set `X` and `Y`.

        Arguments
-----------
        X: a list of `ReilData` as inputs to the learning model.

        Y: a list of float labels for the learning model.
        '''
        raise NotImplementedError
