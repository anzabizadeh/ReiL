import dataclasses
from typing import Any, Dict, Generic, Tuple, TypeVar

from reil import learners
from reil.datatypes import ReilData

T = TypeVar('T')

# TODO: implement `load` and `save`

@dataclasses.dataclass
class TableEntry(Generic[T]):
    value: T
    N: int = 0


class LookupTable(Dict[Any, TableEntry[T]]):
    '''
    A simple lookup table based on `dict` that checks for the type of inputs
    to be `TableEntry`.
    '''
    def __setitem__(self, key, item):
        if isinstance(item, TableEntry):
            super().__setitem__(key, item)
        else:
            raise TypeError('item should be of type TableEntry.')


class QLookupTable(learners.Learner[float]):
    '''
    A Q-learning lookup table class.

    This class stores input data and the corresponding output as a dictionay.

    ### Methods
    predict: predicts `y` for a given input list `X`.

    learn: learns using training set `X` and `y`.

    load: loads a Dense instance.

    save: saves the learner.
    '''
    def __init__(self,
                 learning_rate: learners.LearningRateScheduler,
                 initial_estimate: float = 0.0,
                 minimum_visits: int = 0) -> None:
        '''
        Initialize the lookup table.

        ### Arguments
        learning_rate: a `LearningRateScheduler` object that determines the 
        learning rate of this learner.

        initial_estimate: the value to be returned if not enough observations
        have been collected for a given x.

        minimum_visits: for a given x, if it was learned for more than 
        minimum_visits, the computed estimate is returned. For any less visited
        x, initial_estimate will be returned.
        '''
        self._learning_rate = learning_rate
        self._initial_estimate = initial_estimate
        self._minimum_visits = minimum_visits
        # defaultdict is not efficient.
        # It creates entries as soon as they are looked up.
        self._table: LookupTable[float] = LookupTable()

    def predict(self, X: Tuple[ReilData, ...]) -> Tuple[float, ...]:
        '''
        predicts `y` for a given input list `X`.

        ### Arguments
        X: a list of `ReilData` as inputs to the prediction model.
        '''
        dummy = TableEntry(self._initial_estimate)
        result = tuple(float(self._table.get(Xi, dummy).value
                        if self._table.get(Xi, dummy).N >= self._minimum_visits
                        else self._initial_estimate)
                  for Xi in X)

        return result

    def learn(self, X: Tuple[ReilData, ...], Y: Tuple[float, ...]) -> None:
        '''
        Learns using training set `X` and `Y`.

        ### Arguments
        X: a list of `ReilData` as inputs to the learning model.

        Y: a list of float labels for the learning model.
        '''
        for i, Xi in enumerate(X):
            if Xi not in self._table:
                self._table[Xi] = TableEntry(self._initial_estimate)

            self._table[Xi].value += self._learning_rate.initial_lr * \
                (Y[i] - self._table[Xi].value)
            self._table[Xi].N += 1

    # def load(self, filename: str, path: Optional[Union[str, pathlib.Path]] = None) -> None:
        # temp = defaultdict(
        #     lambda: {'value': self._initial_estimate,
        #              'N': 0})
        # _path = pathlib.Path(path if path is not None else '')
        # with open(_path / f'{filename}.csv', 'r') as f:
        #     for k, v in csv.DictReader(f):
        #         temp[k] = v

    # def save(self,
    #          filename: str,
    #          path: Optional[Union[str, pathlib.Path]] = None) -> Tuple[pathlib.Path, str]:
    #     _path = pathlib.Path(path if path is not None else '')
    #     with open(_path / f'{filename}.csv', 'w') as f:
    #         w = csv.writer(f)
    #         w.writerows(self._table.items())

    #     return _path, filename
