from __future__ import annotations

import dataclasses
import itertools
import warnings
from dataclasses import field
from numbers import Number
from typing import (Any, Callable, Dict, Generic, Hashable, Iterable, Iterator,
                    List, Mapping, Optional, Sequence, Tuple, TypeVar, Union,
                    cast, overload)

T = TypeVar('T')
T_or_SeqT = Union[T, Sequence[T]]
N = TypeVar('N', int, float)
Normalized = Union[N, Tuple[N, ...], None]
Normalizer = Callable[..., Normalized]

@dataclasses.dataclass(frozen=True)
class BoundedData(Generic[T]):  # pylint: disable=unsubscriptable-object
    name: str
    value: Optional[T_or_SeqT] = None
    normalizer: Optional[Normalizer] = None
    lazy_evaluation: Optional[bool] = field(default=None, compare=False)
    categories: Optional[Tuple[T, ...]] = None
    lower: Optional[T] = None
    upper: Optional[T] = None
    is_numerical: Optional[bool] = field(default=None, init=False, repr=False, compare=False)
    _normalized: Normalized = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.value, Hashable):
            warnings.warn('Non-hashable object!')

        if self.categories is not None:
            if self.lower is not None or self.upper is not None:
                raise ValueError('Either categories or lower and upper should be specified.')
            object.__setattr__(self, 'is_numerical', False)
            self._categorical_validator(self.value, self.categories)
            if self.normalizer is None:
                object.__setattr__(self, 'normalizer', self._default_normalizer_categorical)
            if not self.lazy_evaluation:
                object.__setattr__(self, '_normalized',
                    self.normalizer(  # pylint: disable=not-callable
                        self.value, self.categories))
        else:
            object.__setattr__(self, 'is_numerical', True)
            self._numerical_validator(self.value, self.lower, self.upper)
            if self.normalizer is None:
                object.__setattr__(self, 'normalizer', self._default_normalizer_numerical)
            if not self.lazy_evaluation:
                object.__setattr__(self, '_normalized',
                    self.normalizer(  # pylint: disable=not-callable
                        self.value, self.lower, self.upper))

    @staticmethod
    def _default_normalizer_categorical(
        value: Optional[T_or_SeqT], categories: Optional[Tuple[T, ...]]) -> Normalized:
        if categories is None:
            return None

        if isinstance(value, type(categories[0])):
            return list(int(x_i == value)
                        for x_i in categories)

        return tuple(int(x_i == v)
                    for v in value
                    for x_i in categories)

    @staticmethod
    def _default_normalizer_numerical(
        value: Optional[T_or_SeqT], lower: Optional[T], upper: Optional[T]) -> Normalized:
        try:
            denominator = upper - lower  # type: ignore
        except TypeError:  # upper or lower are not defined
            return None

        sequence_check = isinstance(value, (list, tuple))
        try:
            if sequence_check:
                return tuple((v - lower) / denominator
                            for v in value)
            else:
                return (cast(T, value) - lower) / denominator
        except ZeroDivisionError:
            return [1] * len(cast(Sequence[T], value)) if sequence_check else 1

    @staticmethod
    def _numerical_validator(value: T_or_SeqT, lower: T, upper: T) -> None:
        if value is not None:
            is_sequence = isinstance(value, (list, tuple))

            if lower is not None:
                if ((not is_sequence and value < lower) or
                        (is_sequence and any(v < lower for v in value))):
                    raise ValueError(
                        f'value={value} is less than lower={lower}.')

            if upper is not None:
                if ((not is_sequence and value > upper) or
                        (is_sequence and any(v > upper for v in value))):
                    raise ValueError(
                        f'value={value} is greater than upper={upper}.')

    @staticmethod
    def _categorical_validator(value: T_or_SeqT, categories: Tuple[T, ...]) -> None:
        if value is not None:
            is_sequence = isinstance(value, (list, tuple))

            if categories is not None:
                if ((not is_sequence and value not in categories) or
                        (is_sequence and any(v not in categories for v in value))):
                    raise ValueError(
                        f'value={value} is in the categories={categories}.')

    @property
    def normalized(self) -> Normalized:
        if self._normalized is None:
            if self.is_numerical:
                object.__setattr__(self, '_normalized',
                    self.normalizer(self.value, self.lower, self.upper))  # pylint: disable=not-callable
            else:
                object.__setattr__(self, '_normalized',
                    self.normalizer(self.value, self.categories))  # pylint: disable=not-callable

        return self._normalized

    def as_dict(self) -> Dict[str, Any]:
        if self.is_numerical:
            return {'name': self.name, 'value': self.value,
                    'lower': self.lower, 'upper': self.upper,
                    'is_numerical': True}
        else:
            return {'name': self.name, 'value': self.value,
                    'categories': self.categories,
                    'is_numerical': False}


ReilDataInput = Union[Mapping[str, Any],
                      BoundedData[Any]]

class ReilData(Sequence[BoundedData[Any]]):
    def __init__(self,
        data: Union[ReilDataInput, Sequence[ReilDataInput], Iterator[ReilDataInput]],
        lazy_evaluation: Optional[bool] = None):
        '''
        Create a ReilData instance.

        Attributes
-----------
        data: data can be one or a sequence of either BoundedData instances or dicts that include 'name'. Other attributes are optional. If categories are not provided, the object is assumed numerical.

        lazy_evaluation: whether to store normalized values or compute on-demand.
        If not provided, class looks for 'lazy evaluation' in each object.
        If fails, True is assumed.
        '''
        temp = []
        _data = data if isinstance(data, (Sequence, Iterator)) else [data]
        for d in _data:
            if isinstance(d, BoundedData):
                temp.append(d)
            elif isinstance(d, dict):
                temp.append(BoundedData(
                    name=d['name'],
                    value=d.get('value'),
                    **{'categories': d.get('categories'),
                       'lower': d.get('lower'),
                       'upper': d.get('upper'),
                       'normalizer': d.get('normalizer'),
                       'lazy_evaluation': lazy_evaluation if lazy_evaluation is not None
                       else d.get('lazy_evaluation', True)}))

        self._data = tuple(temp)
        self._clear_temps()

    @classmethod
    def single_member(cls,
                   name: str,
                   value: Optional[T_or_SeqT] = None,
                   normalizer: Optional[Normalizer] = None,
                   lazy_evaluation: Optional[bool] = None,
                   categories: Optional[Tuple[T, ...]] = None,
                   lower: Optional[T] = None,
                   upper: Optional[T] = None) -> ReilData:
        '''
        Create a ReilData instance.

        Attributes
-----------
        name: name of the object
        value: value to store
        lower: minimum value if the object is numerical
        upper: maximum value if the object is numerical
        categories: a tuple of categories if the object is categorical
        normalizer: a function that normalizes value
        lazy_evaluation: whether to store normalized values or compute on-demand (Default: False)
        '''
        return cls(BoundedData(
            name=name,
            value=value,
            normalizer=normalizer,
            lazy_evaluation=lazy_evaluation,
            categories=categories,
            lower=lower,
            upper=upper))

    def _clear_temps(self):
        self._value = None
        self._lower = None
        self._upper = None
        self._categories = None
        self._is_numerical = None

    def index(self, value: Any, start: int = 0, stop: Optional[int] = None) -> int:
        _stop = stop if stop is not None else len(self._data)
        if isinstance(value, BoundedData):
            for i in range(start, _stop):
                if self._data[i] == value:
                    return i

            raise ValueError(f'{value} is not on the list.')

        elif isinstance(value, type(self._data[0].name)):
            for i in range(start, _stop):
                if self._data[i].name == value:
                    return i

            raise ValueError(f'{value} is not on the list.')

        else:
            raise ValueError(f'{value} is not on the list.')

    @property
    def value(self) -> Dict[str, Any]:
        '''Returns a dictionary of (name, RangedData) form.'''
        if self._value is None:
            self._value = dict((v.name, v.value) for v in self._data)

        return self._value

    # @value.setter
    # def value(self, v: Dict[str, Any]):
    #     self._value = None
    #     for key, val in v.items():
    #         self._data[self.index(key)].value = val

    @property
    def lower(self) -> Dict[str, Number]:
        if self._lower is None:
            self._lower = dict((v.name, v.lower) for v in self._data if hasattr(v, 'lower'))

        return self._lower

    # @lower.setter
    # def lower(self, value: Dict[str, Number]) -> None:
    #     self._lower = None
    #     for key, val in value.items():
    #         self._data[self.index(key)].lower = val

    @property
    def upper(self) -> Dict[str, Number]:
        if self._upper is None:
            self._upper = dict((v.name, v.upper) for v in self._data if hasattr(v, 'upper'))

        return self._upper

    # @upper.setter
    # def upper(self, value: Dict[str, Number]) -> None:
    #     self._upper = None
    #     for key, val in value.items():
    #         self._data[self.index(key)].upper = val

    @property
    def categories(self) -> Dict[str, Sequence[Any]]:
        if self._categories is None:
            self._categories = dict((v.name, v.categories) for v in self._data if hasattr(v, 'categories'))

        return self._categories

    # @categories.setter
    # def categories(self, value: Dict[str, Sequence[Any]]) -> None:
    #     self._categories = None
    #     for key, val in value.items():
    #         self._data[self.index(key)].categories = val

    @property
    def is_numerical(self) -> Dict[str, Sequence[bool]]:
        if self._is_numerical is None:
            self._is_numerical = dict((v.name, v.is_numerical)
                                      for v in self._data)

        return self._is_numerical

    @property
    def normalized(self) -> ReilData:
        return ReilData(BoundedData(name=v.name, value=v.normalized,
                                    lower=0, upper=1, lazy_evaluation=True)
                        for v in self._data)

    def flatten(self) -> List[Any]:
        def make_iterable(x: Any) -> Iterable[Any]:
            return x if isinstance(x, Iterable) else [x]

        return list(itertools.chain(*[make_iterable(sublist)
            for sublist in self.value.values()]))

    def split(self) -> Union[ReilData, List[ReilData]]:  #, name_suffix: Optional[str] = None):
        if len(self) == 1:
            d = self._data[0]
            if not isinstance(d.value, (list, tuple)):
                splitted_list = ReilData(self._data)
            else:
                splitted_list = [
                    ReilData.single_member(
                        name=d.name,
                        value=v,
                        lower=d.lower,
                        upper=d.upper,
                        categories=d.categories,
                        normalizer=d.normalizer,
                        lazy_evaluation=d.lazy_evaluation)
                    for v in d.value]

        else:
            splitted_list = [ReilData.single_member(
                    name=d.name,
                    value=d.value,
                    upper=d.upper,
                    lower=d.lower,
                    categories=d.categories,
                    normalizer=d.normalizer,
                    lazy_evaluation=d.lazy_evaluation)
                    for d in self._data]

        return splitted_list

    @overload
    def __getitem__(self, i: int) -> BoundedData:
        ...

    @overload
    def __getitem__(self, s: slice) -> Tuple[BoundedData, ...]:  # pylint: disable=function-redefined
        ...

    def __getitem__(self, x):  # pylint: disable=function-redefined
        return self._data.__getitem__(x)

    # def __setitem__(self, i: Union[int, slice], o: Union[Any, Iterable[Any]]):
    #     # TODO: I should somehow check the iterable to make sure it has proper data,
    #     # but currently I have no idea how!
    #     if not isinstance(o, (BoundedData, Iterable)):
    #         raise TypeError(
    #             'Only variables of type BoundedData and subclasses are acceptable.')

    #     self._clear_temps()
    #     return self._data.__setitem__(i, o)

    # def __delitem__(self, i: Union[int, slice]) -> None:
    #     self._data.__delitem__(i)

    # def insert(self, index: int, value: Any) -> None:
    #     if not isinstance(value, BoundedData):
    #         raise TypeError(
    #             'Only variables of type BoundedData and subclasses are acceptable.')

    #     self._clear_temps()
    #     self._data.insert(index, value)

    def __len__(self) -> int:
        return self._data.__len__()

    def __hash__(self) -> int:
        return hash(tuple(self._data))

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and (self._data == other._data)

    # def extend(self, values: R) -> None:
    #     if isinstance(values, ReilData):
    #         for v in values:
    #             self._data.append(v)
    #     else:
    #         for v in values:
    #             self._data.extend(ReilData(v))

    # def append(self, value: Union[Dict[str, Any], ReilDataClass]) -> None:
    #     if isinstance(value, ReilData):
    #         self._data.extend(value)
    #     else:
    #         self._data.extend(ReilData(value))

    def __add__(self, other: Union[BoundedData[Any], ReilData]) -> ReilData:
        if isinstance(other, BoundedData):
            if other.name in self._data:
                raise ValueError(
                    'Cannot have items with same names. Use update() if you need to update an item.')
            new_data = [other]
        elif isinstance(other, ReilData):
            for k in other:
                if k in self._data:
                    raise ValueError(
                        'Cannot have items with same names. Use update() if you need to update an item.')
            new_data = list(other._data)
        else:
            raise TypeError(
                f'Concatenation of type ReilData and {type(other)} not implemented!')

                # new_dict = other.as_dict()

                # new_dict = {v.name: v.as_dict() for v in other}

        return ReilData(list(self._data) + new_data)

    def __neg__(self) -> ReilData:
        temp = {v.name: v.as_dict()
                for v in self._data}
        temp['value'] = -temp['value']

        return ReilData(temp)

    def __repr__(self) -> str:
        return f'[{super().__repr__()} -> {self._data}]'

    def __str__(self) -> str:
        return f"[{', '.join((d.__str__() for d in self._data))}]"


    # # Mixin methods
    # def append(self, value: _T) -> None: ...
    # def clear(self) -> None: ...
    # def extend(self, values: Iterable[_T]) -> None: ...
    # def reverse(self) -> None: ...
    # def pop(self, index: int = ...) -> _T: ...
    # def remove(self, value: _T) -> None: ...
    # def __iadd__(self, x: Iterable[_T]) -> MutableSequence[_T]: ...

    # # Sequence Mixin methods
    # def index(self, value: Any, start: int = ..., stop: int = ...) -> int: ...
    # def count(self, value: Any) -> int: ...
    # def __contains__(self, x: object) -> bool: ...
    # def __iter__(self) -> Iterator[_T_co]: ...
    # def __reversed__(self) -> Iterator[_T_co]: ...


if __name__ == "__main__":
    from timeit import timeit

    def f1():
        t1 = ReilData.single_member(name='test A', value=['a', 'b'],
                    categories=('a', 'b'))
        return t1

    def f2():
        t2 = ReilData(({'name': x, 'value': x, 'categories': list('abcdefghijklmnopqrstuvwxyz')}
            for x in 'abcdefghijklmnopqrstuvwxyz'), lazy_evaluation=True)
        return t2.normalized.flatten()

    def f3():
        t2 = ReilData([{'name': 'A', 'value': 'a', 'categories': ['a', 'b']},
            {'name': 'B', 'value': [10, 20], 'lower': 0, 'upper': 100}], lazy_evaluation=True)
        return t2


    # t1 = f1()
    # t2 = ReilData(t1)
    print(f1() + f3())
    # test = f1().split()
    # print(test)
    # print(timeit(f1, number=1000))
    # print(timeit(f2, number=100))
    # print(timeit(f3, number=1000))
    # print(f2().normalized.as_list())
    # print(a.values)
    # print(a.lower)
    # print(a.categories)
    # a.values = {'test A':'b'}
