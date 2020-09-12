from __future__ import annotations

import itertools
import operator
from collections.abc import MutableSequence
from numbers import Number
from typing import (Any, Callable, Dict, Generator, Iterable, List,
                    Mapping, Optional, Sequence, Tuple, TypeVar, Union)


T = TypeVar('T')  # Type for value
C = TypeVar('C')  # Type for categories
Categories = Sequence[C]

N = Union[Number, Sequence[Number]]  # Type for numerical value

RLDataClass = TypeVar('RLDataClass')
Normalized = Union[Number, Sequence[Number], None]
Normalizer = Callable[[RLDataClass], Normalized]

R = Sequence[Union[Dict[str, Any], RLDataClass]]

class BaseRLData:
    __slots__ = ['name', '_value', '_categorical',
                 '_normalized', '_normalizer', '_lazy_evaluation']

    def __init__(self,
                 name: str,
                 value: T,
                 categorical: bool,
                 normalizer: Normalizer,
                 lazy_evaluation: bool):
        self.name = name
        self._value = value
        self._categorical = categorical
        self._normalizer = normalizer
        self._lazy_evaluation = lazy_evaluation
        self._normalize()

    def _normalize(self) -> None:
        if self._lazy_evaluation:
            self._normalized = None
        else:
            self._normalized = self._normalizer(self)

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, v: T):
        self._value = v
        self._normalize()

    @property
    def categorical(self) -> bool:
        return self._categorical

    @property
    def normalizer(self) -> Normalizer:
        return self._normalizer

    @normalizer.setter
    def normalizer(self, func: Normalizer):
        if callable(func):
            self._normalizer = func
            self._normalize()
        else:
            raise TypeError('Callable argument expected.')

    @property
    def lazy_evaluation(self) -> bool:
        return self._lazy_evaluation

    @lazy_evaluation.setter
    def lazy_evaluation(self, lazy_evaluation: bool):
        self._lazy_evaluation = lazy_evaluation
        self._normalize()

    @property
    def normalized(self) -> Normalized:
        if self._normalized is None:
            self._normalized = self._normalizer(self)

        return self._normalized

    def as_dict(self) -> Dict[str, Any]:
        return {'name': self.name, 'value': self.value}

    def __str__(self) -> str:
        return f'{{{self.name}: {self.value}}}'


class CategoricalData(BaseRLData):
    __slots__ = ['_categories']

    def __init__(self,
                 name: str,
                 value: T,
                 categories: Optional[Categories] = None,
                 normalizer: Optional[Normalizer] = None,
                 lazy_evaluation: bool = False):

        super().__init__(name=name,
                         value=value,
                         categorical=True,
                         normalizer=normalizer if normalizer is not None else self._default_normalizer,
                         lazy_evaluation=True)  # Do not evaluate before assigning categories.

        self._categories = categories
        self.lazy_evaluation = lazy_evaluation

    @staticmethod
    def _default_normalizer(x: CategoricalData) -> Normalized:
        if x.categories is None:  # categories are not defined
            return None

        if isinstance(x.value, type(x.categories[0])):
            return list(int(x_i == x.value) for x_i in x.categories)  # type: ignore
        elif isinstance(x.value[0], type(x.categories[0])):
            return list(int(x_i == v) for v in x.value for x_i in x.categories)  # type: ignore

    @property
    def categories(self) -> Union[Categories, None]:
        return self._categories

    @categories.setter
    def categories(self, cat: Union[Categories, None]):
        if cat is None or self.value in (None, (), []):
            self._categories = cat
            self._normalized = None
        elif not isinstance(cat, (list, tuple)):
            raise TypeError(
                f'A sequence (list or tuple) was expected. Received a {type(cat)}.\n{self.__repr__()}')
        elif (isinstance(self.value, type(cat[0])) and self.value in cat) or \
             (isinstance(self.value[0], type(cat[0])) and all(v in cat for v in self.value)):
            self._categories = cat
            self._normalize()
        else:
            raise ValueError(
                f'Categories list {cat} does not include the current value: {self.value}.\n{self.__repr__()}')

    @property
    def value(self) -> T:
        return super().value

    @value.setter
    def value(self, v: T):
        if (isinstance(v, type(self.categories[0])) and v in self.categories) or \
                (isinstance(v[0], type(self.categories[0])) and all(v_i in self.categories for v_i in v)):
            super(CategoricalData, self.__class__).value.fset(self, v)
        else:
            raise ValueError(f'{v} is not in categories: {self._categories}.')

    def as_dict(self) -> Dict[str, Any]:
        temp_dict = super().as_dict()
        temp_dict.update({'categories': self.categories, 'categorical': True})
        return temp_dict

    def __str__(self):
        return f'{self.name}: {self._value} from {self._categories}'


class NumericalData(BaseRLData):
    __slots__ = ['_lower', '_upper']

    def __init__(self,
                 name: str,
                 value: N,
                 lower: Optional[Number] = None,
                 upper: Optional[Number] = None,
                 normalizer: Optional[Normalizer] = None,
                 lazy_evaluation: bool = False):

        super().__init__(name=name,
                         value=value,
                         categorical=False,
                         normalizer=normalizer if normalizer is not None else self._default_normalizer,
                         lazy_evaluation=True)  # Do not evaluate before assigning lower and upper.
        self.lower = lower
        self.upper = upper
        self.lazy_evaluation = lazy_evaluation

    @staticmethod
    def _default_normalizer(x: NumericalData) -> Normalized:
        try:
            denominator = x.upper - x.lower  # type: ignore
        except TypeError:  # upper or lower are not defined
            return None

        try:
            if isinstance(x.value, (list, tuple)):
                return list((v - x.lower) / denominator for v in x.value)
            else:
                return (x.value - x.lower) / denominator  # type: ignore
        except ZeroDivisionError:
            return [1] * len*x.value if isinstance(x.value, (list, tuple)) else 1  # type: ignore

    @staticmethod
    def _check_new_bound(value: N,
                         new_bound: Number,
                         bound_type: str = 'lower'):
        if bound_type == 'lower':
            op = operator.le
            func = min
        else:
            op = operator.ge
            func = max

        try:
            return op(new_bound, func(value))  # type: ignore
        except TypeError:
            return op(new_bound, value)

    @property
    def lower(self) -> Union[Number, None]:
        return self._lower

    @lower.setter
    def lower(self, l: Union[Number, None]):
        if l is None or self.value in (None, (), []):
            self._lower = l
            self._normalized = None
        elif not isinstance(l, Number):
            raise TypeError(
                f'A numerical value expected. Received a {type(l)}.\n{self.__repr__()}')
        elif self._check_new_bound(self.value, l, 'lower'):
            self._lower = l
            self._normalize()
        else:
            raise ValueError(
                f'Lower bound {l} is greater than current value: {self.value}.\n{self.__repr__()}')

    @property
    def upper(self) -> Union[Number, None]:
        return self._upper

    @upper.setter
    def upper(self, u: Union[Number, None]):
        if u is None or self.value in (None, (), []):
            self._upper = u
            self._normalized = None
        elif not isinstance(u, Number):
            raise TypeError(
                f'A numerical value expected. Received a {type(u)}.\n{self.__repr__()}')
        elif self._check_new_bound(self.value, u, 'upper'):
            self._upper = u
            self._normalize()
        else:
            raise ValueError(
                f'Lower bound {u} is greater than current value: {self.value}.\n{self.__repr__()}')

    @property
    def value(self) -> N:
        return super().value

    @value.setter
    def value(self, v: N):
        if isinstance(v, (list, tuple)):
            if not all(self.lower <= v_i <= self.upper for v_i in v):
                raise ValueError(
                    f'{v} is not in range: [{self.lower}, {self.upper}].')
        elif not (self.lower <= v <= self.upper):  # type: ignore
            raise ValueError(
                f'{v} is not in range: [{self.lower}, {self.upper}].')
        else:
            super(NumericalData, self.__class__).value.fset(self, v)


    def as_dict(self) -> Dict[str, Any]:
        temp_dict = super().as_dict()
        temp_dict.update(
            {'lower': self.lower, 'upper': self.upper, 'categorical': False})
        return temp_dict

    def __str__(self):
        return f'{self.name}: {self.value} of range [{self.lower}, {self.upper}]'


class RangedData:
    def __new__(cls,
                name: str,
                categorical: bool,
                value: Any,
                **kwargs) -> Union[CategoricalData, NumericalData]:
        if categorical:
            obj = CategoricalData(name=name, value=value,
                                  categories=kwargs.get('categories'),
                                  normalizer=kwargs.get('normalizer'),
                                  lazy_evaluation=kwargs.get('lazy_evaluation', True))
        else:
            obj = NumericalData(name=name, value=value,
                                lower=kwargs.get('lower'),
                                upper=kwargs.get('upper'),
                                normalizer=kwargs.get('normalizer'),
                                lazy_evaluation=kwargs.get('lazy_evaluation', True))

        return obj


class RLData(MutableSequence):
    def __init__(self,
        data: Union[Mapping[str, Any], R, Generator[R, Any, Any]],
        lazy_evaluation: Optional[bool] = None) -> None:
        '''
        Create an RLData instance.

        Attributes:
            data: data is either a sequence of dict-like objects that include 'name'
                  or a mapping whose keys are names of variables.
                  Other attributes are optional. If the class cannot find 'categorical',
                  it attemps to find 'categories'. If fails, the object is assumed numerical.
            lazy_evaluation: whether to store normalized values or compute on-demand.
                             If not provided, class looks for 'lazy evaluation' in
                             each object. If fails, True is assumed.
        '''
        self._data = []
        self._clear_temps()

        def _from_tuple(d: Tuple[str, Mapping[str, Any]]) -> RangedData:
            if isinstance(d, BaseRLData):
                return d

            return RangedData(
                    name=d[0],
                    value=d[1].get('value'),
                    # if categorical is not available, check if categories is available.
                    categorical=d[1].get('categorical', d[1].get(
                        'categories') is not None),
                    **{'categories': d[1].get('categories'),
                        'lower': d[1].get('lower'),
                        'upper': d[1].get('upper'),
                        'normalizer': d[1].get('normalizer'),
                        'lazy_evaluation': lazy_evaluation if lazy_evaluation is not None
                            else d[1].get('lazy_evaluation', True)})

        def _from_dict(v: Union[BaseRLData, Mapping[str, Any]]) -> RangedData:
            if isinstance(v, BaseRLData):
                return v

            return RangedData(
                    name=v['name'],
                    value=v.get('value'),
                    # if categorical is not available, check if categories is available.
                    categorical=v.get('categorical', v.get(
                        'categories') is not None),
                    **{'categories': v.get('categories'),
                        'lower': v.get('lower'),
                        'upper': v.get('upper'),
                        'normalizer': v.get('normalizer'),
                        'lazy_evaluation': lazy_evaluation if lazy_evaluation is not None
                            else v.get('lazy_evaluation', True)})

        if isinstance(data, Sequence):  # a sequence of dict, RLData, BaseRLData, etc.
            self._data.extend(_from_dict(v) for v in data)  # each item in the sequence is dict-like

        elif isinstance(data, Mapping):  # dict, RLData, BaseRLData, etc.
            if isinstance(list(data.values())[0], dict):
                self._data.extend(_from_tuple(d) for d in data.items())
            else:
                self._data.append(_from_dict(data))

        elif isinstance(data, Generator):  # type: ignore
            for v in data:
                if isinstance(v, Mapping):
                    self._data.append(_from_dict(v))  # type: ignore
                elif isinstance(v, tuple):
                    self._data.append(_from_tuple(v))  # type: ignore
                else:
                    raise TypeError('Type is not recognized!')
        else:
            raise TypeError('Type is not recognized!')

    @classmethod
    def from_sparse_data(cls,
                        value: Dict[str, T] = {},
                        categorical: Dict[str, bool] = {},
                        lower: Dict[str, Union[Number, None]] = {},
                        upper: Dict[str, Union[Number, None]] = {},
                        categories: Dict[str, Union[Categories, None]] = {},
                        normalizer: Dict[str, Union[Normalizer, None]] = {},
                        lazy_evaluation: Optional[bool] = None) -> RLData:
        '''
        Create an RLData instance.

        Attributes:
            value: value to store as a list or a dictionary
            lower: minimum values for numerical components
            upper: maximum values for numerical components
            categories: set of categories for categorical components
            categorical: whether each component is numerical (True) or categorical (False)
            normalizer: a function that normalizes the respective component
            lazy_evaluation: whether to store normalized values or compute on-demand (Default: False)
        '''

        temp = {}
        for k, v in value.items():
            temp[k] = dict(name=k,
                           categorical=categorical[k],
                           value=v,
                           **{'categories': categories.get(k),
                               'lower': lower.get(k),
                               'upper': upper.get(k),
                               'normalizer': normalizer.get(k), 'lazy_evaluation': lazy_evaluation})

        return cls(temp)

    def _clear_temps(self):
        self._value = None
        self._lower = None
        self._upper = None
        self._categories = None
        self._categorical = None

    def index(self, value: Any, start: int = 0, stop: Optional[int] = None) -> int:
        _stop = stop if stop is not None else len(self._data)
        if isinstance(value, BaseRLData):
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

    @value.setter
    def value(self, v: Dict[str, Any]):
        self._value = None
        for key, val in v.items():
            self._data[self.index(key)].value = val

    @property
    def lower(self) -> Dict[str, Number]:
        if self._lower is None:
            self._lower = dict((v.name, v.lower) for v in self._data if hasattr(v, 'lower'))

        return self._lower

    @lower.setter
    def lower(self, value: Dict[str, Number]) -> None:
        self._lower = None
        for key, val in value.items():
            self._data[self.index(key)].lower = val

    @property
    def upper(self) -> Dict[str, Number]:
        if self._upper is None:
            self._upper = dict((v.name, v.upper) for v in self._data if hasattr(v, 'upper'))

        return self._upper

    @upper.setter
    def upper(self, value: Dict[str, Number]) -> None:
        self._upper = None
        for key, val in value.items():
            self._data[self.index(key)].upper = val

    @property
    def categories(self) -> Dict[str, Sequence[Any]]:
        if self._categories is None:
            self._categories = dict((v.name, v.categories) for v in self._data if hasattr(v, 'categories'))

        return self._categories

    @categories.setter
    def categories(self, value: Dict[str, Sequence[Any]]) -> None:
        self._categories = None
        for key, val in value.items():
            self._data[self.index(key)].categories = val

    @property
    def categorical(self) -> Dict[str, Sequence[bool]]:
        if self._categorical is None:
            self._categorical = dict((v.name, v.categorical) for v in self._data)

        return self._categorical

    @property
    def normalized(self) -> RLData:
        return RLData({'name': v.name, 'categorical': v.categorical,
                       'value': v.normalized, 'lower': 0, 'upper': 1, 'lazy_evaluation': True} for v in self._data)

    def flatten(self) -> List[Any]:
        def make_iterable(x: Any) -> Iterable[Any]:
            return x if isinstance(x, Iterable) else [x]

        return list(itertools.chain(*[make_iterable(sublist)
            for sublist in self.value.values()]))

    def split(self) -> Union[RLData, List[RLData]]:  #, name_suffix: Optional[str] = None):
        if len(self) == 1:
            d = self._data[0]
            if isinstance(d.value, Sequence):
                val: Sequence[Any] = d.value
                name = d.name
                categorical = d.categorical
                lower = d.lower if not d.categorical else None
                upper = d.upper if not d.categorical else None
                categories = d.categories if d.categorical else None
                normalizer = d.normalizer
                lazy_evaluation = d.lazy_evaluation

                splitted_list = [RLData({
                        'name': name,
                        'value': v,
                        'categorical': categorical,
                        'lower': lower,
                        'upper': upper,
                        'categories': categories,
                        'normalizer': normalizer,
                        'lazy_evaluation': lazy_evaluation})
                        for v in val]

            else:
                splitted_list = RLData(self._data)

        else:
            splitted_list = [RLData({
                    'name': d.name,
                    'value': d.value,
                    'categorical': d.categorical,
                    'upper': d.upper if not d.categorical else None,
                    'lower': d.lower if not d.categorical else None,
                    'categories': d.categories if d.categorical else None,
                    'normalizer': d.normalizer,
                    'lazy_evaluation': d.lazy_evaluation})
                    for d in self._data]

        return splitted_list

    def __getitem__(self, i: Union[int, slice]):
        return self._data.__getitem__(i)

    def __setitem__(self, i: Union[int, slice], o: Union[Any, Iterable[Any]]):
        # TODO: I should somehow check the iterable to make sure it has proper data,
        # but currently I have no idea how!
        if not isinstance(o, (BaseRLData, Iterable)):
            raise TypeError(
                'Only variables of type BaseRLData and subclasses are acceptable.')

        self._clear_temps()
        return self._data.__setitem__(i, o)

    def __delitem__(self, i: Union[int, slice]) -> None:
        self._data.__delitem__(i)

    def insert(self, index: int, value: Any) -> None:
        if not isinstance(value, BaseRLData):
            raise TypeError(
                'Only variables of type BaseRLData and subclasses are acceptable.')

        self._clear_temps()
        self._data.insert(index, value)

    def __len__(self) -> int:
        return self._data.__len__()

    def extend(self, values: R) -> None:
        for v in values:
            self._data.append(RLData(v))

    def append(self, value: Union[Dict[str, Any], RLDataClass]) -> None:
        self._data.append(RLData(value))

    def __add__(self, other: RLData) -> RLData:
        if isinstance(other, BaseRLData):
            if other.name in self._data:
                raise ValueError(
                    'Cannot have items with same names. Use update() if you need to update an item.')
        elif isinstance(other, RLData):
            for k in other:
                if k in self._data:
                    raise ValueError(
                        'Cannot have items with same names. Use update() if you need to update an item.')
        else:
            raise TypeError(
                f'Concatenation of type RLData and {type(other)} not implemented!')

                # new_dict = other.as_dict()

                # new_dict = {v.name: v.as_dict() for v in other}

        temp = {v.name: v.as_dict() for v in self._data}
        # temp.update(new_dict)

        return RLData(temp)

    def __neg__(self) -> RLData:
        temp = {v.name: v.as_dict() for v in self._data}
        temp['value'] = -temp['value']

        return RLData(temp)

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
        t1 = RLData.from_sparse_data({'test A': ['a', 'b']},
                    categorical={'test A': True},
                    categories={'test A': ['a', 'b']},
                    # lower={'test B': 0},
                    # upper={'test B': 100}
                    )
        return t1

    def f2():
        t2 = RLData(({'name': x, 'value': x, 'categories': list('abcdefghijklmnopqrstuvwxyz')}
            for x in 'abcdefghijklmnopqrstuvwxyz'), lazy_evaluation=True)
        return t2.normalized.flatten()

    def f3():
        t2 = RLData([{'name': 'A', 'value': 'a', 'categories': ['a', 'b']},
            {'name': 'B', 'value': [10, 20], 'lower': 0, 'upper': 100}], lazy_evaluation=True)
        return t2


    t1 = f1()
    t2 = RLData(t1)
    t1[0].value = ['a'] 
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
    # print(a + RLData({1: 100}, categorical={1: True}))
