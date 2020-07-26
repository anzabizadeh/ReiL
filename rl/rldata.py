import operator
from collections import defaultdict
from collections.abc import MutableMapping
from numbers import Number
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Union


class BaseRLData:
    __slots__ = ['_name', '_value', '_normalized', '_normalizer', '_lazy']

    def __init__(self, name, value, normalizer, lazy_evaluation):
        self._name = name
        self._value = value
        self._normalizer = normalizer
        self._lazy = lazy_evaluation
        self._normalize()

    def _normalize(self):
        if self._lazy:
            self._normalized = None
        else:
            self._normalized = self._normalizer(self)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v
        self._normalize()

    @property
    def normalizer(self):
        return self._normalizer

    @normalizer.setter
    def normalizer(self, func):
        if callable(func):
            self._normalizer = func
            self._normalize()
        else:
            raise TypeError('Callable argument expected.')

    @property
    def lazy(self):
        return self._lazy

    @lazy.setter
    def lazy(self, lazy):
        self._lazy = lazy
        self._normalize()

    @property
    def lazy_evaluation(self):
        return self._lazy

    @lazy_evaluation.setter
    def lazy_evaluation(self, lazy):
        self.lazy = lazy

    @property
    def normalized(self):
        if self._normalized is None:
            self._normalized = self._normalizer(self)

        return self._normalized

    def as_dict(self) -> dict:
        return {'name': self.name, 'value': self.value}

    def __str__(self) -> str:
        return f'{{{self.name}: {self.value}}}'


class CategoricalData(BaseRLData):
    __slots__ = ['_categories']

    def __init__(self, name, value, categories, normalizer=None, lazy_evaluation=False):
        super().__init__(name=name,
                         value=value,
                         normalizer=normalizer if normalizer is not None else self._default_normalizer,
                         lazy_evaluation=True)

        self._categories = categories
        self.lazy_evaluation = lazy_evaluation

    @staticmethod
    def _default_normalizer(x):
        if x.categories is None:  # categories are not defined
            return None

        if isinstance(x.value, type(x.categories[0])):
            return list(int(x_i == x.value) for x_i in x.categories)
        elif isinstance(x.value[0], type(x.categories[0])):
            return list(int(x_i == v) for v in x.value for x_i in x.categories)

    @property
    def categories(self):
        return self._categories

    @categories.setter
    def categories(self, cat):
        if cat is None:
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
    def value(self):
        return super().value

    @value.setter
    def value(self, v):
        if (isinstance(v, type(self.categories[0])) and v in self.categories) or \
                (isinstance(v[0], type(self.categories[0])) and all(v_i in self.categories for v_i in v)):
            self._value = v
            self._normalize()
        else:
            raise ValueError(f'{v} is not in categories: {self._categories}.')

    def as_dict(self) -> dict:
        temp_dict = super().as_dict()
        temp_dict.update({'categories': self.categories, 'categorical': True})
        return temp_dict

    def __str__(self):
        return f'{self._name}: {self._value} from {self._categories}'


class NumericalData(BaseRLData):
    __slots__ = ['_lower', '_upper']

    def __init__(self, name, value, lower, upper, normalizer=None, lazy_evaluation=False):
        super().__init__(name=name,
                         value=value,
                         normalizer=normalizer if normalizer is not None else self._default_normalizer,
                         lazy_evaluation=True)
        self.lower = lower
        self.upper = upper
        self.lazy_evaluation = lazy_evaluation

    @staticmethod
    def _default_normalizer(x):
        try:
            denominator = x.upper - x.lower
        except TypeError:  # upper or lower are not defined
            return None

        try:
            if isinstance(x.value, (list, tuple)):
                return list((v - x.lower) / denominator for v in x.value)
            else:
                return (x.value - x.lower) / denominator
        except ZeroDivisionError:
            return [1] * len*x.value if isinstance(x.value, (list, tuple)) else 1

    @staticmethod
    def _check_new_bound(value: Union[Number, Sequence],
                         new_bound: Number,
                         bound_type: str = 'lower'):
        if bound_type == 'lower':
            op = operator.le
            func = min
        else:
            op = operator.ge
            func = max

        try:
            return op(new_bound, func(value))
        except TypeError:
            return op(new_bound, value)

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, l):
        if l is None:
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
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, u):
        if u is None:
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
    def value(self):
        return super().value

    @value.setter
    def value(self, v):
        if isinstance(v, (list, tuple)):
            if not all(self.lower <= v_i <= self.upper for v_i in v):
                raise ValueError(
                    f'{v} is not in range: [{self.lower}, {self.upper}].')
        elif not (self.lower <= v <= self.upper):
            raise ValueError(
                f'{v} is not in range: [{self.lower}, {self.upper}].')
        else:
            self._value = v
            self._normalize()

    def as_dict(self) -> dict:
        temp_dict = super().as_dict()
        temp_dict.update(
            {'lower': self.lower, 'upper': self.upper, 'categorical': False})
        return temp_dict

    def __str__(self):
        return f'{self.name}: {self.value} of range [{self.lower}, {self.upper}]'


class RangedData:
    def __new__(cls, name, categorical, value, **kwargs):
        if categorical:
            cls = CategoricalData(name, value, kwargs.get('categories'))
        else:
            cls = NumericalData(name, value, kwargs.get(
                'lower'), kwargs.get('upper'))

        return cls


class RLData(MutableMapping):
    def __init__(self, value: dict = {},
                 categorical: Dict[Any, bool] = {},
                 lower: Dict[Any, Number] = {},
                 upper: Dict[Any, Number] = {},
                 categories: dict = {},
                 normalizer: Dict[Any, Callable[[RangedData], Number]] = {},
                 lazy_evaluation: Optional[bool] = False) -> None:
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
        self._data = {}
        for k, v in value.items():
            self._data[k] = RangedData(name=k,
                                       categorical=categorical[k],
                                       value=v,
                                       **{'categories': categories.get(k),
                                          'lower': lower.get(k),
                                          'upper': upper.get(k),
                                          'normalizer': normalizer.get(k), 'lazy_evaluation': lazy_evaluation})

    @classmethod
    def from_dict(cls, _dict: dict = {},
                  lazy_evaluation: Optional[bool] = False) -> None:
        '''
        Create an RLData instance.

        Attributes:
            _dict: a dictionary with mandatory keys 'name', 'categorical', 'value', and optional keys 'upper', 'lower', 'categories' and 'normalizer'
            lazy_evaluation: whether to store normalized values or compute on-demand (Default: False)
        '''
        return RLData(value={_dict['name']: _dict.get('value')},
                      lower={_dict['name']: _dict.get('lower')},
                      upper={_dict['name']: _dict.get('upper')},
                      categories={_dict['name']: _dict.get('categories')},
                      categorical={_dict['name']: _dict.get('categorical')},
                      normalizer={_dict['name']: _dict.get('normalizer')},
                      lazy_evaluation=lazy_evaluation)

    @property
    def value(self):
        '''Returns a dictionary of (name, RangedData) form.'''
        return dict((v.name, v.value) for v in self._data.values())

    @value.setter
    def value(self, v: dict):
        for key, val in v.items():
            self._data[key].value = val

    @property
    def lower(self):
        return dict((v.name, v.lower) for v in self._data.values() if hasattr(v, 'lower'))

    @lower.setter
    def lower(self, value: dict) -> None:
        for k, v in value.items():
            self._data[k].lower = v

    @property
    def upper(self):
        return dict((v.name, v.upper) for v in self._data.values() if hasattr(v, 'upper'))

    @upper.setter
    def upper(self, value: dict) -> None:
        for k, v in value.items():
            self._data[k].upper = v

    @property
    def categories(self):
        return dict((v.name, v.categories) for v in self._data.values() if hasattr(v, 'categories'))

    @categories.setter
    def categories(self, value: dict) -> None:
        for k, v in value.items():
            self._data[k].categories = v

    @property
    def is_categorical(self):
        return dict((v.name, True if isinstance(v, CategoricalData) else False) for v in self._data.values())

    @property
    def normalized(self):
        return dict((v.name, v.normalized) for v in self._data.values())

    @property
    def normalized_list(self):
        return list(v.normalized for v in self._data.values())

    def as_list(self) -> list:
        return self.value

    def __getitem__(self, k):
        return self._data.__getitem__(k)

    def __setitem__(self, k, v):
        if not isinstance(v, BaseRLData):
            raise TypeError(
                'Only variables of type BaseRLData and subclasses are acceptable.')

        return self._data.__setitem__(k, v)

    def __delitem__(self, key: Any) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return self._data.__len__()

    def has_key(self, k: Any) -> bool:
        return k in self._data

    def keys(self) -> Any:
        return self._data.keys()

    def values(self) -> "RLData":
        return self._data.values()

    def items(self) -> enumerate:
        return self._data.items()

    def __contains__(self, item: Any) -> bool:
        return item in self._data

    def __add__(self, other: "RLData") -> "RLData":
        if not isinstance(other, (RLData, BaseRLData)):
            raise TypeError(
                f'Concatenation of type RLData and {type(other)} not implemented!')

        if isinstance(other, BaseRLData):
            if other.name in self._data:
                raise ValueError(
                    'Cannot have items with same names. Use update() if you need to update an item.')
            else:
                new_dict = other.as_dict()
        else:
            for k in other:
                if k in self._data:
                    raise ValueError(
                        'Cannot have items with same names. Use update() if you need to update an item.')

            new_dict = {k: v.as_dict() for k, v in other.items()}

        temp = {k: v.as_dict() for k, v in self._data.items()}
        temp.update(new_dict)

        # reshape the data to have a format acceptable by RLData
        return_value = defaultdict(dict)
        generator_temp = ((k, {w: v})
                          for w, z in temp.items()
                          for k, v in z.items()
                          if k != 'name')

        for key, val in generator_temp:
            return_value[key].update(val)

        #  value: dict = {},
        #  categorical: Dict[Any, bool] = {},
        #  lower: Dict[Any, Number] = {},
        #  upper: Dict[Any, Number] = {},
        #  categories: dict = {},
        #  normalizer: Dict[Any, Callable[[RangedData], Number]] = {},
        #  lazy_evaluation: Optional[bool] = False

        return RLData(**return_value)

    def __repr__(self) -> str:
        return f'[{super().__repr__()} -> {self._data}'

    def __str__(self) -> str:
        return str(self._data)


if __name__ == "__main__":
    from timeit import timeit
    from rl.rldata import RLData

    def f1(): return RLData({'test A': 'a', 'test B': [10, 20]},
                            is_numerical={'test A': False, 'test B': True},
                            categories={'test A': ['a', 'b']},
                            lower={'test B': 0},
                            upper={'test B': 100}) + \
        RLData({'A': 'a', 'B': [10, 20]},
               is_numerical={'A': False, 'B': True},
               categories={'A': ['a', 'b']},
               lower={'B': 0},
               upper={'B': 100})

    def f2(): return RLData({'test A': 'a', 'test B': [10, 20]},
                            categorical={'test A': True, 'test B': False},
                            categories={'test A': ['a', 'b']},
                            lower={'test B': 0},
                            upper={'test B': 100}) + \
        RLData({'A': 'a', 'B': [10, 20]},
               categorical={'A': True, 'B': False},
               categories={'A': ['a', 'b']},
               lower={'B': 0},
               upper={'B': 100})

    # print(timeit(f1, number=100))
    print(timeit(f2, number=100))
    # print(a.values)
    # print(a.lower)
    # print(a.categories)
    # a.values = {'test A':'b'}
    # print(a + RLData({1: 100}, categorical={1: True}))
