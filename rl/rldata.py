# -*- coding: utf-8 -*-
'''
RLData class
==============

A data type used for state and action variables.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

# from __future__ import annotations
from numbers import Number
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Union)


class RLData(dict):
    def __new__(cls, value=[0], **kwargs):
        obj = super().__new__(cls)

        obj._lazy = True
        obj._value = {}
        obj._is_numerical = {}
        obj._normalizer = {}
        obj._categories = {}
        obj._lower = {}
        obj._upper = {}
        obj._normal_form = {}

        return obj

    def __init__(self, value: Optional[Union[Sequence, dict]] = [0],
                 lower: Optional[Union[Sequence, dict]] = None,
                 upper: Optional[Union[Sequence, dict]] = None,
                 categories: Optional[Union[Sequence, dict]] = None,
                 is_numerical: Optional[Union[Sequence[bool], Dict[Any, bool]]] = None,
                 normalizer: Optional[Union[Callable[[dict], Number], Callable[[dict], List[Number]],
                                            Dict[Any, Callable[[dict], Number]], Dict[Any, Callable[[dict], List[Number]]]]] = None,
                 lazy_evaluation: Optional[bool] = False) -> None:
        '''
        Create an RLData instance.

        Attributes:
            value: value to store as a list or a dictionary
            lower: minimum values for numerical components
            upper: maximum values for numerical components
            categories: set of categories for categorical components
            is_numerical: whether each component is numerical (True) or categorical (False)
            normalizer: a function that normalizes the respective component
            lazy_evaluation: whether to store normalized values or compute on-demand (Default: False)
        '''
        self._value = {}

        self.value = value

        if is_numerical is not None:
            self.is_numerical = is_numerical

        if lower is not None:
            self.lower = lower

        if upper is not None:
            self.upper = upper

        if categories is not None:
            self.categories = categories

        if normalizer is not None:
            self.normalizer = normalizer

        self._normalizer_lambda_func = lambda x: (x['value']-x['lower'])/(
            x['upper']-x['lower']) if x['is_numerical'] else list(int(x_i == x['value']) for x_i in x['categories'])

        if lazy_evaluation is not None:
            self._lazy = lazy_evaluation

        if not self._lazy:
            self._normalized = self._normalize()

    @property
    def value(self) -> "RLData":
        return self._value

    @value.setter
    def value(self, v: Union[Sequence, dict]) -> None:
        '''
        Sets the value of the RLData instance.

        Attributes:
            v: value to store as a list or a dictionary. If a list is provided,
            it is stored as `value`. To have named values, use dictionary.
        '''
        try:
            for i, v_i in v.items():
                self.__setitem__(i, v_i)
        except AttributeError:
            self.__setitem__(None, v)  # .copy())

    @property
    def lower(self) -> Union[list, dict]:
        '''
        Return the lower bound
        '''
        return self._lower

    @lower.setter
    def lower(self, value: Union[Sequence, dict]) -> None:
        '''
        Set the lower bound

        Raises ValueError if the provided lower bound is greater than the currently stored value. 
        '''
        def check_min(v, new_min):
            try:
                minimum = min(v)
            except TypeError:
                minimum = v

            if minimum < new_min:
                raise ValueError(
                    f'The provided lower bound ({new_min}) is greater than current smallest number ({minimum}).\n{self.__repr__()}')

            return new_min

        try:
            for i, val in value.items():
                if self._is_numerical[i]:
                    self._lower[i] = check_min(self._value[i], val)

        except AttributeError:
            if self._is_numerical:
                self._lower = check_min(self._value, value)

        if not self._lazy:
            self._normalized = self._normalize()

    @property
    def upper(self) -> Union[list, dict]:
        '''
        Return the upper bound
        '''
        return self._upper

    @upper.setter
    def upper(self, value: Union[Sequence, dict]) -> None:
        '''
        Set the upper bound

        Raises ValueError if the provided upper bound is less than the currently stored value. 
        '''
        def check_max(v, new_max):
            try:
                maximum = max(v)
            except TypeError:
                maximum = v

            if maximum > new_max:
                raise ValueError(
                    f'The provided upper bound ({new_max}) is less than current greatest number ({maximum}).\n{self.__repr__()}')

            return new_max

        try:
            for i, val in value.items():
                if self._is_numerical[i]:
                    self._upper[i] = check_max(self._value[i], val)

        except AttributeError:
            if self._is_numerical:
                self._upper = check_max(self._value, value)

        if not self._lazy:
            self._normalized = self._normalize()

    @property
    def categories(self) -> Union[list, dict]:
        '''
        Return the list of all categories.
        '''
        return self._categories

    @categories.setter
    def categories(self, value: Union[Sequence, dict]) -> None:
        '''
        Set the categories (for categorical components only)
        '''
        try:
            for i, val in value.items():
                if not self._is_numerical[i]:
                    self._categories[i] = val
        except AttributeError:
            if not self._is_numerical:
                self._categories = value

        if not self._lazy:
            self._normalized = self._normalize()

    @property
    def is_numerical(self) -> Union[list, dict]:
        '''
        Return a boolean dataframe that shows if each component is numerical or not.
        '''
        return self._is_numerical

    @is_numerical.setter
    def is_numerical(self, value: Union[Sequence, dict]) -> None:
        # If it was numerical and the user assigns numerical (True and True) then it remains numerical.
        # But if it is not numerical, then we cannot change it to numerical.
        try:
            for i, val in value.items():
                self._is_numerical[i] = self._is_numerical[i] and val
        except AttributeError:
            self._is_numerical = self._is_numerical and value

        if not self._lazy:
            self._normalized = self._normalize()

    @property
    def normalizer(self) -> Union[list, dict]:
        return self._normalizer

    @normalizer.setter
    def normalizer(self, value: Union[Sequence, dict]) -> None:
        try:
            for i, val in value.items():
                self._normalizer[i] = val
        except AttributeError:
            self._normalizer = value

        if not self._lazy:
            self._normalized = self._normalize()

    def as_list(self) -> list:
        ''' return the value as list.'''
        try:
            return self.__remove_nestings(self._value.values())
        except AttributeError:
            return self._value

    def as_rldata_array(self) -> List["RLData"]:
        ''' return the value as a list of RLData.'''
        try:
            array = [RLData(value=self._value[key],
                            lower=self._lower[key],
                            upper=self._upper[key],
                            categories=self._categories[key],
                            is_numerical=self._is_numerical[key],
                            lazy_evaluation=self._lazy) for key in self._value.keys()]
        except AttributeError:
            array = [RLData(value=v,
                            lower=self._lower,
                            upper=self._upper,
                            categories=self._categories,
                            is_numerical=self._is_numerical,
                            lazy_evaluation=self._lazy) for v in self._value]

        return array

    def __remove_nestings(self, l: list) -> list:
        output = []
        for i in l:
            if isinstance(i, (list, dict)):
                output += self.__remove_nestings(i)
            else:
                output.append(i)

        return output

    def _normalize(self, keys: Optional[Sequence] = None) -> list:
        temp = []
        try:
            keys_list = keys if keys is not None else self._value.keys()
            for i in keys_list:
                if self._normal_form[i] is None:
                    if self._normalizer[i] is None:
                        func = self._normalizer_lambda_func
                    else:
                        func = self._normalizer[i]

                    try:
                        self._normal_form[i] = func({'value': self._value[i],
                                                     'lower': self._lower[i],
                                                     'upper': self._upper[i],
                                                     'categories': self._categories[i],
                                                     'is_numerical': self._is_numerical[i]})
                    except TypeError:
                        self._normal_form[i] = list(func({'value': x,
                                                          'lower': self._lower[i],
                                                          'upper': self._upper[i],
                                                          'categories': self._categories[i],
                                                          'is_numerical': self._is_numerical[i]}) for x in self._value[i])
                    except ZeroDivisionError:
                        self._normal_form[i] = [1]
                temp.append(self._normal_form[i])
        except AttributeError:
            if self._normal_form is None:
                if self._normalizer is None:
                    func = self._normalizer_lambda_func
                else:
                    func = self._normalizer

                try:
                    try:
                        self._normal_form = list(func({'value': x,
                                                       'lower': self._lower,
                                                       'upper': self._upper,
                                                       'categories': self._categories,
                                                       'is_numerical': self._is_numerical}) for x in self._value)
                    except TypeError:
                        self._normal_form = [func({'value': self._value,
                                                   'lower': self._lower,
                                                   'upper': self._upper,
                                                   'categories': self._categories,
                                                   'is_numerical': self._is_numerical})]
                except ZeroDivisionError:
                    self._normal_form = [1]

            temp = self._normal_form

        return self.__remove_nestings(temp)

    def normalize(self) -> "RLData":
        '''
        Normalize values.

        This function uses max and min for numericals and categories for categoricals to turn them into [0, 1] values.
        '''
        if self._lazy:
            return RLData(self._normalize(), lower=0, upper=1, lazy_evaluation=True)
        else:
            return RLData(self._normalized, lower=0, upper=1, lazy_evaluation=True)

    def __setitem__(self, key: Any, value: Any) -> "RLData":
        if key is None:  # complete list
            if isinstance(self._value, dict):  # Go from dict to list
                self._is_numerical = None
                self._lower = None
                self._upper = None
                self._categories = None
                self._normalizer = None

            old_lower = self._lower
            old_upper = self._upper
            old_categories = self._categories
            old_normalizer = self._normalizer

            if not hasattr(value, '__iter__') or isinstance(value, str):
                temp = [value]
            else:
                temp = value

            if self._categories is not None:
                for t in temp:
                    if t not in self._categories:
                        self._lower = old_lower
                        self._upper = old_upper
                        self._categories = old_categories
                        self._normalizer = old_normalizer
                        raise ValueError(
                            f'{t} is not found in {self._categories}.')
            elif self._lower is not None:
                for t in temp:
                    if not (self._lower <= t <= self._upper):
                        self._lower = old_lower
                        self._upper = old_upper
                        self._categories = old_categories
                        self._normalizer = old_normalizer
                        raise ValueError(
                            f'{t} is outside the range [{self._lower}, {self._upper}].')
            else:
                self._is_numerical = all(isinstance(v_i, Number)
                                         for v_i in temp)
                if self._is_numerical:
                    self._lower = min(temp)
                    self._upper = max(temp)
                    self._categories = None
                else:
                    self._lower = None
                    self._upper = None
                    self._categories = temp

            self._value = temp
            self._normal_form = None

        elif isinstance(key, slice):  # slice of a list
            if isinstance(self._value, dict):
                raise TypeError(
                    'Cannot use slice for a RLData instance of type dict.')

            if self._categories is not None:
                for t in value:
                    if t not in self._categories:
                        raise ValueError(
                            f'{t} is not found in {self._categories}.')
            elif self._lower is not None:
                for t in value:
                    if not (self._lower <= t <= self._upper):
                        raise ValueError(
                            f'{t} is outside the range [{self._lower}, {self._upper}].')
            else:
                raise RuntimeError(
                    'RLData is corrupted! No categories, lower or upper attributes found!')

            self._value[key] = value
            self._normal_form = None
        elif isinstance(self._value, dict):
            if not hasattr(value, '__iter__') or isinstance(value, str):
                temp = [value]
            else:
                temp = value

            if key not in self._value.keys():
                self._normalizer[key] = None
                self._is_numerical[key] = all(
                    isinstance(v_i, Number) for v_i in temp)
                if self._is_numerical[key]:
                    self._lower[key] = min(temp)
                    self._upper[key] = max(temp)
                    self._categories[key] = None
                else:
                    self._lower[key] = None
                    self._upper[key] = None
                    self._categories[key] = temp

            if self._is_numerical[key]:
                for v in temp:
                    if not (self._lower[key] <= v <= self._upper[key]):
                        raise ValueError(
                            f'{v} is outside the range [{self._lower[key]}, {self._upper[key]}].')
            else:
                if value not in self._categories[key]:
                    for v in temp:
                        if v not in self._categories[key]:
                            raise ValueError(
                                f'{value} is not found in {self._categories[key]}.')

            self._value[key] = value
            self._normal_form[key] = None

        else:  # index of a list
            if self._categories is not None:
                if value not in self._categories:
                    raise ValueError(
                        f'{value} is not found in {self._categories}.')
            elif self._lower is not None:
                if not (self._lower <= value <= self._upper):
                    raise ValueError(
                        f'{value} is outside the range [{self._lower}, {self._upper}].')
            else:
                raise RuntimeError(
                    'RLData is corrupted! No categories, lower or upper attributes found!')

            self._value[key] = value
            self._normal_form = None

        if not self._lazy:
            self._normalized = self._normalize()

        return value

    def __getitem__(self, key: Any) -> "RLData":
        try:
            return RLData(self._value[key],
                          lower=self._lower[key],
                          upper=self._upper[key],
                          categories=self._categories[key],
                          is_numerical=self._is_numerical[key],
                          normalizer=self._normalizer[key],
                          lazy_evaluation=self._lazy)
        except (TypeError, IndexError):
            if isinstance(key, Number):
                return self._value[key]
            else:
                return RLData(self._value[key],
                              lower=self._lower,
                              upper=self._upper,
                              categories=self._categories,
                              is_numerical=self._is_numerical,
                              normalizer=self._normalizer,
                              lazy_evaluation=self._lazy)

    def __delitem__(self, key: Any) -> None:
        del self._value[key]

        try:
            del self._lower[key]
            del self._upper[key]
        except KeyError:
            del self._categories[key]

        try:
            del self._is_numerical[key]
            del self._normalizer[key]
            del self._normalized[key]
        except KeyError:
            pass

    def clear(self) -> Union[list, dict]:
        return self._value.clear()

    def has_key(self, k: Any) -> bool:
        return k in self._value.keys()

    def update(self, kwargs: Union["RLData", list, dict]) -> "RLData":
        try:
            if isinstance(kwargs._value, list):
                if self.is_numerical:
                    if min(kwargs._value) >= self.lower and max(kwargs._value) <= self.upper:
                        self._value += kwargs._value
                else:
                    if all(item in self.categories for item in kwargs._value):
                        self._value += kwargs._value

                return self.value

        except AttributeError:
            pass

        for k, v in kwargs.items():
            self.__setitem__(k, v)

        return self.value

    def keys(self) -> Any:
        try:
            return self._value.keys()
        except AttributeError:
            try:
                return slice(0, len(self._value))
            except TypeError:
                return 0

    def values(self) -> "RLData":
        try:
            return self._value.values()
        except AttributeError:
            return self._value

    def items(self) -> enumerate:
        try:
            return self._value.items()
        except AttributeError:
            try:
                return enumerate(self._value)
            except TypeError:
                return enumerate([self._value])

    def pop(self, *args: Any) -> "RLData":
        return self._value.pop(*args)

    def __contains__(self, item: Any) -> bool:
        return item in self._value

    def __iter__(self) -> Iterator:
        return iter(self._value)

    def __add__(self, other: Union["RLData", list, dict]) -> "RLData":
        temp = RLData(value=self._value,
                      lower=self.lower,
                      upper=self.upper,
                      categories=self.categories,
                      is_numerical=self.is_numerical,
                      normalizer=self._normalizer
                      )
        temp.update(other)
        return temp

    def __iadd__(self, other: Union["RLData", list, dict]) -> "RLData":
        self.update(other)
        return self

    def __eq__(self, other: "RLData") -> bool:
        try:
            return (self.value == other.value).bool() and (
                ((self.upper == other.upper).bool() and (self.lower == other.lower).bool()) if self.is_numerical.bool() else
                (self.categories == other.categories).bool())
        except AttributeError:
            return (self.value == other.value) and (
                ((self.upper == other.upper) and (self.lower == other.lower)) if self.is_numerical else
                (self.categories == other.categories))

    def __ge__(self, other: Union["RLData", list, dict]) -> bool:
        if isinstance(other, RLData):
            other_value = other.value
        else:
            other_value = other

        try:
            return (self.value >= other_value).bool()
        except AttributeError:
            return (self.value >= other_value)

    def __gt__(self, other: Union["RLData", list, dict]) -> bool:
        if isinstance(other, RLData):
            other_value = other.value
        else:
            other_value = other

        try:
            return (self.value > other_value).bool()
        except AttributeError:
            return (self.value > other_value)

    def __le__(self, other: Union["RLData", list, dict]) -> bool:
        if isinstance(other, RLData):
            other_value = other.value
        else:
            other_value = other

        try:
            return (self.value <= other_value).bool()
        except AttributeError:
            return (self.value <= other_value)

    def __lt__(self, other: Union["RLData", list, dict]) -> bool:
        if isinstance(other, RLData):
            other_value = other.value
        else:
            other_value = other

        try:
            return (self.value < other_value).bool()
        except AttributeError:
            return (self.value < other_value)

    def __ne__(self, other: Union["RLData", list, dict]) -> bool:
        if isinstance(other, RLData):
            other_value = other.value
        else:
            other_value = other

        try:
            return (self.value != other_value).bool()
        except AttributeError:
            return (self.value != other_value)

    def __format__(self, formatstr: str) -> str:
        try:
            return '[' + ', '.join(format(i, formatstr) for i in self.value) + ']'
        except TypeError:
            return format(self.value, formatstr)
        except AttributeError:
            return False

    def __len__(self) -> int:
        try:
            return len(self._value)
        except TypeError:
            return 1

    def __hash__(self) -> int:
        return self._value.__hash__()

    def __repr__(self) -> str:
        return f'[{str(self.value)}]\nlower={str(self._lower)}\nupper={str(self._upper)}\ncategories={str(self._categories)}'

    def __str__(self) -> str:
        return str(self._value)


if __name__ == '__main__':
    # d = RLData([1, 2, 3], lower=1, upper=10)
    # print(d._value)
    # print(d._normalized)
    # d.value = 10
    # print(d._normalized)

    # d = RLData({'a': [10, 20], 'b': [30, 10, 5, 40], 'c': 50, 'd': 'hello'},
    #            lower={'a': 1, 'b': 2, 'c': 3, 'd': 'a'})
    # print(d._value)
    # print(d._normalized)
    # d.upper = {'b': 100, 'c': 50, 'a': 20, 'd': 'zzzzzz'}
    # d.lower = {'b': -10, 'c': 0, 'a': 1, 'd': '0'}
    # print(d._value)
    # print(d._normalized)
    # # d.is_numerical={'a': False}
    # for temp in d.as_rldata_array():
    #     print(temp)

    # print(d.normalize())
    d = RLData([1, 2, 3], lazy_evaluation=True)
    print(d.value)
    print(d.normalize())
    print(d.normalize())
    d += [1, 2, 3]
    print(d.normalize())
    print(d.as_rldata_array())
    print(d == d)
    d += RLData([1.5, 2.5, 3], lower=0)
    print(d)

    d = RLData({'a': 1, 'b': 2, 'c': 3},
               lower={'a': 0, 'b': 0},
               upper={'a': 10, 'b': 10},
               categories={'c': (1, 2, 3)},
               is_numerical={'c': False})

    print(d+RLData({'a': 5, 'c': 1}, is_numerical={'a': True,
                                                   'c': False}, lazy_evaluation=True))

    d1 = RLData(['a', 'b', 'c'], categories=['a', 'b', 'c'])
    assert d1.value == ['a', 'b', 'c']
    print(d1.value)
    print(d1.normalize())
    print(d1.as_rldata_array())
    d = RLData({'tuples': [(1, 1), (1, 2), (1, 3)], 'ints': 1})
    print(d.value)
    print(d.normalize())
    d_temp = d['tuples']
    print(d_temp[0])
    print(d.as_rldata_array()[0].normalize().as_list())
