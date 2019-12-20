# -*- coding: utf-8 -*-
'''
RLData class
==============

A data type used for state and action variables.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from numbers import Number
from collections.abc import Iterable

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

    def __init__(self, value=[0], **kwargs):
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

        self.value = value

        try:
            self.is_numerical = kwargs['is_numerical']
        except KeyError:
            pass

        try:
            self.lower = kwargs['lower']
        except KeyError:
            pass

        try:
            self.upper = kwargs['upper']
        except KeyError:
            pass

        try:
            self.categories = kwargs['categories']
        except KeyError:
            pass

        try:
            self.normalizer = kwargs['normalizer']
        except KeyError:
            pass
        self._normalizer_lambda_func = lambda x: (x['value']-x['lower'])/(x['upper']-x['lower']) if x['is_numerical'] else list(int(x_i == x['value']) for x_i in x['categories'])

        self._lazy = kwargs.get('lazy_evaluation', False)

        if not self._lazy:
            self._normalized = self._normalize()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        '''
        Sets the value of the RLData instance.

        Attributes:
            v: value to store as a list or a dictionary. If a list is provided, it is stored as `value`. To have named values, use dictionary.
        '''
        try:
            for i, v_i in v.items():
                self.__setitem__(i, v_i)
        except AttributeError:
            self.__setitem__(None, v)

    @property
    def lower(self):
        '''
        Return the lower bound
        '''
        return self._lower

    @lower.setter
    def lower(self, value):
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
    def upper(self):
        '''
        Return the upper bound
        '''
        return self._upper

    @upper.setter
    def upper(self, value):
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
    def categories(self):
        '''
        Return the list of all categories.
        '''
        return self._categories

    @categories.setter
    def categories(self, value):
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
    def is_numerical(self):
        '''
        Return a boolean dataframe that shows if each component is numerical or not.
        '''
        return self._is_numerical

    @is_numerical.setter
    def is_numerical(self, value):
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
    def normalizer(self):
        return self._normalizer

    @normalizer.setter
    def normalizer(self, value):
        try:
            for i, val in value.items():
                self._normalizer[i] = val
        except AttributeError:
            self._normalizer = value

        if not self._lazy:
            self._normalized = self._normalize()

    def as_list(self):
        ''' return the value as list.'''
        try:
            return self.__remove_nestings(self._value.values())
        except AttributeError:
            return self._value

    def as_rldata_array(self):
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

    def __remove_nestings(self, l):
        output = []
        for i in l:
            if isinstance(i, (list, dict)):
                output += self.__remove_nestings(i)
            else:
                output.append(i)

        return output

    def _normalize(self):
        temp = []
        try:
            for i in self._value.keys():
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

    def normalize(self):
        '''
        Normalize values.

        This function uses max and min for numericals and categories for categoricals to turn them into [0, 1] values.
        '''
        if self._lazy:
            return RLData(self._normalize(), lower=0, upper=1, lazy_evaluation=True)
        else:
            return RLData(self._normalized, lower=0, upper=1, lazy_evaluation=True)

    def __setitem__(self, key, value):
        if key is None and isinstance(self._value, dict):
            self._value = []
            del self._is_numerical
        if not isinstance(value, Iterable) or isinstance(value, str):
            value = [value]
        try:
            if key not in self._value.keys():
                non_string_iterable = hasattr(value, '__iter__') and not isinstance(value, str)
                self._value[key] = value
                self._is_numerical[key] = all(isinstance(v_i, Number) for v_i in value) if non_string_iterable else isinstance(value, Number)
                self._normalizer[key] = None
                self._categories[key] = None

                if self._is_numerical[key]:
                    self._lower[key] = min(value) if non_string_iterable else value
                    self._upper[key] = max(value) if non_string_iterable else value
                    self._categories[key] = None
                else:
                    self._lower[key] = None
                    self._upper[key] = None
                    self._categories[key] = value if non_string_iterable else [value]

            else:
                if self._is_numerical[key]:
                    if not all(self.lower[key] <= v <= self.upper[key] for v in value):
                        raise ValueError(f'{value} is not in the range of {self.lower[key]} to {self.upper[key]}.')
                elif any(v not in self.categories[key] for v in value):
                    raise ValueError(f'{value} is not a valid category.')

                self._value[key] = value

            self._normal_form[key] = None

        except AttributeError:
            try:
                if self._is_numerical:
                    if not (self.lower <= value <= self.upper):
                        raise ValueError(f'{value} is not in the range of {self.lower} to {self.upper}.')
                elif value not in self.categories:
                    raise ValueError(f'{value} is not a valid category.')

                self._value = value

            except AttributeError:
                non_string_iterable = hasattr(value, '__iter__') and not isinstance(value, str)
                self._value = value
                self._is_numerical = all(isinstance(v_i, Number) for v_i in value) if non_string_iterable else isinstance(value, Number)
                self._normalizer = None
                self._categories = None

                if self._is_numerical:
                    self._lower = min(value) if non_string_iterable else value
                    self._upper = max(value) if non_string_iterable else value
                    self._categories = None
                else:
                    self._lower = None
                    self._upper = None
                    self._categories = value if non_string_iterable else [value]


            self._normal_form = None

        if not self._lazy:
            self._normalized = self._normalize()

        return value

    def __getitem__(self, key):
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

    def __delitem__(self, key):
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

    def clear(self):
        return self._value.clear()

    def has_key(self, k):
        return k in self._value.keys()

    def update(self, kwargs):
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

    def keys(self):
        try:
            return self._value.keys()
        except AttributeError:
            try:
                return slice(0, len(self._value))
            except TypeError:
                return 0

    def values(self):
        try:
            return self._value.values()
        except AttributeError:
            return self._value

    def items(self):
        try:
            return self._value.items()
        except AttributeError:
            try:
                return enumerate(self._value)
            except TypeError:
                return enumerate([self._value])

    def pop(self, *args):
        return self._value.pop(*args)

    def __contains__(self, item):
        return item in self._value

    def __iter__(self):
        return iter(self._value)

    def __add__(self, other):
        temp = RLData(value=self._value,
                      lower=self.lower,
                      upper=self.upper,
                      categories=self.categories,
                      is_numerical=self.is_numerical,
                      normalizer=self._normalizer
                      )
        temp.update(other)
        return temp

    def __iadd__(self, other):
        self.update(other)
        return self

    def __eq__(self, other):
        try:
            return (self.value == other.value).bool() and (
                ((self.upper == other.upper).bool() and (self.lower == other.lower).bool()) if self.is_numerical.bool() else 
                    (self.categories == other.categories).bool())
        except AttributeError:
            return (self.value == other.value) and (
                ((self.upper == other.upper) and (self.lower == other.lower)) if self.is_numerical else 
                    (self.categories == other.categories))

    def __ge__(self, other):
        if isinstance(other, RLData):
            other_value = other.value
        else:
            other_value = other

        try:
            return (self.value >= other_value).bool()
        except AttributeError:
            return (self.value >= other_value)

    def __gt__(self, other):
        if isinstance(other, RLData):
            other_value = other.value
        else:
            other_value = other

        try:
            return (self.value > other_value).bool()
        except AttributeError:
            return (self.value > other_value)

    def __le__(self, other):
        if isinstance(other, RLData):
            other_value = other.value
        else:
            other_value = other

        try:
            return (self.value <= other_value).bool()
        except AttributeError:
            return (self.value <= other_value)

    def __lt__(self, other):
        if isinstance(other, RLData):
            other_value = other.value
        else:
            other_value = other

        try:
            return (self.value < other_value).bool()
        except AttributeError:
            return (self.value < other_value)

    def __ne__(self, other):
        if isinstance(other, RLData):
            other_value = other.value
        else:
            other_value = other

        try:
            return (self.value != other_value).bool()
        except AttributeError:
            return (self.value != other_value)

    def __format__(self, formatstr):
        try:
            return '[' + ', '.join(format(i, formatstr) for i in self.value) + ']'
        except TypeError:
            return format(self.value, formatstr)
        except AttributeError:
            return False

    def __len__(self):
        try:
            return len(self._value)
        except TypeError:
            return 1

    # def __contains__(self, x):
    #     for i in range(len(self)):
    #         try:
    #             if (x in self.value[i]) | (x == self.value[i]):
    #                 return True
    #         except TypeError:
    #             if x == self.value[i]:
    #                 return True
    #     return False

    def __hash__(self):
        return self._value.__hash__()

    # def __repr__(self):
    #     return ''.join(['[', str(self.value), '], min=', str(self._min), ', max=', str(self._max)])

    def __repr__(self):
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
    d = RLData([1, 2, 3])
    print(d.value)
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

    print(d+RLData({'a': 5, 'c': 1}, is_numerical={'a': True, 'c': False}, lazy_evaluation=True))

    d1 = RLData(['a', 'b', 'c'], categories=['a', 'b', 'c'])
    assert d1.value==['a', 'b', 'c']
    print(d1.value)
    print(d1.normalize())
    print(d1.as_rldata_array())
    d = RLData({'tuples': [(1, 1), (1, 2), (1, 3)], 'ints': 1})
    print(d.value)
    print(d.normalize())
    d_temp = d['tuples']
    print(d_temp[0])
    print(d.as_rldata_array()[0].normalize().as_list())
