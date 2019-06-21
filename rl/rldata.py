# -*- coding: utf-8 -*-
'''
RLData class
==============

A data type used for state and action variables.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import numpy as np
import pandas as pd
from numbers import Number


class RLData:
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
        '''

        self._value = pd.DataFrame()
        self.value = value

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
            self.is_numerical = kwargs['is_numerical']
        except KeyError:
            pass

        try:
            self.normalizer = kwargs['normalizer']
        except KeyError:
            pass

    @property
    def value(self):
        return self._value.loc['value']

    @value.setter
    def value(self, v):
        if isinstance(v, dict):
            same_columns = list(self._value.columns) == list(v.keys())
        else:
            try:
                same_columns = ['value'] == list(v)
            except TypeError:
                same_columns = ['value'] == list([v])
        if same_columns:
            self._value.loc['value'] = v
        else:
            try:
                self._value = pd.DataFrame(columns=v.keys())
                self._value.loc['value'] = v
            except AttributeError:
                self._value = pd.DataFrame(columns=['value'])
                self._value.loc['value'] = [v]

        if not same_columns:
            self._value.loc['is_numerical'] = dict((i, all(isinstance(v, Number) for v in val[0]) if hasattr(
                val[0], '__iter__') and not isinstance(val[0], str) else isinstance(val[0], Number)) for i, val in self._value.iteritems())
            self._value.loc['normalizer'] = None

            for i, val in self._value.iteritems():
                if self._value.loc['is_numerical', i]:
                    self._value.loc['lower', i] = min(val[0]) if hasattr(val[0], '__iter__') and not isinstance(val[0], str) else val[0]
                    self._value.loc['upper', i] = max(val[0]) if hasattr(val[0], '__iter__') and not isinstance(val[0], str) else val[0]
                else:
                    self._value.loc['categories', i] = [val[0]]

    @property
    def lower(self):
        '''
        Return the lower bound
        '''
        return self._value.loc['lower']

    @lower.setter
    def lower(self, value):
        '''
        Set the lower bound
        '''
        try:
            for i, val in value.items():
                if self._value.loc['is_numerical', i]:
                    try:
                        if min(self._value.loc['value', i]) < val:
                            raise ValueError(
                                'The provided value is greater than the smallest number I have.')
                    except TypeError:
                        if self._value.loc['value', i] < val:
                            raise ValueError(
                                'The provided value is greater than the smallest number I have.')

                    self._value.loc['lower', i] = val
        except AttributeError:
            if self._value.loc['is_numerical', 'value']:
                try:
                    if min(self._value.loc['value', 'value']) < value:
                        raise ValueError('The provided value is greater than the smallest number I have.')
                except TypeError:
                    if self._value.loc['value', 'value'] < value:
                        raise ValueError('The provided value is greater than the smallest number I have.')

                self._value.loc['lower'] = value

    @property
    def upper(self):
        '''
        Return the upper bound
        '''
        return self._value.loc['upper']

    @upper.setter
    def upper(self, value):
        '''
        Set the upper bound
        '''
        try:
            for i, val in value.items():
                if self._value.loc['is_numerical', i]:
                    try:
                        if max(self._value.loc['value', i]) > val:
                            raise ValueError(
                                'The provided value is less than the biggest number I have.')
                    except TypeError:
                        if self._value.loc['value', i] > val:
                            raise ValueError(
                                'The provided value is less than the biggest number I have.')

                    self._value.loc['upper', i] = val

        except AttributeError:
            if self._value.loc['is_numerical', 'value']:
                try:
                    if max(self._value.loc['value', 'value']) > value:
                        raise ValueError('The provided value is less than the biggest number I have.')
                except TypeError:
                    if self._value.loc['value', 'value'] > value:
                        raise ValueError('The provided value is greater than the smallest number I have.')
                self._value.loc['upper'] = value

    @property
    def categories(self):
        return self._value.loc['categories']

    @categories.setter
    def categories(self, value):
        '''
        Set the categories (for categorical components only)
        '''
        try:
            for i, val in value.items():
                if not self._value.loc['is_numerical', i]:
                    self._value.loc['categories', i] = val
        except AttributeError:
            if not self._value.loc['is_numerical', 'value']:
                self._value.loc['categories', 'value'] = value

    @property
    def is_numerical(self):
        return self._value.loc['is_numerical']

    @is_numerical.setter
    def is_numerical(self, value):
        # If it was numerical and the user assigns numerical (True and True) then it remains numerical.
        # But if it is not numerical, then we cannot change it to numerical.
        try:
            for i, val in value.items():
                self._value.loc['is_numerical', i] = self._value.loc['is_numerical', i] and val
        except AttributeError:
            self._value.loc['is_numerical', 'value'] = self._value.loc['is_numerical', 'value'] and value

    @property
    def normalizer(self):
        return self._value.loc['normalizer']

    @normalizer.setter
    def normalizer(self, value):
        try:
            for i, val in value.items():
                self._value.loc['normalizer', i] = val
        except AttributeError:
            self._value.loc['normalizer', 'value'] = value

    def as_list(self):
        ''' return the value as list.'''
        return list(self._value.loc['value'])

    def as_nparray(self):
        ''' return the value as numpy array.'''
        return np.array(self.as_list())

    def as_rldata_array(self):
        ''' return the value as a list of RLData.'''
        array = [RLData(value=self._value.loc['value', col],
                        lower=self._value.loc['lower', col],
                        upper=self._value.loc['upper', col],
                        categories=self._value.loc['categories', col],
                        is_numerical=self._value.loc['is_numerical', col]) for col in self._value.columns]

        return array

    def normalize(self):
        '''
        Normalize values.

        This function uses max and min for numericals and categories for categoricals to turn them into [0, 1] values.
        '''
        temp = np.array([])
        for i in self._value.columns:
            if not np.isnan(self._value.loc['normalizer', i]):
                func = self._value.loc['normalizer', i]
            elif self._value.loc['is_numerical', i]:
                func = lambda x: (x['value']-x['lower'])/(x['upper']-x['lower'])
            else:  # categorical
                func = lambda x: list(int(x_i == x['value']) for x_i in x['categories'])

            try:
                temp = np.append(temp, [func({'value': self._value.loc['value', i],
                                   'lower': self._value.loc['lower', i],
                                   'upper': self._value.loc['upper', i],
                                   'categories': self._value.loc['categories', i]})])
            except TypeError:
                temp = np.append(temp, [func({'value': x,
                                   'lower': self._value.loc['lower', i],
                                   'upper': self._value.loc['upper', i],
                                   'categories': self._value.loc['categories', i]}) for x in self._value.loc['value', i]])

        return RLData(temp, lower=0, upper=1)

    def __eq__(self, other):
        try:
            return self._value == other._value
        except AttributeError:
            return False

    def __ge__(self, other):
        try:
            return self._value >= other._value
        except AttributeError:
            return False

    def __gt__(self, other):
        try:
            return self._value > other._value
        except AttributeError:
            return False

    def __le__(self, other):
        try:
            return self._value <= other._value
        except AttributeError:
            return False

    def __lt__(self, other):
        try:
            return self._value < other._value
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self._value != other._value
        except AttributeError:
            return False

    def __format__(self, formatstr):
        try:
            return '[' + ', '.join(format(i, formatstr) for i in self._value) + ']'
        except AttributeError:
            return False

    def __len__(self):
        return len(self._value)

    def __contains__(self, x):
        for i in range(len(self)):
            try:
                if (x in self.value[i]) | (x == self.value[i]):
                    return True
            except TypeError:
                if x == self.value[i]:
                    return True
        return False

    def __hash__(self):
        return self.value.__hash__()

    # def __repr__(self):
    #     return ''.join(['[', str(self.value), '], min=', str(self._min), ', max=', str(self._max)])

    def __repr__(self):
        return self._value.to_string()


if __name__ == '__main__':
    d = RLData([1, 2, 3], lower=1, max=10)
    print(d._value)
    d.value = {'value': 10}
    d = RLData({'a': [10, 20], 'b': [30, 10, 5, 40], 'c': 50, 'd': 'hello'},
               lower={'a': 1, 'b': 2, 'c': 3, 'd': 'a'})
    print(d._value)
    d.upper = {'b': 100, 'c': 50, 'a': 20, 'd': 'zzzzzz'}
    d.lower = {'b': -10, 'c': 0, 'a': 1, 'd': '0'}
    d.range = {'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'd': [7, 8]}
    print(d._value)
    # d.is_numerical={'a': False}
    d.range = {'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'd': [7, 8]}
    for temp in d.as_rldata_array():
        print(temp)

    print(d.normalize())
