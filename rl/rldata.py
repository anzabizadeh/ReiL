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
            lazy_evaluation: whether to store normalized values or compute on-demand (Default: False)
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

        self._lazy = kwargs.get('lazy_evaluation', False)

    @property
    def value(self):
        return self._value.value

    @value.setter
    def value(self, v):
        if isinstance(v, dict):
            same_index = list(self._value.index) == list(v.keys())
        else:
            same_index = len(self._value.index) == 1

        if same_index:
            try:
                self._value.value = v
            except ValueError:
                self._value.value = self._value.value.astype(object)
                self._value.value.at['value'] = v
        else:
            try:
                self._value = pd.DataFrame(index=v.keys(), columns=['value', 'lower', 'upper', 'categories', 'is_numerical', 'normalizer', 'modified'])
                self._value.value.loc[list(v.keys())] = list(v.values())
            except AttributeError:
                self._value = pd.DataFrame(index=['value'], columns=['value', 'lower', 'upper', 'categories', 'is_numerical', 'normalizer', 'modified'])
                self._value.value = [v]

        self._value.modified = True

        if not same_index:
            self._value.is_numerical = tuple(all(isinstance(v, Number) for v in val[0]) if hasattr(
                val[0], '__iter__') and not isinstance(val[0], str) else isinstance(val[0], Number) for i, val in self._value.iterrows())
            self._value.normalizer = None
            self._value.categories = None

            for i, val in self._value.iterrows():
                if self._value.at[i, 'is_numerical']:
                    self._value.at[i, 'lower'] = min(val[0]) if hasattr(val[0], '__iter__') and not isinstance(val[0], str) else val[0]
                    self._value.at[i, 'upper'] = max(val[0]) if hasattr(val[0], '__iter__') and not isinstance(val[0], str) else val[0]
                else:
                    self._value.at[i, 'categories'] = [val[0]]

    @property
    def lower(self):
        '''
        Return the lower bound
        '''
        return self._value.lower

    @lower.setter
    def lower(self, value):
        '''
        Set the lower bound
        '''
        try:
            for i, val in value.items():
                if self._value.at[i, 'is_numerical']:
                    try:
                        if min(self._value.at[i, 'value']) < val:
                            raise ValueError(
                                'The provided value is greater than the smallest number I have.')
                    except TypeError:
                        if self._value.at[i, 'value'] < val:
                            raise ValueError(
                                'The provided value is greater than the smallest number I have.')

                    self._value.at[i, 'lower'] = val
        except AttributeError:
            if self._value.at['value', 'is_numerical']:
                try:
                    if min(self._value.at['value', 'value']) < value:
                        raise ValueError('The provided value is greater than the smallest number I have.')
                except TypeError:
                    if self._value.at['value', 'value'] < value:
                        raise ValueError('The provided value is greater than the smallest number I have.')

                self._value.lower = value

    @property
    def upper(self):
        '''
        Return the upper bound
        '''
        return self._value.upper

    @upper.setter
    def upper(self, value):
        '''
        Set the upper bound
        '''
        try:
            for i, val in value.items():
                if self._value.at[i, 'is_numerical']:
                    try:
                        if max(self._value.at[i, 'value']) > val:
                            raise ValueError(
                                'The provided value is less than the biggest number I have.')
                    except TypeError:
                        if self._value.at[i, 'value'] > val:
                            raise ValueError(
                                'The provided value is less than the biggest number I have.')

                    self._value.at[i, 'upper'] = val

        except AttributeError:
            if self._value.at['value', 'is_numerical']:
                try:
                    if max(self._value.at['value', 'value']) > value:
                        raise ValueError('The provided value is less than the biggest number I have.')
                except TypeError:
                    if self._value.at['value', 'value'] > value:
                        raise ValueError('The provided value is greater than the smallest number I have.')
                self._value.upper = value

    @property
    def categories(self):
        return self._value.categories

    @categories.setter
    def categories(self, value):
        '''
        Set the categories (for categorical components only)
        '''
        try:
            for i, val in value.items():
                if not self._value.at[i, 'is_numerical']:
                    self._value.at[i, 'categories'] = val
        except AttributeError:
            if not self._value.at['value', 'is_numerical']:
                self._value.at['value', 'categories'] = value

    @property
    def is_numerical(self):
        return self._value.is_numerical

    @is_numerical.setter
    def is_numerical(self, value):
        # If it was numerical and the user assigns numerical (True and True) then it remains numerical.
        # But if it is not numerical, then we cannot change it to numerical.
        try:
            for i, val in value.items():
                self._value.at[i, 'is_numerical'] = self._value.at[i, 'is_numerical'] and val
        except AttributeError:
            self._value.at['value', 'is_numerical'] = self._value.at['value', 'is_numerical'] and value

    @property
    def normalizer(self):
        return self._value.normalizer

    @normalizer.setter
    def normalizer(self, value):
        try:
            for i, val in value.items():
                self._value.at[i, 'normalizer'] = val
        except AttributeError:
            self._value.at['value', 'normalizer'] = value

    def as_list(self):
        ''' return the value as list.'''
        return list(self._value.value)

    def as_nparray(self):
        ''' return the value as numpy array.'''
        return np.array(self.as_list())

    def as_rldata_array(self):
        ''' return the value as a list of RLData.'''
        if self._value.shape[1] > 1:
            array = [RLData(value=self._value.at[row, 'value'],
                            lower=self._value.at[row, 'lower'],
                            upper=self._value.at[row, 'upper'],
                            categories=self._value.at[row, 'categories'],
                            is_numerical=self._value.at[row, 'is_numerical']) for row in self._value.index]
        else:
            array = [RLData(value=v,
                            lower=self._value.at['value', 'lower'],
                            upper=self._value.at['value', 'upper'],
                            categories=self._value.at['value', 'categories'],
                            is_numerical=self._value.at['value', 'is_numerical']) for v in self._value.at['value', 'value']]

        return array

    def normalize(self):
        '''
        Normalize values.

        This function uses max and min for numericals and categories for categoricals to turn them into [0, 1] values.
        '''
        temp = np.array([])
        for i in self._value.index:
            if self._value.at[i, 'normalizer'] is not None:
                func = self._value.at[i, 'normalizer']
            elif self._value.at[i, 'is_numerical']:
                func = lambda x: (x['value']-x['lower'])/(x['upper']-x['lower'])
            else:  # categorical
                func = lambda x: list(int(x_i == x['value']) for x_i in x['categories'])

            try:
                temp = np.append(temp, [func({'value': self._value.at[i, 'value'],
                                   'lower': self._value.at[i, 'lower'],
                                   'upper': self._value.at[i, 'upper'],
                                   'categories': self._value.at[i, 'categories']})])
            except TypeError:
                temp = np.append(temp, [func({'value': x,
                                   'lower': self._value.at[i, 'lower'],
                                   'upper': self._value.at[i, 'upper'],
                                   'categories': self._value.at[i, 'categories']}) for x in self._value.at[i, 'value']])

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
    d.value = 10
    d = RLData({'a': [10, 20], 'b': [30, 10, 5, 40], 'c': 50, 'd': 'hello'},
               lower={'a': 1, 'b': 2, 'c': 3, 'd': 'a'})
    print(d._value)
    d.upper = {'b': 100, 'c': 50, 'a': 20, 'd': 'zzzzzz'}
    d.lower = {'b': -10, 'c': 0, 'a': 1, 'd': '0'}
    print(d._value)
    # d.is_numerical={'a': False}
    for temp in d.as_rldata_array():
        print(temp)

    print(d.normalize())
