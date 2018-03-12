# -*- coding: utf-8 -*-
'''
ValueSet class
==============

A data type used for state and action variables.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import numpy as np


def main():
    # create an empty ValueSet and print it with different formats:
    e = ValueSet()
    print(e.value, e.to_list(), e.to_nparray())
    # create a ValueSet with data and print binary representation and normalized versions:
    s = ValueSet(1, 7, 10, 3)
    print(s.binary_representation().value)
    print(s.normalizer(0, 1).value, s.to_nparray())

class ValueSet():
    '''
    Provide a data type for state and action in reinforcement learning.

    Attributes
    ----------
        value: the value stored as a tuple
        min: minimum value in the stored tuple
        max: maximum value in the stored tuple

    Methods
    -------
        to_list: return the value as a list
        to_nparray: return the value as a numpy array
        binary_representation: return the value as a zero-one vector
        normalizer: normalize the value
    '''
    def __init__(self, *args):
        self.value = args
        if not self.value:
            self._min = None
            self._max = None
        elif self._one_type:
            self._min = min(self.value)
            self._max = max(self.value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        '''
        Set the value.

        During assignment, a couple of flags are set:
            _scalable = True, if the data is scalable (list or tuple of int or float).
            _enumerable = True, if the data is enumerable (list or tuple of anything other than float).
            _one_type = True, if all elements of the data is of the same type.
        '''
        self._scalable = True
        self._enumerable = True
        self._one_type = True
        try:
            _ = (e for e in value)
            first_type = type(value[0])
            v = value
        except TypeError:
            first_type = type(value)
            v = [value]
        except IndexError:
            first_type = type(value)
            v = []

        for val in v:
            if not isinstance(val, first_type):
                self._one_type = False
            if not isinstance(val, int):
                if isinstance(val, float):
                    self._enumerable = False
                else:
                    self._scalable = False
        self._value = tuple(v)
        # else:
        #     self._scalable = False
        #     self._enumerable = False
        #     self._one_type = True
        #     self._value = tuple([])

    @property
    def max(self):
        '''
        Return the max
        
        returns the max if _one_type is True, None otherwise.
        '''
        if self._one_type:
            return self._max
        else:
            return None

    @max.setter
    def max(self, value):
        '''
        Set the max
        
        sets the max if _one_type is True.
        raises TypeError if data is mixed.
        raises ValueError if the given value is less than the max(value).
        '''
        if not self._one_type:
            raise TypeError('Mixed data doesn\'t have min!')
        if self._max is None:
            self._max = value
            return
        if value < max(self.value):
            raise ValueError('The provided value is less than the biggest number I have.')
        self._max = value

    @property
    def min(self):
        '''
        Return the min
        
        returns the min if _one_type is True, None otherwise.
        '''
        if self._one_type:
            return self._min
        else:
            return None

    @min.setter
    def min(self, value):
        '''
        Set the min
        
        sets the min if _one_type is True.
        raises TypeError if data is mixed.
        raises ValueError if the given value is greater than the min(value).
        '''
        if not self._one_type:
            raise TypeError('Mixed data doesn\'t have min!')
        if self._min is None:
            self._min = value
            return
        if value > min(self.value):
            raise ValueError('The provided value is greater than the smallest number I have.')
        self._min = value

    def to_list(self):
        ''' return the value as list.'''
        return list(self._value)

    def to_nparray(self):
        ''' return the value as numpy array.'''
        return np.array(list(self._value))

    def binary_representation(self):
        '''
        Convert the value to a zero-one sequence.

        This is useful if you want to use a state or an action as an input to a neural network.
        Raises TypeError if data is not enumerable or of mixed type.

        Note: The result is returned as a new ValueSet. Use .value, .to_list, or .to_nparray to use it.
        '''
        if not self._enumerable:
            raise TypeError('The type of data doesn\'t allow binary representation!')
        if not self._one_type:
            raise TypeError('Mixed data doesn\'t allow binary representation!')
        data_range = self.max - self.min + 1
        bin_rep = [0]*data_range*len(self.value)
        for i in range(len(self.value)):
            bin_rep[(self.value[i]-self.min)+(i*data_range)] = 1
        return ValueSet(*bin_rep)

    def normalizer(self, lower_bound=0, upper_bound=1):
        '''
        normalize the value.

        Raises TypeError if data is not scalable.
        Raises ValueError if lower_bound is greater than upper_bound.

        Note: The result is returned as a new ValueSet. Use .value, .to_list, or .to_nparray to use it.
        '''
        if not self._scalable:
            raise TypeError('The type of data doesn\'t allow scaling!')
        if lower_bound >= upper_bound:
            raise ValueError('lower_bound should be less than upper_bound!')
        factor = (upper_bound - lower_bound)/(self.max - self.min)
        normal = ((factor * (v - self.min) + lower_bound) for v in self.value)
        return ValueSet(*list(normal))

    def __eq__(self, other):
        try:
            return self._value == other._value
        except AttributeError:
            return False
        
    def __hash__(self):
        return self.value.__hash__()


if __name__ == '__main__':
    main()