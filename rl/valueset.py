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
    s = ValueSet([1, 7, 10, 3], min=0, max=15)
    print('{}'.format(s))
    print([10, 3] in s)
    print('{}'.format(s.binary_representation()))
    print('{:2.2} {}'.format(s.normalizer(0, 1), s.to_nparray()))
    directions = ['U', 'D', 'L', 'R']
    t = ValueSet(directions, binary=lambda x: (directions.index(x), len(directions)))
    print('{}, {}'.format(t, t.binary_representation()))
    def convert(x, directions = ['U', 'D', 'L', 'R']):
        bin_rep = [0]*(len(directions)-1)
        for value in x:
            index = directions.index(value)
            if index != 0:
                bin_rep[index-1] = 1
        return bin_rep

    t = ValueSet(directions, binary=convert)
    print('{}, {}'.format(t, t.binary_representation()))
    array = s.as_valueset_array()
    for a in array:
        print(a.value, a.max, a.min)

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
    def __init__(self, value=[], **kwargs):
        '''
        Initialize a ValueSet.

        Arguments:
            value: the value to be stored.
            min: minimum possible value (if not supplied, min is calculated from the data).
            max: maximum possible value (if not supplied, max is calculated from the data).
            normalizer: the function with which the value should be normalized (Default=None).
            binary: the function with which the value should be turned into a zero-one vector (Default=None).
        '''
        self.value = value
        try:
            self._min = kwargs['min']
        except KeyError:
            if not self.value:
                self._min = None
            else:
                self._min = min(self.value)
        try:
            self._max = kwargs['max']
        except KeyError:
            if not self.value:
                self._max = None
            else:
                self._max = max(self.value)
        try:
            self._normalizer_function = kwargs['normalizer']
        except KeyError:
            self._normalizer_function = None
        try:
            self._binary_function = kwargs['binary']
        except KeyError:
            self._binary_function = None

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

    def as_valueset_array(self):
        ''' return the value as a list of ValueSets.'''
        array = [ValueSet(v) for v in self._value]
        for a in array:
            a.max = self.max
            a.min = self.min
            a._normalizer_function = self._normalizer_function
            a._binary_function = self._binary_function
        return array

    def binary_representation(self):
        '''
        Convert the value to a zero-one sequence.

        This is useful if you want to use a state or an action as an input to a neural network.
        Raises TypeError if data is not enumerable or of mixed type.

        Note: The result is returned as a new ValueSet. Use .value, .to_list, or .to_nparray to use it.
        '''
        if self._binary_function is not None:
            bin_rep = []
            for i in range(len(self.value)):
                temp = self._binary_function(self.value[i])
                if (min(temp) == 0) & (max(temp) <= 1):  # whether the function returns the list or index, length pair
                    bin_rep = bin_rep + temp
                else:
                    index, length = temp
                    temp = [0]*(length-1)
                    if index != 0:
                        temp[index-1] = 1  # for n categories, we need n-1 bins.
                    bin_rep = bin_rep + temp 
            return ValueSet(bin_rep)

        if not self._enumerable:
            raise TypeError('The type of data doesn\'t allow binary representation!')
        if not self._one_type:
            raise TypeError('Mixed data doesn\'t allow binary representation!')
        try:
            data_range = self.max - self.min  # for n categories, we need n-1 bins.
            bin_rep = [0]*data_range*len(self.value)
            for i in range(len(self.value)):
                index = (self.value[i]-self.min) + (i*data_range) - 1
                if index != 0:
                    bin_rep[index] = 1
        except TypeError:
            if self._binary_function is None:
                raise RuntimeError('Failed to automatically convert to binary representation.\n Use set_function to provide custom function.')

        return ValueSet(bin_rep)

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
        return ValueSet(list(normal))

    def set_function(self, normalizer=None, binary=None):
        '''
        set custom normalizer and binary representation functions.

        normalizer should get a value and return its normalized result.
        binary should get a value and return a tuple of form (index, len). len is the length of 
        the array and index is the index of the 1 in the list.
        '''
        if normalizer is not None:
            self._normalizer_function = normalizer
        if binary is not None:
            self._binary_function = binary 

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


if __name__ == '__main__':
    main()