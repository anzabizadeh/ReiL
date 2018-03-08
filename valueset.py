# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 23:39:00 2018

@author: Sadjad Anzabi Zadeh

This module contains a class that implements values.

"""

import numpy as np


def main():
    s = ValueSet(1, 15, 10, 3)
    print(s.binary_representation().value)
    print(s.normalizer(0, 1).value, s.to_nparray())

class ValueSet():
    def __init__(self, *args):
        self.value = args
        if self._one_type:
            self._min = min(self.value)
            self._max = max(self.value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value:
            self._scalable = True
            self._enumerable = True
            self._one_type = True
            temp = []
            first_type = type(value[0])
            for val in value:
                if not isinstance(val, first_type):
                    self._one_type = False
                if not isinstance(val, int):
                    if isinstance(val, float):
                        self._enumerable = False
                    else:
                        self._scalable = False
                temp.append(val)
            self._value = tuple(temp)
        else:
            self._scalable = False
            self._enumerable = False
            self._one_type = False
            self._value = tuple([])

    @property
    def max(self):
        if self._one_type:
            return self._max
        else:
            return None

    @max.setter
    def max(self, value):
        if not self._one_type:
            raise TypeError('Mixed data doesn\'t have max!')
        if value < max(self.value):
            raise ValueError('The provided value is less than the biggest number I have.')
        self._max = value
        return value

    @property
    def min(self):
        if self._one_type:
            return self._min
        else:
            return None

    @min.setter
    def min(self, value):
        if not self._one_type:
            raise TypeError('Mixed data doesn\'t have min!')
        if value > min(self.value):
            raise ValueError('The provided value is greater than the smallest number I have.')
        self._min = value
        return value

    def to_list(self):
        return list(self._value)

    def to_nparray(self):
        return np.array(list(self._value))

    def binary_representation(self):
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