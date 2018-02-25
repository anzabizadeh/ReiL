# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 19:01:29 2018

@author: Sadjad


"""

def main():
    pass


class ann():
    def __init__(self, **kwargs):
        raise NotImplementedError
    
    def estimate(self, X):
        raise NotImplementedError

    def learn(self, X, y):
        raise NotImplementedError


class backpropagation_ann(ann):
    def __init__(self, **kwargs):
        raise NotImplementedError

    def add_layer(self, )

if __name__ == '__main__':
    main()