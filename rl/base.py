# -*- coding: utf-8 -*-
'''
RLBase class
=================

The base class for reinforcement learning

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import pickle

from .data_collector import DataCollector


class RLBase():
    '''
    Super class of all classes in rl package.
    
    Attributes
    ----------

    Methods
    -------
        set_params: set parameters.
        set_defaults: set default values for parameters.
        load: load an object from a file.
        save: save the object to a file.
        report: report the requested data.
    '''
    def __init__(self, **kwargs):
        self._defaults = {}
        self.data_collector = DataCollector()

    def set_params(self, **params):
        '''
        set parameters to values.

        Arguments
        ---------
            params: a dictionary containing parameter names and their values.
        '''
        self.__dict__.update(('_'+key, params.get(key, self._defaults[key]))
                              for key in self._defaults if key in params)

    def set_defaults(self, **params):
        '''
        set parameters default values.

        Arguments
        ---------
            params: a dictionary containing parameter names and their default values.

        Note: this method overwrites all variable names.
        '''
        if not hasattr(self, '_defaults'):
            self._defaults = {}
        for key, value in params.items():
            self._defaults[key] = value
            if not hasattr(self, '_'+key):
                self.__dict__['_'+key] = value

    def load(self, **kwargs):
        '''
        Load an object from a file.

        Arguments
        ---------
            filename: the name of the file to be loaded.

        Raises ValueError if the filename is not specified.
        '''
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')
        with open(filename + '.pkl', 'rb') as f:
            self.__dict__ = pickle.load(f)

    def save(self, **kwargs):
        '''
        Save the object to a file.

        Arguments
        ---------
            filename: the name of the file to be saved.

        Raises ValueError if the filename is not specified.
        '''
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')
        with open(filename + '.pkl', 'wb+') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
