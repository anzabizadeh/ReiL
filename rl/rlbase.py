# -*- coding: utf-8 -*-
'''
RLBase class
=================

The base class for reinforcement learning

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from dill import load, dump, HIGHEST_PROTOCOL
from random import randrange
import pathlib
import os
from time import sleep

from .data_collector import DataCollector


class RLBase():
    '''
    Super class of all classes in rl package.
    
    Attributes
    ----------
        data_collector: a DataCollector object

    Methods
    -------
        set_params: set parameters.
        set_defaults: set default values for parameters.
        load: load an object from a file.
        save: save the object to a file.
    '''
    def __init__(self, **kwargs):
        self._defaults = {}
        self.data_collector = DataCollector(object=self)
        self.set_defaults(name=self.__repr__() + ' - {:0<7}'.format(str(randrange(1, 1000000))), version=0.3, path='.')
        self.set_params(**kwargs)

        if False: self._name, self._version, self._path = [], [], []

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
            path: the path of the file to be loaded. (Default='.')

        Raises ValueError if the filename is not specified.
        '''
        try:  # filename
            filename = kwargs['filename']
        except KeyError:
            raise ValueError('name of the output file not specified.')
        path = kwargs.get('path', self._path)

        with open(os.path.join(path, filename + '.pkl'), 'rb') as f:
            try:
                data = load(f)
            except EOFError:
                try:
                    sleep(5)
                    data = load(f)
                except EOFError:
                    raise RuntimeError('Corrupted data file: '+filename)
            for key, value in data.items():
                self.__dict__[key] = value
            self.data_collector._object = self

    def save(self, **kwargs):
        '''
        Save the object to a file.

        Arguments
        ---------
            filename: the name of the object (Default=self._name)
            path: the path of the file to be loaded. (Default='.')
            data: what to save (Default: saves everything)
        '''

        filename = kwargs.get('filename', self._name)
        path = kwargs.get('path', self._path)
        try:  # data
            data = {}
            for d in kwargs['data']:
                data[d] = self.__dict__[d]
            for key in ('_name', '_version', '_path'):  # these should be saved automatically
                data[key] = self.__dict__[key]
        except KeyError:
            data = self.__dict__

        pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
        with open(os.path.join(path, filename + '.pkl'), 'wb+') as f:
            dump(data, f, HIGHEST_PROTOCOL)

        return path, filename

    def _report(self, **kwargs):
        return

    def __repr__(self):
        return 'RLBase'
