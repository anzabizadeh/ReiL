# -*- coding: utf-8 -*-
'''
RLBase class
=================

The base class for reinforcement learning

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import pickle


class RLBase():
    '''
    Super class of all classes in rl package.
    
    Attributes
    ----------

    Methods
    -------
        set_params: set parameters.
        load: load an object from a file.
        save: save the object to a file.
    '''
    def __init___(self):
        self.__defaults = {}

    def set_params(self, **params):
        '''
        set parameters to values.

        Arguments
        ---------
            params: a dictionary containing parameter names and their values.
        '''
        self.__dict__.update(('_'+key, params.get(key, self.__defaults[key]))
                              for key in self.__defaults if key in params)

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
