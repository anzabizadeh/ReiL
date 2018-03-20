# -*- coding: utf-8 -*-
'''
DataCollector class
=================

The base class for data collection in reinforcement learning

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from pickle import load, dump, HIGHEST_PROTOCOL
from copy import deepcopy


def main():
    class test():
        def __init__(self):
            self.a = 1
    t = test()
    dc = DataCollector(object=t)
    dc.available_statistics = {'stat 1': [True, lambda new, old, statistic: new['a']-old['a'], 'a']}
    dc.active_statistics = ['stat 1']
    dc.start()
    dc.collect()
    t.a = 10
    print(dc.retrieve())
    print(dc.report())
    print(dc.report())
    dc.stop(flush=True)


class DataCollector():
    '''
    A class to collect data during reinforcement learning experiment.
    
    Attributes
    ----------
        is_active: whether the data collector is active.
        available_statistics: a list of available statistics (set/get)
        active_statistics: a list of active statistics (set/get)

    Methods
    -------
        start: start data collection
        stop: stop data collection
        collect: collect data for all or the specified statistics
        retrieve: return all or the specified recorded data
        report: create a report on all or the specified statistics
        load: load collected data from a file.
        save: save the collected data to a file.
        set_params: set parameters.
        set_defaults: set default values for parameters.
    '''
    def __init__(self, **kwargs):
        self.set_defaults(available_statistics={}, active_statistics={}, data={}, is_active=False, object=None)
        self.set_params(**kwargs)

        if False:
            self._available_statistics, self._active_statistics, self._data = {}, {}, {}
            self._is_active, self._object = False, None

    @property
    def is_active(self):
        ''' Return the status of the data collector.'''
        return self._is_active

    @property
    def available_statistics(self):
        '''Return the dictionary of available statistics.'''
        return self._available_statistics

    @available_statistics.setter
    def available_statistics(self, statistics):
        '''
        Set the dictionary of available statistics.

        The dictionary should be in {statistic's name: delta flag, function, parameters}
            statistic's name: the name of the statistic, e.g. 'time elapsed', 'diffQ'.
            delta flag: True if old and new data are both required to calculate the statistic,
                        False otherwise.
            function: a reference to the function that calculates the statistic. Note that for
                        statistics with deta flag=True, the function gets 3 arguments: old, new, statistic
                        and for delta flag=False, 2 arguments will be passed: data and statistic
            
        Raises exceptions if data collector is not started or if the expected data is not provided.
        '''
        if self._is_active:
            raise RuntimeError('Data collector should be stopped before assignment.')
        for s, args in statistics.items():
            if not isinstance(s, str):
                raise TypeError('Statistic\'s name should be of type str ({}).'.format(s))
            if not isinstance(args[0], bool):
                raise TypeError('Statistic\'s first item should be of type bool ({}).'.format(args[0]))
            if not callable(args[1]):
                raise TypeError('Statistic\'s second item should be a function ({}).'.format(args[1]))
            if len(args) < 3:
                raise ValueError('A statistic should have at least one variable name.')
            for i in range(2,len(args)):
                if not isinstance(args[i], str):
                    raise TypeError('Variable\'s name should be of type str ({}).'.format(args[i]))
        self._available_statistics = statistics

    @property
    def active_statistics(self):
        '''Return the list of active statistics.'''
        return self._available_statistics

    @active_statistics.setter
    def active_statistics(self, statistics):
        '''
        Set the list of active statistics.
            
        Raises exceptions if data collector is not started or if the expected data is not provided.
        '''
        if self._is_active:
            raise RuntimeError('Data collector should be stopped before assignment.')
        if statistics == 'all':
            self._active_statistics = self._available_statistics.keys()
            return
        for s in statistics:
            if not isinstance(s, str):
                raise TypeError('Statistic\'s name should be of type str ({}).'.format(s))
            if not s in self._available_statistics:
                raise ValueError('Requested statistic is not available ({}).'.format(s))
        self._active_statistics = statistics

    def start(self, flush=False):
        '''
        Start the data collector.

        Arguments
        ---------
            flush: if True, previously collected data is flushed.

        Raises error if available or active statistics are not set. 
        '''
        if not self._available_statistics:
            raise ValueError('Data collector has not been set up. Use available_statistics to set up.')
        if not self._active_statistics:
            raise ValueError('No statistic is defined. Use active_statistics to determine what to collect.')
        if flush:
            self._data = {}
        self._is_active = True

    def stop(self, flush=False):
        '''
        Stop the data collector.

        Arguments
        ---------
            flush: if True, the collected data is flushed.
        '''
        if flush:
            self._data = {}
        self._is_active = False

    def collect(self, **kwargs):  # data should be a dictionary
        '''
        Collect data.

        Arguments
        ---------
            statistic: a list of statistics for which the data should be collected. (Default='all')

        Raises error if the data collector is not started or the expected attribute is not available in the object.

        Note: collect only collects data for statistics whose flag is True, i.e. statistics that calculate a delta.
        '''

        if not self._is_active:
            raise RuntimeError('Data collector is not active. Use start() method.')

        if not kwargs:
            stats = self._active_statistics
        else:
            try:
                stats = kwargs['statistics']
            except KeyError:
                try:
                    stats = kwargs['statistic']
                except KeyError:
                    raise KeyError('statistics are not specified.')

        for s in stats:
            # if self._available_statistics[statistic][0]:  # if it should be collected
            self._data[s] = {}  # enabling '.' notation
            for variable in self._available_statistics[s][2:]:
                arguments = variable.split('.')
                temp = self._object
                for arg in arguments:
                    temp = deepcopy(temp.__dict__[arg])
                try:
                    self._data[s][variable] = deepcopy(temp)
                    # self._data[s] = dict((variable, self._object.__dict__[variable].copy()) 
                    #                                 for variable in self._available_statistics[s][2:])
                except AttributeError:
                    self._data[s][variable] = deepcopy(temp)
                    # self._data[s] = dict((variable, self._object.__dict__[variable]) 
                    #                                 for variable in self._available_statistics[s][2:])
                except KeyError:
                    raise RuntimeError('Object doesn\'t have the attribute provided for {}.'.format(s))

    def retrieve(self, **kwargs):
        '''
        Retrieve the requested data.

        Arguments
        ---------
            statistics: a list of statistics for which the data should be retrieved. (Default='all')

        Raises error if the data collector is not started or the expected attribute is not available in the object.
        '''
        if not self._is_active:
            raise RuntimeError('Data collector is not active. Use start() method.')

        if not kwargs:
            return self._data
        else:
            try:
                stats = kwargs['statistics']
            except KeyError:
                try:
                    stats = kwargs['statistic']
                except KeyError:
                    raise KeyError('statistics are not specified.')

        data = {}
        for stat in stats:
            try:
                data[stat] = self._data[stat]
            except KeyError:
                pass

        return data

    def report(self, **kwargs):
        '''
        Report the requested statistics.

        Arguments
        ---------
            statistic: a list of statistics for which the data should be collected. (Default='all')

        Raises error if the data collector is not started or the expected attribute is not available in the object.

        Note: report uses the provided functions to calculate the results.
        '''
        if not self._is_active:
            raise RuntimeError('Data collector is not active. Use start() method.')

        if not kwargs:
            stats = self._active_statistics
        else:
            try:
                stats = kwargs['statistics']
            except KeyError:
                try:
                    stats = kwargs['statistic']
                except KeyError:
                    raise KeyError('statistics are not specified.')

        old_data = deepcopy(self._data)
        self.collect(statistics=stats)

        results = {}
        for s in stats:
            if self._available_statistics[s][0]:  # if it should be collected
                try:
                    results[s] = self._available_statistics[s][1](new=self._data[s], old=old_data[s], statistic=s)
                except KeyError:
                    raise RuntimeError('Object doesn\'t have the attribute provided for {}.'.format(s))
            else:
                results[s] = self._available_statistics[s][1](data=self._data[s], statistic=s)
        
        self._data = old_data

        return results

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
            self.__dict__ = load(f)

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
            dump(self.__dict__, f, HIGHEST_PROTOCOL)

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


if __name__ == '__main__':
    main()