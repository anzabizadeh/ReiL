# -*- coding: utf-8 -*-
'''
DataCollector class
=================

The base class for data collection in reinforcement learning

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import pathlib
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import dill
import copy


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
    '''
    def __init__(self,
        object: Any,
        available_statistics: Dict[str, Tuple[bool, Callable[[Any], Any], str]] = {},
        active_statistics: Sequence[str] = ()) -> None:

        self._available_statistics = available_statistics
        self._active_statistics = active_statistics
        self._data = {}
        self._is_active = False
        self.object = object

    @property
    def is_active(self):
        ''' Return the status of the data collector.'''
        return self._is_active

    @property
    def available_statistics(self) -> Dict[str, Tuple[bool, Callable[[Any], Any], str]]:
        '''Return the dictionary of available statistics.'''
        return self._available_statistics

    @available_statistics.setter
    def available_statistics(self,
        statistics: Dict[str, Tuple[bool, Callable[[Any], Any], str]]) -> None:
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
            if not isinstance(s, str):  # type: ignore
                raise TypeError(f'Statistic\'s name should be of type str ({s}).')
            if not isinstance(args[0], bool):  # type: ignore
                raise TypeError(f'Statistic\'s first item should be of type bool ({args[0]}).')
            if not callable(args[1]):
                raise TypeError(f'Statistic\'s second item should be a function ({args[1]}).')
            if len(args) < 3:
                raise ValueError('A statistic should have at least one variable name.')
            for i in range(2,len(args)):
                if not isinstance(args[i], str):
                    raise TypeError(f'Variable\'s name should be of type str ({args[i]}).')
        self._available_statistics = statistics

    @property
    def active_statistics(self) -> Sequence[str]:
        '''Return the list of active statistics.'''
        return self._active_statistics

    @active_statistics.setter
    def active_statistics(self, statistics: Sequence[str]):
        '''
        Set the list of active statistics.
            
        Raises exceptions if data collector is not started or if the expected data is not provided.
        '''
        if self._is_active:
            raise RuntimeError('Data collector should be stopped before assignment.')
        if statistics == 'all':
            self._active_statistics = list(self._available_statistics.keys())
            return
        for s in statistics:
            if not isinstance(s, str):  # type: ignore
                raise TypeError(f'Statistic\'s name should be of type str ({s}).')
            if not s in self._available_statistics:
                raise ValueError(f'Requested statistic is not available ({s}).')
        self._active_statistics = statistics

    def start(self, flush: bool = False) -> None:
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

    def stop(self, flush: bool = False) -> None:
        '''
        Stop the data collector.

        Arguments
        ---------
            flush: if True, the collected data is flushed.
        '''
        if flush:
            self._data = {}
        self._is_active = False

    def collect(self, **kwargs: Any) -> None:  # data should be a dictionary
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
                arguments: str = variable.split('.')
                temp = self.object
                for arg in arguments:
                    temp = copy.deepcopy(temp.__dict__[arg])
                try:
                    self._data[s][variable] = copy.deepcopy(temp)
                    # self._data[s] = dict((variable, self.object.__dict__[variable].copy()) 
                    #                                 for variable in self._available_statistics[s][2:])
                except AttributeError:
                    self._data[s][variable] = copy.deepcopy(temp)
                    # self._data[s] = dict((variable, self.object.__dict__[variable]) 
                    #                                 for variable in self._available_statistics[s][2:])
                except KeyError:
                    raise RuntimeError(f'Object doesn\'t have the attribute provided for {s}.')

    def retrieve(self, **kwargs: Any) -> None:
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

    def report(self, **kwargs: Any) -> None:
        '''
        Report the requested statistics.

        Arguments
        ---------
            statistic: a list of statistics for which the data should be collected. (Default='all')
            update_data: (Default=False)
                True: update data after report
                False: do not change old data

        Raises error if the data collector is not started or the expected attribute is not available in the object.

        Note: report uses the provided functions to calculate the results.
        '''
        if not self._is_active:
            raise RuntimeError('Data collector is not active. Use start() method.')

        if not kwargs:
            stats = self._active_statistics
            update_data = False
        else:
            try:
                stats = kwargs['statistics']
            except KeyError:
                try:
                    stats = kwargs['statistic']
                except KeyError:
                    stats = self._active_statistics
            try:
                update_data = kwargs['update_data']
            except KeyError:
                update_data = False

        old_data = copy.deepcopy(self._data)
        self.collect(statistics=stats)

        results = {}
        for s in stats:
            if self._available_statistics[s][0]:  # if it should be collected
                try:
                    results[s] = self._available_statistics[s][1](new=self._data[s], old=old_data[s], statistic=s)
                except KeyError:
                    raise RuntimeError(f'Object doesn\'t have the attribute provided for {s}.')
            else:
                results[s] = self._available_statistics[s][1](data=self._data[s], statistic=s)
        
        if not update_data:
            self._data = old_data

        return results

    def load(self, filename: str, path: Optional[str] = None) -> None:
        '''
        Load an object from a file.

        Arguments
        ---------
            filename: the name of the file to be loaded.
            path: the path of the file to be loaded. (Object's default path will be used if not provided)

        '''
        _path = pathlib.Path(path if path is not None else '.')

        with open(_path / f'{filename}.pkl', 'rb') as f:
                self.__dict__ = dill.load(f)  #type: ignore

    def save(self,
             filename: str,
             path: Optional[str] = None) -> None:
        '''
        Save the object to a file.

        Arguments
        ---------
            filename: the name of the object (Default=self._name)
            path: the path of the file to be loaded. (Default='.')
            data: what to save (Default: saves everything)
        '''

        _path: pathlib.Path = pathlib.Path(path if path is not None else '.')

        with open(_path / f'{filename}.pkl', 'wb+') as f:
            dill.dump(self.__dict__, f, dill.HIGHEST_PROTOCOL)  # type: ignore

    def __repr__(self):
        if len(self._available_statistics) == 0:
            return 'data_collector with no statistics'
        return 'data_collector with ' + '-'.join(self._available_statistics)

if __name__ == '__main__':
    main()