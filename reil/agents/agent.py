# -*- coding: utf-8 -*-
'''
Agent class
=============

This `agent` class is the base class of all agent classes. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import pathlib
import random
from reil.utils import ExplorationStrategy
from reil.learners.learner import Learner
from reil import rldata, utils
from typing import Any, List, Optional, Tuple, Union

from reil import rlbase


class Agent(rlbase.RLBase):
    '''
    The base class of all agent classes.

    Attributes
    ----------
    status: return the status of the agent

    Methods
    -------
    act: return an action based on the given state.

    learn: learn using either history or action, reward, and state.

    reset: reset the agent.
    '''

    def __init__(self,
                 learner: Learner,
                 exploration_strategy: ExplorationStrategy,
                 discount_factor: float = 1.0,
                 default_actions: Tuple[rldata.RLData, ...] = (),
                 **kwargs: Any):

        super().__init__(**kwargs)

        self._learner = learner
        if not 0.0 <= discount_factor <= 1.0:
            self._logger.warning(f'{self.__class__.__qualname__} discount_factor'
                f' should be in [0.0, 1.0]. Got {discount_factor}. Set to 1.0.')

        self._discount_factor = min(discount_factor, 1.0)

        self._exploration_strategy = exploration_strategy
        self._default_actions = default_actions
        self._normalized_action_list = [action.normalized.flatten()
                                        for action in self._default_actions]

        self.training_mode: bool = kwargs.get('training_mode', False)

    @property
    def status(self) -> str:
        '''Return the status of the agent as 'training' or 'test'.'''
        if self.training_mode:
            return 'training'
        else:
            return 'test'

    @status.setter
    def status(self, value: str) -> None:
        '''
        Set the status of the agent as 'training' or 'test'.
        '''
        self.training_mode = (value == 'training')

    def act(self,
            state: rldata.RLData,
            actions: Optional[List[rldata.RLData]] = None,
            episode: int = 0) -> rldata.RLData:
        '''
        Return an action based on the given state.

        Arguments
        ---------
            state: the state for which the action should be returned.
            actions: the set of possible actions to choose from.

        Note: If state is 'training' (_training_flag=false), then this function should not return any random move due to exploration.
        '''
        possible_actions = utils.get_argument(actions, self._default_actions)

        if self.training_mode and self._exploration_strategy.explore(episode):
            result = possible_actions
        else:
            result = self.best_actions(state, actions)

        action = random.choice(result)

        return action

    def learn(self, history: Optional[rlbase.History] = None,
              observation: Optional[rlbase.Observation] = None) -> None:
        '''Learn using either history or action, reward, and state.'''
        if not self.training_mode:
            raise ValueError('Not in training mode!')

        if history is not None:
            X, Y = self._prepare_training_from_history(history)
        elif observation is not None:
            X, Y = self._prepare_training_from_observation(observation)
        else:
            X, Y = [], []

        self._learner.learn(X, Y)

    def best_actions(self,
                     state: rldata.RLData,
                     actions: Optional[List[rldata.RLData]] = None) -> Tuple[rldata.RLData, ...]:
        raise NotImplementedError

    def reset(self):
        '''Reset the agent at the end of a learning episode.'''
        if self.training_mode:
            self._learner.reset()

    def load(self, filename: str, path: Optional[Union[str, pathlib.Path]] = None) -> None:
        '''
        Load an object from a file.

        Arguments
        ---------
            filename: the name of the file to be loaded.

        Note: tensorflow part is saved in filename.tf folder

        Raises ValueError if the filename is not specified.
        '''
        super().load(filename, path)

        # when loading, self._learner is the object type, not an instance.
        self._learner = self._learner.from_pickle(filename, path)

    def save(self,
             filename: Optional[str] = None,
             path: Optional[Union[str, pathlib.Path]] = None,
             data_to_save: Optional[List[str]] = None) -> Tuple[pathlib.Path, str]:
        '''
        Save the object to a file.

        Arguments
        ---------
            filename: the name of the file to be saved.

        '''
        pickle_data = utils.get_argument(data_to_save, self.__dict__)
        save_learner = '_learner' in pickle_data
        if save_learner:
            pickle_data['_learner'] = type(self._learner)

        _path, _filename = super().save(filename, path, data_to_save=pickle_data)

        if save_learner:
            _path, _filename = self._learner.save(_filename, _path / 'learner')

        return _path, _filename

    def _prepare_training_from_history(self,
                                       history: rlbase.History) -> Tuple[List[rldata.RLData], List[float]]:
        raise NotImplementedError

    def _prepare_training_from_observation(self,
                                           observation: rlbase.Observation) -> Tuple[List[rldata.RLData], List[float]]:
        raise NotImplementedError
