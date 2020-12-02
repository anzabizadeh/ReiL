# -*- coding: utf-8 -*-
'''
Agent class
============

This `agent` class is the base class of all agent classes that can learn from
`history`. 

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

import pathlib
from typing import Any, List, Optional, Tuple, Union

from reil import agents, stateful
from reil.datatypes.reildata import ReilData
from reil.learners.learner import Learner
from reil.utils import functions
from reil.utils.exploration_strategies import ExplorationStrategy
from typing_extensions import Literal

TrainingData = Tuple[List[ReilData], List[float]]


class Agent(agents.NoLearnAgent):
    '''
    The base class of all agent classes that learn from history.

    ### Methods
    learn: learns using history, which consists of state, action, rewards.

    _prepare_training: converts a history into X and y vectors for training.
    '''

    def __init__(self,
                 learner: Learner,
                 exploration_strategy: ExplorationStrategy,
                 discount_factor: float = 1.0,
                 default_actions: Tuple[ReilData, ...] = (),
                 training_mode: bool = False,
                 tie_breaker: Literal['first', 'last', 'random'] = 'random',
                 **kwargs: Any):
        '''
        Initializes the `agent`.

        ### Arguments
        learner: the `Learner` object that does the learning.

        exploration_strategy: an `ExplorationStrategy` object that determines
        whether the `action` should be exploratory or not for a given `state`
        at a given `epoch`.

        discount_factor: by what factor should future rewards be discounted?

        default_actions: a tuple of default actions.

        training_mode: whether the agent is in training mode or not.

        tie_breaker: how to choose the `action` if more than one is candidate
        to be chosen.
        '''
        super().__init__(default_actions, tie_breaker, **kwargs)

        self.training_mode = training_mode
        self._learner = learner
        if not 0.0 <= discount_factor <= 1.0:
            self._logger.warning(
                f'{self.__class__.__qualname__} discount_factor'
                f' should be in [0.0, 1.0]. Got {discount_factor}. Set to 1.0.')
        self._discount_factor = min(discount_factor, 1.0)
        self._exploration_strategy = exploration_strategy

    def act(self,
            state: ReilData,
            actions: Optional[Tuple[ReilData, ...]] = None,
            epoch: int = 0) -> ReilData:
        '''
        Return an action based on the given state.

        ### Arguments
        state: the state for which the action should be returned.

        actions: the set of possible actions to choose from.

        epoch: the epoch in which the agent is acting.
        '''
        if self.training_mode and self._exploration_strategy.explore(epoch):
            possible_actions = functions.get_argument(
                actions, self._default_actions)
            action = self._break_tie(
                possible_actions, self._tie_breaker)  # type: ignore
        else:
            action = super().act(state, actions, epoch)

        return action

    def learn(self, history: stateful.History) -> None:
        '''
        Learn using history.

        ### Arguments
        history: a `History` object from which the `agent` learns.
        '''
        if not self.training_mode:
            raise ValueError('Not in training mode!')

        if history is not None:
            X, Y = self._prepare_training(history)
        else:
            X, Y = [], []

        self._learner.learn(X, Y)

    def reset(self):
        '''Reset the agent at the end of a learning epoch.'''
        super().reset()
        if self.training_mode:
            self._learner.reset()

    def load(self, filename: str,
             path: Optional[Union[str, pathlib.Path]] = None) -> None:
        '''
        Load an object from a file.

        ### Arguments
        filename: the name of the file to be loaded.

        path: the path in which the file is saved.

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

        ### Arguments
        filename: the name of the file to be saved.

        path: the path in which the file should be saved.

        data_to_save: a list of variables that should be pickled. If omitted,
            the `agent` is saved completely.
        '''
        pickle_data = functions.get_argument(data_to_save, self.__dict__)
        save_learner = '_learner' in pickle_data
        if save_learner:
            pickle_data['_learner'] = type(self._learner)

        _path, _filename = super().save(filename, path, data_to_save=pickle_data)

        if save_learner:
            self._learner.save(_filename, _path / 'learner')

        return _path, _filename

    def _prepare_training(self,
                          history: stateful.History) -> TrainingData:
        '''
        Uses `history` to create the training set in the form of `X` and `y`
        vectors.

        ### Arguments
        history: a `History` object from which the `agent` learns.
        '''
        raise NotImplementedError
