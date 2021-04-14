# -*- coding: utf-8 -*-
'''
Agent class
===========

This `agent` class is the base class of all agent classes that can learn from
`history`.
'''

import pathlib
from collections import defaultdict
from typing import Any, Dict, Generator, Generic, Optional, Tuple, Union

from reil import agents, stateful
from reil.datatypes import FeatureArray
from reil.learners.learner import Learner, LabelType
from reil.utils.exploration_strategies import ExplorationStrategy
from typing_extensions import Literal


class Agent(agents.NoLearnAgent, Generic[LabelType]):
    '''
    The base class of all agent classes that learn from history.
    '''

    def __init__(self,
                 learner: Learner[LabelType],
                 exploration_strategy: ExplorationStrategy,
                 discount_factor: float = 1.0,
                 default_actions: Tuple[FeatureArray, ...] = (),
                 tie_breaker: Literal['first', 'last', 'random'] = 'random',
                 training_trigger: Literal[
                     'none', 'termination',
                     'state', 'action', 'reward'] = 'termination',
                 **kwargs: Any):
        '''
        Arguments
        ---------
        learner:
            the `Learner` object that does the learning.

        exploration_strategy:
            an `ExplorationStrategy` object that determines
            whether the `action` should be exploratory or not for a given
            `state` at a given `epoch`.

        discount_factor:
            by what factor should future rewards be discounted?

        default_actions:
            a tuple of default actions.

        tie_breaker:
            how to choose the `action` if more than one is candidate
            to be chosen.

        training_trigger:
            When to learn from observations. This arguments is used in
            `observe` method to determine when `learn` method should be called.
            `none` avoids any call to `learn`; `state`, `action` and `reward`
            trigger the `learn` method after receiving their corresponding
            value; `termination` waits until `.close()` method of the generator
            is called.
        '''
        self._tie_breaker: Literal['first', 'last', 'random']

        super().__init__(default_actions, tie_breaker, **kwargs)

        self._learner: Learner[LabelType] = learner
        if not 0.0 <= discount_factor <= 1.0:
            self._logger.warning(
                f'{self.__class__.__qualname__} discount_factor should be in'
                f' [0.0, 1.0]. Got {discount_factor}. Set to 1.0.')
        self._discount_factor = min(discount_factor, 1.0)
        self._exploration_strategy = exploration_strategy
        self._training_trigger = training_trigger
        self._history: Dict[int, stateful.History] = defaultdict(list)

    @classmethod
    def _empty_instance(cls):
        return cls(None)  # type: ignore

    def act(self,
            state: FeatureArray,
            subject_id: int,
            actions: Optional[Tuple[FeatureArray, ...]] = None,
            epoch: int = 0) -> FeatureArray:
        '''
        Return an action based on the given state.

        Arguments
        ---------
        state:
            the state for which the action should be returned.

        subject_id:
            the ID of the `subject` on which action should occur.

        actions:
            the set of possible actions to choose from.

        epoch:
            the epoch in which the agent is acting.

        Raises
        ------
        ValueError
            Subject with `subject_id` not found.

        Returns
        -------
        :
            the action
        '''
        if subject_id not in self._entity_list:
            raise ValueError(f'Subject with ID={subject_id} not found.')

        if (self._training_trigger != 'none' and
                self._exploration_strategy.explore(epoch)):
            possible_actions = actions or self._default_actions
            action = self._break_tie(
                possible_actions, self._tie_breaker)
        else:
            action = super().act(state=state, subject_id=subject_id,
                                 actions=actions, epoch=epoch)

        return action

    def reset(self):
        '''Reset the agent at the end of a learning epoch.'''
        super().reset()
        if self._training_trigger != 'none':
            self._learner.reset()

    def load(self, filename: str,
             path: Optional[Union[str, pathlib.PurePath]] = None) -> None:
        '''
        Load an object from a file.

        Arguments
        ---------
        filename:
            the name of the file to be loaded.

        path:
            the path in which the file is saved.

        Raises
        ------
            ValueError
                Filename is not specified.
        '''
        _path = pathlib.Path(path or self._path)
        super().load(filename, _path)

        # when loading, self._learner is the object type, not an instance.
        self._learner = self._learner.from_pickle(  # type: ignore
            filename, _path / 'learner')

    def save(self,
             filename: Optional[str] = None,
             path: Optional[Union[str, pathlib.PurePath]] = None,
             data_to_save: Optional[Tuple[str, ...]] = None
             ) -> Tuple[pathlib.PurePath, str]:
        '''
        Save the object to a file.

        Arguments
        ---------
        filename:
            the name of the file to be saved.

        path:
            the path in which the file should be saved.

        data_to_save:
            a list of variables that should be pickled. If omitted,
            the `agent` is saved completely.

        Returns
        -------
        :
            a `Path` object to the location of the saved file and its name as
            `str`
        '''
        data = list(data_to_save or self.__dict__)
        save_learner = '_learner' in data
        if save_learner:
            temp, self._learner = (  # type: ignore
                self._learner, type(self._learner))

        _path = pathlib.Path(path or self._path)
        _filename = filename or self._name

        try:
            super().save(
                _filename, _path, data_to_save=tuple(data))
            if save_learner:
                self._learner = temp  # type: ignore
                self._learner.save(_filename, _path / 'learner')
        finally:
            if save_learner:
                self._learner = temp  # type: ignore

        return _path, _filename

    def _prepare_training(self,
                          history: stateful.History) -> agents.TrainingData:
        '''
        Use `history` to create the training set in the form of `X` and `y`
        vectors.

        Arguments
        ---------
        history:
            a `History` object from which the `agent` learns.

        Returns
        -------
        :
            a `TrainingData` object that contains `X` and 'y` vectors


        :meta public:
        '''
        raise NotImplementedError

    def learn(self, history: stateful.History) -> None:
        '''
        Learn using history.

        Arguments
        ---------
        subject_id:
            the ID of the `subject` whose history is being used for learning.

        next_state:
            The new `state` of the `subject` after taking `agent`'s action.
            Some methods
        '''
        if history is not None:
            X, Y = self._prepare_training(history)
        else:
            X, Y = [], []

        if X:
            self._learner.learn(X, Y)  # type: ignore

    def observe(self, subject_id: int, stat_name: Optional[str],  # noqa: C901
                ) -> Generator[Union[FeatureArray, None], Any, None]:
        '''
        Create a generator to interact with the subject (`subject_id`).
        Extends `NoLearnAgent.observe`.

        This method creates a generator for `subject_id` that
        receives `state`, yields `action` and receives `reward`
        until it is closed. When `.close()` is called on the generator,
        `statistics` are calculated.

        Arguments
        ---------
        subject_id:
            the ID of the `subject` on which action happened.

        stat_name:
            The name of the `statistic` that should be computed at the end of
            each trajectory.

        Raises
        ------
        ValueError
            Subject with `subject_id` not found.
        '''
        if subject_id not in self._entity_list:
            raise ValueError(f'Subject with ID={subject_id} not found.')

        history: stateful.History = []
        new_observation = stateful.Observation()
        while True:
            try:
                new_observation = stateful.Observation()
                temp = yield
                new_observation.state = temp['state']
                actions: Tuple[FeatureArray, ...] = temp['actions']
                epoch: int = temp['epoch']

                if self._training_trigger == 'state':
                    self.learn([history[-1], new_observation])

                if actions is not None:
                    new_observation.action = self.act(
                        state=new_observation.state,  # type: ignore
                        subject_id=subject_id,
                        actions=actions, epoch=epoch)

                    if self._training_trigger == 'action':
                        self.learn([history[-1], new_observation])

                    new_observation.reward = (yield new_observation.action)

                    history.append(new_observation)

                    if self._training_trigger == 'reward':
                        self.learn(history[-2:])
                else:
                    yield

            except GeneratorExit:
                if new_observation.reward is None:  # terminated early!
                    history.append(new_observation)

                if self._training_trigger == 'termination':
                    self.learn(history)

                if stat_name is not None:
                    self.statistic.append(stat_name, subject_id)

                return
