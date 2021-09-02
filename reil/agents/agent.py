# -*- coding: utf-8 -*-
'''
Agent class
===========

This `agent` class is the base class of all agent classes that can learn from
`history`.
'''

from typing import (Any, Dict, Generator, Generic, Literal, Optional, Tuple,
                    Union)

from reil.agents.agent_base import AgentBase
from reil.datatypes import History
from reil.datatypes.dataclasses import Observation
from reil.datatypes.feature import FeatureArray
from reil.learners.learner import LabelType, Learner
from reil.utils.exploration_strategies import (ConstantEpsilonGreedy,
                                               ExplorationStrategy)

TrainingData = Tuple[Tuple[FeatureArray, ...], Tuple[LabelType, ...]]


class Agent(AgentBase, Generic[LabelType]):
    '''
    The base class of all agent classes that learn from history.
    '''

    def __init__(
            self,
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
            `state` at a given `iteration`.

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
        self._training_trigger: Literal[
            'none', 'termination', 'state', 'action', 'reward'
            ] = training_trigger

    @classmethod
    def _empty_instance(cls):
        return cls(Learner._empty_instance(), ConstantEpsilonGreedy())

    def act(self,
            state: FeatureArray,
            subject_id: int,
            actions: Optional[Tuple[FeatureArray, ...]] = None,
            iteration: int = 0) -> FeatureArray:
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

        iteration:
            the iteration in which the agent is acting.

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
                self._exploration_strategy.explore(iteration)):
            possible_actions = actions or self._default_actions
            action = self._break_tie(
                possible_actions, self._tie_breaker)
        else:
            action = super().act(
                state=state, subject_id=subject_id,
                actions=actions, iteration=iteration)

        return action

    def reset(self):
        '''Reset the agent at the end of a learning iteration.'''
        super().reset()
        if self._training_trigger != 'none':
            self._learner.reset()

    # def load(
    #         self, filename: str,
    #         path: Optional[Union[str, pathlib.PurePath]] = None) -> None:
    #     '''
    #     Load an object from a file.

    #     Arguments
    #     ---------
    #     filename:
    #         the name of the file to be loaded.

    #     path:
    #         the path in which the file is saved.

    #     Raises
    #     ------
    #         ValueError
    #             Filename is not specified.
    #     '''
    #     _path = pathlib.Path(path or self._path)
    #     super().load(filename, _path)

    #     # when loading, self._learner is the object type, not an instance.
    #     self._learner = self._learner.from_pickle(  # type: ignore
    #         filename, _path.parent / 'learner')

    # def save(
    #         self,
    #         filename: Optional[str] = None,
    #         path: Optional[Union[str, pathlib.PurePath]] = None
    # ) -> pathlib.PurePath:
    #     '''
    #     Save the object to a file.

    #     Arguments
    #     ---------
    #     filename:
    #         the name of the file to be saved.

    #     path:
    #         the path in which the file should be saved.

    #     data_to_save:
    #         a list of variables that should be pickled. If omitted,
    #         the `agent` is saved completely.

    #     Returns
    #     -------
    #     :
    #         a `Path` object to the location of the saved file and its name as
    #         `str`
    #     '''
    #     full_path = super().save(filename, path)
    #     self._learner.save(full_path.name, full_path.parent / 'learner')

    #     return full_path

    def _prepare_training(self, history: History) -> TrainingData[LabelType]:
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

    def learn(self, history: History) -> None:
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
        training_data: TrainingData[Any] = (), ()
        if history is not None:
            training_data = self._prepare_training(history)

        X, Y = training_data
        if X:
            self._learner.learn(X, Y)

    def observe(  # noqa: C901
            self, subject_id: int, stat_name: Optional[str],
    ) -> Generator[Union[FeatureArray, None], Dict[str, Any], None]:
        '''
        Create a generator to interact with the subject (`subject_id`).
        Extends `AgentBase.observe`.

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

        trigger = self._training_trigger
        learn_on_state = trigger == 'state'
        learn_on_action = trigger == 'action'
        learn_on_reward = trigger == 'reward'
        learn_on_termination = trigger == 'termination'

        history: History = []
        new_observation = None
        while True:
            try:
                new_observation = Observation()
                temp: Dict[str, Any] = yield
                state: FeatureArray = temp['state']
                actions: Tuple[FeatureArray, ...] = temp['actions']
                iteration: int = temp['iteration']

                new_observation.state = state
                if learn_on_state:
                    self.learn([history[-1], new_observation])

                if actions is not None:
                    new_observation.action = self.act(
                        state=state, subject_id=subject_id,
                        actions=actions, iteration=iteration)

                    if learn_on_action:
                        self.learn([history[-1], new_observation])

                    new_observation.reward = (
                        yield new_observation.action)['reward']

                    history.append(new_observation)

                    if learn_on_reward:
                        self.learn(history[-2:])
                else:  # No actions to take, so skip the reward.
                    yield

            except GeneratorExit:
                if new_observation is None:
                    new_observation = Observation()
                if new_observation.reward is None:  # terminated early!
                    history.append(new_observation)

                if learn_on_termination:
                    self.learn(history)

                if stat_name is not None:
                    self.statistic.append(stat_name, subject_id)

                return

    # def __getstate__(self):
    #     state = super().__getstate__()

    #     state['_learner'] = type(self._learner)

    #     return state
