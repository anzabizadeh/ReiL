# -*- coding: utf-8 -*-
'''
Agent class
===========

This `agent` class is the base class of all agent classes that can learn from
`history`.
'''

from collections.abc import Generator
from typing import Any, Generic, Literal, TypeVar

import scipy.signal

from reil.agents.base_agent import BaseAgent
from reil.datatypes import History, Observation
from reil.datatypes.feature import FeatureGeneratorType, FeatureSet
from reil.learners.learner import LabelType, Learner, LearnerProtocol
from reil.utils.exploration_strategies import (ConstantEpsilonGreedy,
                                               ExplorationStrategy)
from reil.utils.tf_utils import reset_metric

InputType = TypeVar('InputType')
TrainingData = tuple[
    tuple[InputType, ...], tuple[LabelType, ...], dict[str, Any]]


class Agent(BaseAgent, Generic[InputType, LabelType]):
    '''
    The base class of all agent classes that learn from history.
    '''

    def __init__(
            self,
            learner: LearnerProtocol[InputType, LabelType],
            exploration_strategy: float | ExplorationStrategy,
            discount_factor: float = 1.0,
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

        super().__init__(tie_breaker, **kwargs)

        self._learner: LearnerProtocol[InputType, LabelType] = learner
        if not 0.0 <= discount_factor <= 1.0:
            self._logger.warning(
                f'{self.__class__.__qualname__} discount_factor should be in'
                f' [0.0, 1.0]. Got {discount_factor}. Set to 1.0.')
        self._discount_factor = min(discount_factor, 1.0)
        if isinstance(exploration_strategy, (float, int)):
            self._exploration_strategy = ConstantEpsilonGreedy(
                exploration_strategy)
        else:
            self._exploration_strategy = exploration_strategy

        self._training_trigger: Literal[
            'none', 'termination', 'state', 'action', 'reward'
        ] = training_trigger

    @classmethod
    def _empty_instance(cls):
        '''
        Return an empty instance of the agent.

        Returns
        -------
        :
            an empty instance of the agent.
        '''
        return cls(Learner._empty_instance(), ConstantEpsilonGreedy())  # type: ignore

    @staticmethod
    def discounted_cum_sum(r: list[float], discount: float) -> list[float]:
        '''
        Compute the discounted cumulative sum of a list of rewards.

        Arguments
        ---------
        r:
            a list of rewards.

        discount:
            the discount factor.

        Returns
        -------
        :
            the discounted cumulative sum of the rewards.
        '''
        # Copied from OpenAI SpinUp: algos/tf1/ppo/core.py
        return scipy.signal.lfilter(  # type: ignore
            [1], [1, float(-discount)], r[::-1], axis=0)[::-1]

    @staticmethod
    def extract_reward(
            history: History,
            min_clip: float | None = None,
            max_clip: float | None = None) -> list[float]:
        '''
        Extract rewards from history.

        Arguments
        ---------
        history:
            a `History` object from which the `agent` learns.

        min_clip:
            clip the rewards to be at least `min_clip`.

        max_clip:
            clip the rewards to be at most

        Returns
        -------
        :
            a list of rewards.
        '''
        rewards: list[float] = [
            a.reward for a in history
            if a.reward is not None]

        if min_clip is not None:
            rewards = [max(r, min_clip) for r in rewards]
        if max_clip is not None:
            rewards = [min(r, max_clip) for r in rewards]

        return rewards

    @staticmethod
    def get_active_history(history: History) -> History:
        '''
        Extract active history.

        When history is one complete trajectory, the last observation
        contains only the terminal state. In this case, we don't have an
        action and a reward for the last observation, so we should clip it in
        Q-learning, actor critic, and similar methods.

        Arguments
        ---------
        history:
            a `History` object from which the `agent` learns.

        Returns
        -------
        :
            the active history.
        '''
        for h in history[:-1]:
            if h.state is None or (
                    h.action is None and h.action_taken is None):
                raise ValueError(f'state and action cannot be None.\n{h}')

        if history[-1].action_taken is None and history[-1].action is None:
            active_history = history[:-1]
        else:
            active_history = history

        return active_history

    def act(self,
            state: FeatureSet,
            subject_id: int,
            actions: FeatureGeneratorType,
            iteration: int = 0) -> FeatureSet:
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

        if (
                self._training_trigger != 'none' and
                self._exploration_strategy.explore(iteration)):
            action = actions.send('choose feature exclusive')
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

    def _prepare_training(
            self, history: History) -> TrainingData[InputType, LabelType]:
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

    def learn(self, history: History) -> dict[str, float]:
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
        training_data: TrainingData[Any, Any] = (), (), {}
        # if history is not None:
        training_data = self._prepare_training(history)

        X, Y, kwargs = training_data
        metrics: dict[str, float] = {}
        if Y:
            for name, m in self._metrics.items():
                metrics[name] = m.result()
                reset_metric(m)

            metrics.update(self._learner.learn(X, Y, **kwargs))

        return metrics

    def learn_and_record(
            self, history: History, stat_name: str | None,
            subject_id: int,
            learned_in_trajectory: bool = False,
            fire_termination_learn: bool = True) -> dict[str, float]:
        '''Trajectory-end hook: learn → TB write → stat.append → reset.

        Extends `BaseAgent.learn_and_record` by adding the learn() call
        and TensorBoard write. Used both by `observe()`'s GeneratorExit
        branch (the original serial path) and by the parallel-rollout
        main loop (Phase B), so per-patient TB metrics, the LR-schedule
        advance (via learner.reset()), and statistic.append all fire in
        exactly the same order regardless of how the trajectory was
        collected.

        Arguments
        ---------
        history:
            Per-patient trajectory to learn from.
        stat_name:
            Subject statistic name to append (matches the observer's
            `stat_name` parameter).
        subject_id:
            Subject id for the statistic append.
        learned_in_trajectory:
            Did any earlier learn() call this trajectory return metrics?
            Used by the TB-write gate to avoid re-emitting stale
            `_computed_metrics` at the frozen learner iteration when no
            actual learning happened (matches the original observer
            logic; see warfarin_ppo_agent.py for context).
        fire_termination_learn:
            If True (the default), call `self.learn(history)` here. False
            corresponds to `_training_trigger != 'termination'`, where
            the learn calls happened earlier in `observe()` and this hook
            just needs to do the TB write + stat + reset.
        '''
        if fire_termination_learn:
            self._computed_metrics = self.learn(history)
            learned_in_trajectory = (
                learned_in_trajectory or bool(self._computed_metrics))

        if self._summary_writer and learned_in_trajectory:
            self._summary_writer.write(
                self._computed_metrics,
                self._learner._iteration)  # type: ignore

        # super().learn_and_record handles stat.append + self.reset() so
        # the BaseAgent contract stays the single source of truth for
        # those two steps.
        super().learn_and_record(history, stat_name, subject_id)
        return self._computed_metrics

    def observe(  # noqa: C901
            self, subject_id: int, stat_name: str | None,
    ) -> Generator[FeatureSet | None, dict[str, Any], None]:
        '''
        Create a generator to interact with the subject (`subject_id`).
        Extends `BaseAgent.observe`.

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
        # Tracks whether `learn()` produced any new metrics during this
        # trajectory. Used to gate the end-of-trajectory TB write so test
        # passes (trigger='none', no learning) don't re-emit stale
        # `_computed_metrics` at the frozen learner iteration. See discussion
        # in warfarin_ppo_agent.py for the warfarin-side counterpart.
        learned_this_trajectory = False
        while True:
            try:
                new_observation = Observation()
                temp: dict[str, Any] = yield
                state: FeatureSet = temp['state']
                possible_actions: FeatureGeneratorType | None = temp['possible_actions']
                iteration: int = temp['iteration']

                new_observation.state = state
                new_observation.possible_actions = possible_actions
                if learn_on_state:
                    m = self.learn([history[-1], new_observation])
                    self._computed_metrics.update(m)
                    learned_this_trajectory = learned_this_trajectory or bool(m)

                if possible_actions is not None:
                    new_observation.action = self.act(
                        state=state, subject_id=subject_id,
                        actions=possible_actions, iteration=iteration)

                    temp = yield new_observation.action

                    new_observation.action_taken = temp['action_taken']
                    new_observation.lookahead = temp.get('lookahead')

                    if learn_on_action:
                        m = self.learn([history[-1], new_observation])
                        self._computed_metrics.update(m)
                        learned_this_trajectory = learned_this_trajectory or bool(m)

                    new_observation.reward = (yield None)['reward']

                    history.append(new_observation)

                    if learn_on_reward:
                        m = self.learn(history[-2:])
                        self._computed_metrics.update(m)
                        learned_this_trajectory = learned_this_trajectory or bool(m)
                else:  # No actions to take, so skip the reward.
                    yield

            except GeneratorExit:
                if new_observation is None:
                    new_observation = Observation()
                if new_observation.reward is None:  # terminated early!
                    history.append(new_observation)

                # The TB-write gate depends on whether ANY learn() call
                # produced metrics during this trajectory (in-trajectory
                # state/action/reward triggers OR the terminal one). We
                # pass that signal into learn_and_record so the
                # termination-path logic can compose with the prior
                # in-trajectory signal correctly.
                self.learn_and_record(
                    history, stat_name, subject_id,
                    learned_in_trajectory=learned_this_trajectory,
                    fire_termination_learn=learn_on_termination)

                return

    def get_parameters(self) -> Any:
        return self._learner.get_parameters()

    def set_parameters(self, parameters: Any):
        self._learner.set_parameters(parameters)
