# -*- coding: utf-8 -*-
'''
BaseAgent class
===============

The base class of all `agent` classes.
'''

import random
from collections.abc import Generator
from typing import Any, Literal, TypeVar

from reil.datatypes import History, Observation
from reil.datatypes.feature import FeatureGeneratorType, FeatureSet
from reil.stateful import Stateful

T = TypeVar('T')


class BaseAgent(Stateful):
    '''
    The base class of all `agent` classes. This class does not support any
    `learner`.
    '''

    def __init__(
            self,
            tie_breaker: Literal['first', 'last', 'random'] = 'random',
            variable_action_count: bool = True,
            **kwargs: Any):
        '''
        Arguments
        ---------
        tie_breaker:
            How to choose the `action` if more than one is candidate
            to be chosen. If `first` is chosen, the first candidate is
            chosen. If `last` is chosen, the last candidate is chosen. If
            `random` is chosen, a random candidate is chosen.

        variable_action_count:
            Does this `agent` can accept a variable number of `actions`? For
            Q-learning, for example, the number of actions can vary at each
            decision point. For Policy Gradient methods, however, the number
            of actions to choose from should be fixed.

        Raises
        ------
        ValueError:
            `tie_breaker` is not one of 'first', 'last', and 'random'.
        '''
        super().__init__(**kwargs)

        self._variable_action_count = variable_action_count

        self._training_trigger: Literal[
            'none', 'termination', 'state', 'action', 'reward'] = 'none'

        if tie_breaker not in ['first', 'last', 'random']:
            raise ValueError(
                'Tie breaker should be one of first, last, or random options.')
        self._tie_breaker: Literal['first', 'last', 'random'] = tie_breaker

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
            The state for which the action should be returned.

        actions:
            The set of possible actions to choose from.

        iteration:
            The iteration in which the agent is acting.

        Returns
        -------
        :
            The action
        '''
        possible_actions: tuple[FeatureSet, ...]
        query = (
            'return feature exclusive' if self._variable_action_count
            else 'return feature')
        try:
            possible_actions = tuple(actions.send(query))
        except AttributeError:
            possible_actions = actions  # type: ignore

        try:
            result = self.best_actions(state, possible_actions)
        except NotImplementedError:
            result = possible_actions

        len_result = len(result)
        if len_result == 0:
            raise ValueError('No possible actions available to select from.')
        if len_result > 1:
            action = self._break_tie(result, self._tie_breaker)
        else:
            action = result[0]

        return action

    def best_actions(
            self, state: FeatureSet,
            actions: tuple[FeatureSet, ...]
    ) -> tuple[FeatureSet, ...]:
        '''
        Find the best `action`s for the given `state`.

        Arguments
        ---------
        state:
            The state for which the action should be returned.

        actions:
            The set of possible actions to choose from.

        Returns
        -------
        :
            A list of best actions.
        '''
        raise NotImplementedError

    def observe(
            self, subject_id: int, stat_name: str | None
    ) -> Generator[FeatureSet | None, dict[str, Any], None]:
        '''
        Create a generator to interact with the subject (`subject_id`).

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

        history: History = []
        new_observation = None
        while True:
            try:
                new_observation = Observation()
                temp: dict[str, Any] = yield
                state: FeatureSet = temp['state']
                actions: FeatureGeneratorType | None = temp['possible_actions']
                iteration: int = temp['iteration']

                new_observation.state = state
                if actions is not None:
                    new_observation.action = self.act(
                        state=state, subject_id=subject_id,
                        actions=actions, iteration=iteration)

                    # BaseAgent do not consider `Lookahead` data, because it
                    # has no learning mechanism.
                    new_observation.action_taken = (
                        yield new_observation.action)['action_taken']

                    new_observation.reward = (
                        yield new_observation.action)['reward']

                    history.append(new_observation)
                else:  # No actions to take, so skip the reward.
                    yield

            except GeneratorExit:
                if new_observation is None:
                    new_observation = Observation()
                if new_observation.reward is None:  # terminated early!
                    history.append(new_observation)

                if stat_name is not None:
                    self.statistic.append(stat_name, subject_id)

                return

    def learn_and_record(
            self, history: History, stat_name: str | None,
            subject_id: int) -> dict[str, float]:
        '''End-of-trajectory hook: bundle stat-append + reset.

        Mirrors the four side-effects of `observe()`'s GeneratorExit branch
        (learn → TB write → stat.append → reset) for the simple BaseAgent
        case where there is no learner. `Agent` overrides this to add the
        learn + TB-write steps.

        Phase B's parallel-rollout flow calls this in the MAIN process for
        each history returned by workers, so the per-patient observer
        side-effects (TB metrics x-axis advance, LR schedule advance,
        statistic.append) all still fire — only the rollout itself moved
        to a worker. See `reil.environments.parallel_rollout`.
        '''
        if stat_name is not None:
            self.statistic.append(stat_name, subject_id)
        self.reset()
        return {}

    def collect_trajectory(
            self, subject: Any, agent_id: int, subject_id: int,
            protocol: Any, iteration: int = 0) -> History:
        '''Drive one `subject` to termination with current weights; return History.

        Flat (no-generator) alternative to `observe()` for use cases that
        need a plain Python rollout — most importantly, parallel workers
        that collect trajectories with a frozen policy snapshot, ship them
        back, and let the main process apply `learn()` sequentially
        (synchronous parallel PPO; see
        `reil.environments.parallel_rollout`).

        Behaviour matches the per-step accumulation in
        `Single.interact` + `observe` for the simple
        (no-lookahead, single-agent) case: each `Observation` records
        (state, possible_actions, action, action_taken, reward) where
        `reward` is the value `subject.reward()` returns AFTER the action
        has been applied (collected on the next loop iteration). The
        in-flight observation at termination is appended even if its
        reward never arrived — matches `observe`'s GeneratorExit branch.

        Action sampling vs. argmax is still gated by `_training_trigger`
        through `act()` (in `Agent` subclasses), so callers wanting on-
        policy sampling must keep the trigger at a non-`'none'` value for
        the duration of the call. This method does NOT call `learn()` —
        the caller is responsible for that.
        '''
        state_name = protocol.state_name
        action_name = protocol.action_name
        reward_name = protocol.reward_name

        history: History = []
        pending: Observation | None = None

        while True:
            reward = subject.reward(name=reward_name, _id=agent_id)
            if pending is not None:
                pending.reward = reward
                history.append(pending)
                pending = None

            if subject.is_terminated(None):
                break

            state = subject.state(name=state_name, _id=agent_id)
            possible_actions = subject.possible_actions(
                name=action_name, _id=agent_id)
            if not possible_actions:
                break
            try:
                next(possible_actions)
            except TypeError:
                pass

            new_obs = Observation()
            new_obs.state = state
            new_obs.possible_actions = possible_actions
            new_obs.action = self.act(
                state=state, subject_id=subject_id,
                actions=possible_actions, iteration=iteration)
            new_obs.action_taken = subject.take_effect(
                new_obs.action, agent_id)
            pending = new_obs

        if pending is not None:
            history.append(pending)

        return history

    @staticmethod
    def _break_tie(
            input_tuple: tuple[T, ...],
            method: Literal['first', 'last', 'random']) -> T:
        '''
        Choose one item from the supplied list of options, based on the method.

        Arguments
        ---------
        input_tuple:
            The set of options to choose from.

        method:
            Method of choosing an item from `input_tuple`.

        Returns
        -------
        :
            One of the items from the list


        :meta public:
        '''
        if method == 'first':
            action = input_tuple[0]
        elif method == 'last':
            action = input_tuple[-1]
        else:
            action = random.choice(input_tuple)

        return action

    def get_parameters(self) -> Any:
        return None

    def set_parameters(self, parameters: Any):
        pass
