import copy
from typing import Any, Generator, Literal

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from reil.agents.agent import AgentBase, TrainingData
from reil.agents.proximal_policy_optimization import PPO, PPOLearner
from reil.datatypes.buffers.buffer import Buffer
from reil.datatypes.dataclasses import History, Observation
from reil.datatypes.feature import FeatureGeneratorType, FeatureSet
from reil.utils.action_dist_modifier import ActionModifier
from reil.utils.metrics import ActionMetric, INRMetric, PTTRMetric
from reil.utils.tf_utils import MeanMetric


class PPO4Warfarin(PPO):
    def __init__(
            self, learner: PPOLearner,
            buffer: Buffer[FeatureSet, tuple[tuple[int, ...], float, float]],
            reward_clip: tuple[float | None, float | None] = ...,
            gae_lambda: float = 1, momentum_coef: float = 0.,
            momentum_mode: Literal['most recent', 'carry'] | None = None,
            action_modifiers: list[ActionModifier] | None = None,
            **kwargs: Any):
        super().__init__(learner, buffer, reward_clip, gae_lambda, **kwargs)
        self._metrics['PTTR_h'] = PTTRMetric('PTTR', mode='histogram')
        self._metrics['INR_h'] = INRMetric('INR', mode='histogram')
        self._metrics['PTTR'] = PTTRMetric('PTTR', mode='scalar')
        self._metrics['INR'] = INRMetric('INR', mode='scalar')
        self._metrics['dose'] = ActionMetric('dose', 0)
        self._metrics['duration'] = ActionMetric('duration', 1)

        self._previous_action = [
            tf.zeros(x) for x in self._learner._model._action_per_head]
        self._momentum_coef = momentum_coef
        self._carry = momentum_mode == 'carry'
        self._action_modifiers = action_modifiers or []
        for modifier in self._action_modifiers:
            self._metrics[f'modifier_{modifier.name}'] = MeanMetric(
                f'modifier_{modifier.name}_scale', dtype=tf.float32)

        if self._summary_writer:
            self._summary_writer.set_data_types({
                'PTTR_h': 'histogram', 'INR_h': 'histogram',
                'dose': 'histogram', 'duration': 'histogram'
            })

    def _update_metrics(self, **kwargs: Any) -> None:
        super()._update_metrics(**kwargs)

        state_list = kwargs.get('state_list')
        if state_list:
            self._metrics['PTTR_h'].update_state(state_list)
            self._metrics['INR_h'].update_state(state_list)
            self._metrics['PTTR'].update_state(state_list)
            self._metrics['INR'].update_state(state_list)

        action_indices = kwargs.get('action_indices')
        if action_indices:
            self._metrics['dose'].update_state(action_indices)
            try:
                self._metrics['duration'].update_state(action_indices)
            except IndexError:
                pass

    def act(
            self, state: FeatureSet, subject_id: int,
            actions: FeatureGeneratorType, iteration: int = 0) -> FeatureSet:
        if subject_id not in self._entity_list:
            raise ValueError(f'Subject with ID={subject_id} not found.')

        training_mode = self._training_trigger != 'none'
        logits: list[Tensor] = list(  # type: ignore
            self._learner.predict((state,), training=training_mode)[0])
        if training_mode:
            temp = logits
            for i, x in enumerate(self._previous_action):
                logits[i] += tf.multiply(self._momentum_coef, x)  # type: ignore
            self._previous_action = logits if self._carry else temp

            for i, modifier in enumerate(self._action_modifiers):
                if modifier is not None:
                    logits[i] = modifier(logits[i])
                    self._metrics[f'modifier_{modifier.name}'].update_state(
                        modifier._scale_fn.last_call)

        mask = list(actions.send('return mask_vector'))
        mask_index = [
            [i for i, j in enumerate(m) if j]
            for m in mask
        ]
        masked_logits = [
            tf.gather(lo, m, axis=1)
            for lo, m in zip(logits, mask_index)
        ]

        if training_mode:
            permissible_action_index = [
                int(tf.random.categorical(logits=lo, num_samples=1))  # type: ignore
                for lo in masked_logits]
        else:
            permissible_action_index = [
                int(np.argmax(lo))  # type: ignore
                for lo in masked_logits]

        if len(permissible_action_index) == 1:
            # In the implementation of feature.byindex(), if index is one dimensional
            # it excludes masked values. Hence, the following:
            action_index = permissible_action_index[0]
        else:
            action_index = [
                mask_index[i][permissible_action_index[i]]
                for i, masked_logit in enumerate(masked_logits)
            ]

        action: FeatureSet = actions.send(f'lookup {action_index}')

        return action

    def reset(self) -> None:
        self._previous_action = [
            tf.zeros(x) for x in self._learner._model._action_per_head]
        return super().reset()


class PPO4Warfarin2Phase(PPO4Warfarin):
    def __init__(
        self, init_agent: AgentBase, switch_day: int,
        init_state_comps: tuple[str, ...], main_state_comps: tuple[str, ...],
        learner: PPOLearner,
        buffer: Buffer[FeatureSet, tuple[tuple[int, ...], float, float]],
        reward_clip: tuple[float | None, float | None] = ...,
        gae_lambda: float = 1, momentum_coef: float = 0,
        momentum_mode: Literal['most recent', 'carry'] | None = None,
        **kwargs: Any
    ):
        super().__init__(
            learner, buffer, reward_clip,
            gae_lambda, momentum_coef, momentum_mode, **kwargs)
        self._init_agent = init_agent
        self._switch_day = switch_day
        self._init_state_comps = init_state_comps
        self._main_state_comps = main_state_comps

    def act(
            self, state: FeatureSet, subject_id: int,
            actions: FeatureGeneratorType, iteration: int = 0) -> FeatureSet:
        val = state.value
        if val['day'] < self._switch_day:  # type: ignore
            for f in set(val.keys()).difference(self._init_state_comps):
                state.pop(f)

            action = self._init_agent.act(state, subject_id, actions, iteration)
            return action

        for f in set(val.keys()).difference(self._main_state_comps):
            state.pop(f)

        return super().act(state, subject_id, actions, iteration)

    def _prepare_training(self, history: History) -> TrainingData[FeatureSet, int]:
        # days = [
        #     90 - t for t in self.discounted_cum_sum([
        #         h.state['duration_history'].value[-1]  # type: ignore
        #         for h in history if h.state is not None], 1)
        # ][1:]
        # temp = [h for h, d in zip(history, days) if d > self._switch_day]
        temp = [h for h in history if h.state is not None and 'day' not in h.state.value]

        return super()._prepare_training(temp)


class ActionSplitter:
    def __init__(self, action_gen: FeatureGeneratorType, index: int = 0) -> None:
        self._action_gen = action_gen
        self._index = index

    def send(self, query: str | None):
        if query is None:
            self._action_gen.send(None)  # type: ignore
        elif query.startswith('return'):
            if 'split' not in query:
                query += ' split'
            result = list(self._action_gen.send(query))
            return [result[self._index]]
        elif query.startswith('lookup'):
            index = int(query.split()[1])
            features = list(self._action_gen.send('return feature exclusive split'))
            i, f = -1, None
            for i, f in zip(range(index + 1), features[self._index]):
                pass
            if i == index:
                return f
            raise IndexError('index not found.')
        else:
            raise RuntimeError('query not supported by ActionSplitter.')


class PPO4Warfarin2Part(AgentBase):
    def __init__(
            self, dose_agent: PPO4Warfarin | PPO4Warfarin2Phase,
            duration_agent: PPO4Warfarin | PPO4Warfarin2Phase,
            training_switch: tuple[tuple[str, int], ...] | None = None,
            dose_first: bool = True):
        super().__init__()
        self._dose_agent = dose_agent
        self._duration_agent = duration_agent

        if self._dose_agent._training_trigger != self._duration_agent._training_trigger:
            raise ValueError('training_trigger must be the same for both agents.')
        self._training_trigger = self._dose_agent._training_trigger

        self._dose_first = dose_first
        self._training_switch = training_switch or (('all', 10000),)
        self._training_counter: int = 0
        self._current_training_index = 0

    def register(self, entity_name: str, _id: int | None = None) -> int:
        self._dose_agent.register(entity_name=entity_name, _id=_id)
        return self._duration_agent.register(entity_name=entity_name, _id=_id)

    def act(
            self, state: FeatureSet, subject_id: int,
            actions: FeatureGeneratorType, iteration: int = 0) -> FeatureSet:
        state_copy = copy.deepcopy(state)
        if self._dose_first:
            dose = self._dose_agent.act(
                state, subject_id,
                ActionSplitter(actions, 0), iteration
            )
            if 'duration' in dose.value:
                del dose['duration']

            duration = self._duration_agent.act(
                state_copy + dose, subject_id,
                ActionSplitter(actions, 1), iteration
            )
            if 'dose' in duration.value:
                del duration['dose']
        else:
            duration = self._duration_agent.act(
                state, subject_id,
                ActionSplitter(actions, 1), iteration
            )
            if 'dose' in duration.value:
                del duration['dose']

            dose = self._dose_agent.act(
                state_copy + duration, subject_id,
                ActionSplitter(actions, 0), iteration
            )
            if 'duration' in dose.value:
                del dose['duration']

        return dose + duration

    @staticmethod
    def _split_history(
            history: History, dose_first: bool = True) -> tuple[History, History]:
        dose_history, duration_history = History(), History()
        for h in history:
            o_dose = Observation(
                None, h.possible_actions, None, None, h.lookahead, h.reward)
            o_duration = Observation(
                None, h.possible_actions, None, None, h.lookahead, h.reward)
            action = h.action
            if action is not None:
                dose_part_name = [a for a in action.value if 'dose' in a][0]
                o_dose.action = FeatureSet(action[dose_part_name])
                o_duration.action = FeatureSet(
                    a for a in action
                    if dose_part_name not in a.name
                )
                if h.action_taken is not None:
                    o_dose.action_taken = FeatureSet(h.action_taken[dose_part_name])
                    o_duration.action_taken = FeatureSet(
                        a for a in h.action_taken
                        if dose_part_name not in a.name
                    )
                if h.state is not None:
                    if dose_first:
                        o_dose.state = h.state
                        o_duration.state = h.state + (o_dose.action_taken or o_dose.action)
                    else:
                        o_duration.state = h.state
                        o_dose.state = h.state + (o_duration.action_taken or o_duration.action)

            dose_history.append(o_dose)
            duration_history.append(o_duration)

        return dose_history, duration_history

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
        if history is not None:
            dose_history, duration_history = self._split_history(
                history, self._dose_first)

            key = self._training_switch[self._current_training_index][0]
            iteration = max(
                self._dose_agent._learner._iteration,
                self._duration_agent._learner._iteration
            )
            metrics = {}
            metrics_temp = {}
            if key in ('dose', 'all'):
                metrics = self._dose_agent.learn(dose_history)
                # metrics = {
                #     f'dose_{name}': value
                #     for name, value in metrics.items()
                # }

                if self._dose_agent._summary_writer:
                    self._dose_agent._summary_writer.write(metrics, iteration)

            if key in ('duration', 'all'):
                metrics_temp = self._duration_agent.learn(duration_history)
                if not metrics:
                    metrics = {
                        name: m
                        for name, m in metrics_temp.items()
                        if name in ('PTTR_h', 'INR_h', 'PTTR', 'INR', 'duration')
                    }
                    if self._dose_agent._summary_writer:
                        self._dose_agent._summary_writer.write(metrics, iteration)
                metrics_temp = {
                    name: m
                    for name, m in metrics_temp.items()
                    if name not in ('PTTR_h', 'INR_h', 'PTTR', 'INR')
                }

                # metrics_temp = {
                #     f'duration_{name}': value
                #     for name, value in metrics.items()
                # }

                if self._duration_agent._summary_writer:
                    self._duration_agent._summary_writer.write(metrics_temp, iteration)

            metrics.update(metrics_temp)

            if metrics:  # training has really happened!
                self._training_counter += 1
                if self._training_counter >= self._training_switch[self._current_training_index][1]:
                    self._training_counter = 0
                    self._current_training_index += 1
                    if self._current_training_index >= len(self._training_switch):
                        self._current_training_index = 0

        return metrics

    def observe(  # noqa: C901
            self, subject_id: int, stat_name: str | None,
    ) -> Generator[FeatureSet | None, dict[str, Any], None]:
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
        if (subject_id not in self._dose_agent._entity_list) or (subject_id not in self._duration_agent._entity_list):
            raise ValueError(f'Subject with ID={subject_id} not found.')

        # trigger = self._training_trigger
        # learn_on_state = trigger == 'state'
        # learn_on_action = trigger == 'action'
        # learn_on_reward = trigger == 'reward'
        # learn_on_termination = trigger == 'termination'

        history: History = []
        new_observation = None
        while True:
            try:
                new_observation = Observation()
                temp: dict[str, Any] = yield
                state: FeatureSet = temp['state']
                possible_actions: FeatureGeneratorType = temp['possible_actions']
                iteration: int = temp['iteration']

                new_observation.state = state
                new_observation.possible_actions = possible_actions
                # if learn_on_state:
                #     self._computed_metrics.update(
                #         self.learn([history[-1], new_observation]))

                if possible_actions is not None:
                    new_observation.action = self.act(
                        state=state, subject_id=subject_id,
                        actions=possible_actions, iteration=iteration)

                    temp = yield new_observation.action

                    new_observation.action_taken = temp['action_taken']
                    new_observation.lookahead = temp.get('lookahead')

                    # if learn_on_action:
                    #     self._computed_metrics.update(
                    #         self.learn([history[-1], new_observation]))

                    new_observation.reward = (yield None)['reward']

                    history.append(new_observation)

                    # if learn_on_reward:
                    #     self._computed_metrics.update(self.learn(history[-2:]))
                else:  # No actions to take, so skip the reward.
                    yield

            except GeneratorExit:
                if new_observation is None:
                    new_observation = Observation()
                if new_observation.reward is None:  # terminated early!
                    history.append(new_observation)

                # if learn_on_termination:
                    # self._computed_metrics = self.learn(history)
                if self._dose_agent._training_trigger == 'termination':
                    self._computed_metrics.update(self.learn(history))

                # if self._summary_writer:
                #     self._summary_writer.write(
                #         self._computed_metrics, self._learner._iteration)

                if stat_name is not None:
                    self.statistic.append(stat_name, subject_id)

                self.reset()

                return

    def get_parameters(self) -> Any:
        return {
            'dose_agent': self._dose_agent.get_parameters(),
            'duration_agent': self._duration_agent.get_parameters(),
            'dose_first': self._dose_first
        }

    def set_parameters(self, parameters: Any):
        self._dose_agent.set_parameters(parameters['dose_agent'])
        self._duration_agent.set_parameters(parameters['duration_agent'])
        self._dose_first = parameters.get('dose_first', True)


class PPO4WarfarinSeparate(PPO4Warfarin2Part):
    @staticmethod
    def _split_history(
            history: History, dose_first: bool = True) -> tuple[History, History]:
        dose_history, duration_history = PPO4Warfarin2Part._split_history(
            history, dose_first)
        for h in duration_history:
            if h.state is not None:
                state_val = h.state.value
                tau: int = state_val['duration_history'][-1]
                inr: float = state_val['INR_history'][-2]
                d = abs(inr - 2.5)
                h.reward = -(
                    tau * d * 0.3 if d > 0.5 else
                    1. * (28 - tau) * (0.5 - d))

        return dose_history, duration_history


class PPO4WarfarinSeparateSimpleReward(PPO4Warfarin2Part):
    @staticmethod
    def _split_history(
            history: History, dose_first: bool = True) -> tuple[History, History]:
        dose_history, duration_history = PPO4Warfarin2Part._split_history(
            history, dose_first)
        for h in duration_history:
            if h.state is not None:
                state_val = h.state.value
                tau: int = state_val['duration_history'][-1]
                inr: float = state_val['INR_history'][-1]
                d = abs(inr - 2.5)
                if d > 0.5:
                    h.reward = -1. if tau > 1 else 0.
                else:
                    h.reward = (tau - 7) / 28

        return dose_history, duration_history
