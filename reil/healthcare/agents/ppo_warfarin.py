from typing import Any, Literal

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from reil.agents.agent import AgentBase, TrainingData
from reil.agents.proximal_policy_optimization import PPO, PPOLearner
from reil.datatypes.buffers.buffer import Buffer
from reil.datatypes.dataclasses import History
from reil.datatypes.feature import FeatureGeneratorType, FeatureSet
from reil.utils.metrics import INRMetric, PTTRMetric
from reil.utils.action_dist_modifier import ActionModifier


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
        self._metrics['PTTR'] = PTTRMetric('PTTR')
        self._metrics['INR'] = INRMetric('INR')
        self._previous_action = [
            tf.zeros(x) for x in self._learner._model._action_per_head]
        self._momentum_coef = momentum_coef
        self._carry = momentum_mode == 'carry'
        self._action_modifiers = action_modifiers or []

    def _update_metrics(self, **kwargs: Any) -> None:
        super()._update_metrics(**kwargs)

        state_list = kwargs.get('state_list')
        if state_list:
            self._metrics['PTTR'].update_state(state_list)
            self._metrics['INR'].update_state(state_list)

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
                logits[i] += tf.multiply(self._momentum_coef, x)
            self._previous_action = logits if self._carry else temp

            for i, modifier in enumerate(self._action_modifiers):
                if modifier is not None:
                    logits[i] = modifier(logits[i])

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
                int(tf.random.categorical(logits=lo, num_samples=1))
                for lo in masked_logits]
        else:
            permissible_action_index = [
                int(np.argmax(lo)) for lo in masked_logits]

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
        temp = [h for h in history if (h.action or {}).get('dose') is None]
        return super()._prepare_training(temp)
