from typing import Any, Callable, List, Literal, Optional, Tuple

import numpy as np
import tensorflow as tf

from reil.agents.agent import AgentBase, TrainingData
from reil.agents.proximal_policy_optimization import PPO, PPOLearner
from reil.datatypes.buffers.buffer import Buffer
from reil.datatypes.dataclasses import History
from reil.datatypes.feature import FeatureGeneratorType, FeatureSet
from reil.utils.metrics import INRMetric, PTTRMetric
from reil.utils.ricker_wavelet import RickerWavelet2


class PPO4Warfarin(PPO):
    def __init__(
            self, learner: PPOLearner,
            buffer: Buffer[FeatureSet, Tuple[Tuple[int, ...], float, float]],
            reward_clip: Tuple[Optional[float], Optional[float]] = ...,
            gae_lambda: float = 1, momentum_coef: float = 0.,
            momentum_mode: Literal['most recent', 'carry'] = 'most recent',
            ricker_instances: Optional[List[
                Tuple[RickerWavelet2, Callable[[tf.Tensor], tf.Tensor]]]] = None,
            **kwargs: Any):
        super().__init__(learner, buffer, reward_clip, gae_lambda, **kwargs)
        self._metrics['PTTR'] = PTTRMetric('PTTR')
        self._metrics['INR'] = INRMetric('INR')
        self._previous_action = [
            tf.zeros(x) for x in self._learner._model._output_lengths]
        self._momentum_coef = momentum_coef
        self._carry = momentum_mode == 'carry'
        self._ricker_instances = ricker_instances or []
        self._ricker_counter = tf.constant(0, dtype=tf.int32)

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
        logits = list(self._learner.predict((state,), training=training_mode)[0])
        if training_mode:
            temp = logits
            for i, x in enumerate(self._previous_action):
                logits[i] += self._momentum_coef * x
            self._previous_action = logits if self._carry else temp

            self._ricker_counter += 1
            for i, ricker in enumerate(self._ricker_instances):
                if ricker is not None:
                    r = ricker[0].f(ricker[1](self._ricker_counter))
                    logits[i] += r

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

        action_index = [
            mask_index[i][permissible_action_index[i]]
            for i, masked_logit in enumerate(masked_logits)
        ]

        if len(action_index) == 1:
            action_index = action_index[0]
        action: FeatureSet = actions.send(f'lookup {action_index}')

        return action

    def reset(self) -> None:
        self._previous_action = [
            tf.zeros(x) for x in self._learner._model._output_lengths]
        return super().reset()


class PPO4Warfarin2Phase(PPO4Warfarin):
    def __init__(
        self, init_agent: AgentBase, switch_day: int,
        init_state_comps: Tuple[str, ...], main_state_comps: Tuple[str, ...],
        learner: PPOLearner,
        buffer: Buffer[FeatureSet, Tuple[Tuple[int, ...], float, float]],
        reward_clip: Tuple[Optional[float], Optional[float]] = ...,
        gae_lambda: float = 1, momentum_coef: float = 0,
        momentum_mode: Literal['most recent', 'carry'] = 'most recent',
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
