from typing import Any, Literal, Optional, Tuple

import numpy as np
import tensorflow as tf
from reil.agents.proximal_policy_optimization import PPO, PPOLearner
from reil.datatypes.buffers.buffer import Buffer
from reil.datatypes.feature import FeatureGeneratorType, FeatureSet
from reil.utils.metrics import INRMetric, PTTRMetric


class PPO4Warfarin(PPO):
    def __init__(
            self, learner: PPOLearner,
            buffer: Buffer[FeatureSet, Tuple[Tuple[int, ...], float, float]],
            reward_clip: Tuple[Optional[float], Optional[float]] = ...,
            gae_lambda: float = 1, prev_action_weight: float = 0.,
            prev_action_mode: Literal['most recent', 'carry'] = 'most recent',
            **kwargs: Any):
        super().__init__(learner, buffer, reward_clip, gae_lambda, **kwargs)
        self._metrics['PTTR'] = PTTRMetric('PTTR')
        self._metrics['INR'] = INRMetric('INR')
        self._previous_action = [
            tf.zeros(x) for x in self._learner._model._output_lengths]
        self._prev_action_weight = prev_action_weight
        self._carry = prev_action_mode == 'carry'

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
        logits = self._learner.predict((state,), training=training_mode)[0]
        if training_mode:
            temp = logits
            for i, x in enumerate(self._previous_action):
                logits[i] += self._prev_action_weight * x
            self._previous_action = logits if self._carry else temp
            action_index = [
                int(tf.random.categorical(logits=lo, num_samples=1))
                for lo in logits]
        else:
            action_index = [int(np.argmax(lo)) for lo in logits]

        if len(action_index) == 1:
            action_index = action_index[0]
        action: FeatureSet = actions.send(f'lookup {action_index}')

        return action

    def reset(self) -> None:
        self._previous_action = [
            tf.zeros(x) for x in self._learner._model._output_lengths]
        return super().reset()
