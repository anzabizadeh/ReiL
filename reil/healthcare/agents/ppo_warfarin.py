from typing import Any, Optional, Tuple

from reil.agents.proximal_policy_optimization import PPO, PPOLearner
from reil.datatypes.buffers.buffer import Buffer
from reil.datatypes.feature import FeatureSet
from reil.utils.metrics import INRMetric, PTTRMetric


class PPO4Warfarin(PPO):
    def __init__(
            self, learner: PPOLearner,
            buffer: Buffer[FeatureSet, Tuple[Tuple[int, ...], float, float]],
            reward_clip: Tuple[Optional[float], Optional[float]] = ...,
            gae_lambda: float = 1, **kwargs: Any):
        super().__init__(learner, buffer, reward_clip, gae_lambda, **kwargs)
        self._metrics['PTTR'] = PTTRMetric('PTTR')
        self._metrics['INR'] = INRMetric('INR')

    def _update_metrics(self, **kwargs: Any) -> None:
        super()._update_metrics(**kwargs)

        state_list = kwargs.get('state_list')
        if state_list:
            self._metrics['PTTR'].update_state(state_list)
            self._metrics['INR'].update_state(state_list)
