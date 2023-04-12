from typing import Any
from reil.agents.agent_base import AgentBase
from reil.datatypes.feature import FeatureGeneratorType, FeatureSet


class TwoPhaseAgent(AgentBase):
    def __init__(
        self, first_agent: AgentBase, second_agent: AgentBase, switch_day: int,
        init_state_comps: tuple[str, ...], main_state_comps: tuple[str, ...],
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._first_agent = first_agent
        self._second_agent = second_agent
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

            action = self._first_agent.act(
                state, subject_id, actions, iteration)
            return action

        for f in set(val.keys()).difference(self._main_state_comps):
            state.pop(f)

        return self._second_agent.act(state, subject_id, actions, iteration)
