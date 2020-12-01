import dataclasses

from typing_extensions import Literal


@dataclasses.dataclass
class InteractionProtocol:
    agent_name: str
    subject_name: str
    state_name: str
    reward_function_name: str
    agent_statistic_name: str
    subject_statistic_name: str
    n: int
    unit: Literal['interaction', 'instance', 'epoch']
