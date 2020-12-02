import dataclasses

from typing_extensions import Literal


@dataclasses.dataclass
class InteractionProtocol:
    '''
    The datatype to specify how an `agent` should interact with a `subject` in
    an `environment`.
    '''
    agent_name: str
    subject_name: str
    state_name: str
    reward_function_name: str
    agent_statistic_name: str
    subject_statistic_name: str
    n: int
    unit: Literal['interaction', 'instance', 'epoch']
