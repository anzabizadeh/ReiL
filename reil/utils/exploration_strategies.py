import random
from typing import Callable


class ExplorationStrategy:
    def __init__(self) -> None:
        pass

    def explore(self, episode: int = 0) -> bool:
        return True


class ConstantEpsilonGreedy(ExplorationStrategy):
    def __init__(self, epsilon: float) -> None:
        self._epsilon = epsilon

    def explore(self, episode: int = 0) -> bool:
        return random.random() < self._epsilon


class VariableEpsilonGreedy(ExplorationStrategy):
    def __init__(self, epsilon: Callable[[int], float]) -> None:
        self._epsilon = epsilon

    def explore(self, episode: int) -> bool:
        return random.random() < self._epsilon(episode)


# def epsilon(n): return 1/(1+n/200)
