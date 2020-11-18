import itertools
from typing import Any, Dict, List, Optional, Tuple

from reil import rldata


class ActionGenerator:
    def __init__(self) -> None:
        self._components: Dict[str, List[List[Dict[str, Any]]]] = {}
        self._max_index: int = 0
        self._lower: Dict[str, Any] = {}
        self._upper: Dict[str, Any] = {}
        self._categories: Dict[str, Any] = {}
        self.reset()

    def add_categorical(self,
                        component_name: str,
                        possible_values: List[List[Any]],
                        categories: List[Any]) -> None:
        if component_name in self._components:
            raise KeyError(f'Key {component_name} already exists.')

        self._components[component_name] = [[{'name': component_name,
                                              'categorical': True,
                                              'value': vi,
                                              'categories': categories}
                                             for vi in v]
                                            for v in possible_values]
        self._categories[component_name] = categories

        self._max_index = max(self._max_index,
                              len(self._components[component_name]))

    def add_numerical(self,
                      component_name: str,
                      possible_values: List[Any],
                      lower: Any,
                      upper: Any) -> None:
        if component_name in self._components:
            raise KeyError(f'Key {component_name} already exists.')

        self._components[component_name] = [[{'name': component_name,
                                              'categorical': False,
                                              'value': vi,
                                              'lower': lower,
                                              'upper': upper}
                                             for vi in v]
                                            for v in possible_values]
        self._lower[component_name] = lower
        self._upper[component_name] = upper

        self._max_index = max(self._max_index,
                              len(self._components[component_name]))

    def possible_actions(self, state: Optional[rldata.RLData] = None) -> Tuple[rldata.RLData, ...]:
        if self._index >= self._max_index:  # avoid recreating actions
            result = self._recent_possible_actions
        else:
            actions = itertools.product(*[action_list[min(self._index, len(action_list)-1)]
                                        for action_list in self._components.values()])
            self._index += 1
            result = tuple(rldata.RLData(a) for a in actions)

        return result

    @property
    def components(self):
        return self._components.keys()

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def categories(self):
        return self._categories

    def reset(self) -> None:
        self._index: int = 0
        self._recent_possible_actions: Tuple[rldata.RLData, ...] = ()
