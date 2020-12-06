# -*- coding: utf-8 -*-
'''
ActionGenerator class
=====================

Gets lists of categorical or numerical lists as components, and generates lists
of `ReilData` objects using the product of these components.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''
import itertools
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar

from reil.datatypes import ReilData
from reil.reilbase import ReilBase

Categorical = TypeVar('Categorical')
Numerical = TypeVar('Numerical', int, float)


class ActionGenerator(ReilBase, Generic[Categorical, Numerical]):  # pylint: disable=unsubscriptable-object
    '''
    Gets lists of categorical or numerical lists as components, and generates
    lists of `ReilData` objects using the product of these components.
    '''
    def __init__(self) -> None:
        '''
        Initializes the `ActionGenerator` instance.
        '''
        self._components: Dict[str, Tuple[Tuple[Dict[str, Any], ...], ...]] = {}
        self._max_index: int = 0
        self._lower: Dict[str, Numerical] = {}
        self._upper: Dict[str, Numerical] = {}
        self._categories: Dict[str, Tuple[Categorical, ...]] = {}
        self.reset()

    def add_categorical(self,
                        component_name: str,
                        possible_values: Tuple[Tuple[Categorical, ...], ...],
                        categories: Tuple[Categorical, ...]) -> None:
        '''
        Adds a categorical component.

        ### Arguments
        component_name: name of the component.

        possible_values: a list of lists of categorical values.

        categories: a list of all possible categories.

        Raises `KeyError` if `component_name` is duplicate.

        ### Example:
        >>> AG = ActionGenerator()
        >>> AG.add_categorical(
        ...     component_name='compass_directions',
        ...     possible_values=(('N',), ('E', 'W'), ('N', 'S')),
        ...     categories=('N', 'S', 'E', 'W'))
        '''
        if component_name in self._components:
            raise KeyError(f'Key {component_name} already exists.')

        self._components[component_name] = tuple(
            tuple(
                {'name': component_name,
                 'categorical': True,
                 'value': vi,
                 'categories': categories}
                for vi in v)
            for v in possible_values)
        self._categories[component_name] = categories

        self._max_index = max(self._max_index,
                              len(self._components[component_name]))

    def add_numerical(self,
                      component_name: str,
                      possible_values: Tuple[Tuple[Numerical, ...]],
                      lower: Numerical,
                      upper: Numerical) -> None:
        '''
        Adds a numerical component.

        ### Arguments
        component_name: name of the component.

        possible_values: a list of lists of numerical values.

        lower: minimum value possible.

        upper: maximum value possible.

        Raises `KeyError` if `component_name` is duplicate.

        ### Example:
        >>> AG = ActionGenerator()
        >>> AG.add_numerical(component_name='odds',
        ...     possible_values=((1,), (1, 3)),
        ...     lower=1, upper=9)
        '''
        if component_name in self._components:
            raise KeyError(f'Key {component_name} already exists.')

        self._components[component_name] = tuple(
            tuple(
                {'name': component_name,
                 'categorical': False,
                 'value': vi,
                 'lower': lower,
                 'upper': upper}
                for vi in v)
            for v in possible_values)
        self._lower[component_name] = lower
        self._upper[component_name] = upper

        self._max_index = max(self._max_index,
                              len(self._components[component_name]))

    def possible_actions(self,
        state: Optional[ReilData] = None) -> Tuple[ReilData, ...]:
        '''
        Generates and returns a list of possible actions.

        In this implementation, an `index` keeps track of where on the list it
        is on each component, and each call of this method generates the product
        of component values and returns the result as a list of `ReilData`. The
        `index` is incremented by 1 unit. The last lits of a component is used
        if it is exhausted.

        ### Arguments
        state: an optional argument that provides the generator with the current
        state of a subject. Subclasses of `ActionGenerator` can use this argument
        to generate tailored actions.

        ### Example
        >>> AG = ActionGenerator()
        >>> AG.add_categorical(
        ...     component_name='compass_directions',
        ...     possible_values=(('N',), ('E', 'W'), ('N', 'S')),
        ...     categories=('N', 'S', 'E', 'W'))
        >>> AG.add_numerical(component_name='odds',
        ...     possible_values=((1,), (1, 3)),
        ...     lower=1, upper=9)
        >>> for i in range(5):
        ...     print(f'calling possible_actions {i}th time:')
        ...     for action in AG.possible_actions():
        ...         print(action.value)
        calling possible_actions 0th time:
        {'compass_directions': 'N', 'odds': 1}
        calling possible_actions 1th time:
        {'compass_directions': 'E', 'odds': 1}
        {'compass_directions': 'E', 'odds': 3}
        {'compass_directions': 'W', 'odds': 1}
        {'compass_directions': 'W', 'odds': 3}
        calling possible_actions 2th time:
        {'compass_directions': 'N', 'odds': 1}
        {'compass_directions': 'N', 'odds': 3}
        {'compass_directions': 'S', 'odds': 1}
        {'compass_directions': 'S', 'odds': 3}
        calling possible_actions 3th time:
        {'compass_directions': 'N', 'odds': 1}
        {'compass_directions': 'N', 'odds': 3}
        {'compass_directions': 'S', 'odds': 1}
        {'compass_directions': 'S', 'odds': 3}
        '''
        if self._index >= self._max_index:  # avoid recreating actions
            result = self._recent_possible_actions
        else:
            actions = itertools.product(
                *[action_list[min(self._index, len(action_list)-1)]
                  for action_list in self._components.values()])
            self._index += 1
            result = self._recent_possible_actions = tuple(ReilData(a)
                                                           for a in actions)

        return result

    @property
    def components(self):
        return self._components.keys()

    @property
    def lower(self) -> Dict[str, Numerical]:
        return self._lower

    @property
    def upper(self) -> Dict[str, Numerical]:
        return self._upper

    @property
    def categories(self) -> Dict[str, Tuple[Categorical, ...]]:
        return self._categories

    def reset(self) -> None:
        ''' Resets the generator.'''
        self._index: int = 0
        self._recent_possible_actions: Tuple[ReilData, ...] = ()
