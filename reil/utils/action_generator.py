# -*- coding: utf-8 -*-
'''
ActionGenerator class
=====================

Gets lists of categorical or numerical lists as components, and generates lists
of `ReilData` objects using the product of these components.


'''
import dataclasses
import itertools
from typing import Any, Dict, Generic, Iterator, Optional, Tuple, TypeVar, Union

from reil.datatypes import ReilData
from reil.reilbase import ReilBase

Categorical = TypeVar('Categorical')
Numerical = TypeVar('Numerical', int, float)


@dataclasses.dataclass(frozen=True)
class CategoricalComponent(Generic[Categorical]):  # pylint: disable=unsubscriptable-object
    name: str
    possible_values: Tuple[Tuple[Categorical, ...], ...]
    categories: Tuple[Categorical, ...]
    length: int = dataclasses.field(init=False, hash=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, 'length', len(self.possible_values))

    def generate(self, index: int) -> Iterator[Dict[str, Any]]:
        _index = min(index, self.length - 1)
        return (
            {'name': self.name,
             'categorical': True,
             'value': vi,
             'categories': self.categories}
            for vi in self.possible_values[_index])


@dataclasses.dataclass(frozen=True)
class NumericalComponent(Generic[Numerical]):  # pylint: disable=unsubscriptable-object
    name: str
    possible_values: Tuple[Tuple[Numerical, ...], ...]
    lower: Numerical
    upper: Numerical
    length: int = dataclasses.field(init=False, hash=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, 'length', len(self.possible_values))

    def generate(self, index: int) -> Iterator[Dict[str, Any]]:
        _index = min(index, self.length - 1)
        return (
            {'name': self.name,
             'categorical': False,
             'value': vi,
             'lower': self.lower,
             'upper': self.upper}
            for vi in self.possible_values[_index])


class ActionGenerator(ReilBase, Generic[Categorical, Numerical]):  # pylint: disable=unsubscriptable-object
    '''
    Gets lists of categorical or numerical lists as components, and generates
    lists of `ReilData` objects using the product of these components.
    '''

    def __init__(self,
                 components: Optional[Dict[str, Union[CategoricalComponent, NumericalComponent]]] = None,
                 **kwargs: Any) -> None:
        '''
        Initializes the `ActionGenerator` instance.
        '''
        super().__init__(**kwargs)

        if components is not None:
            self._components = components
        else:
            self._components: Dict[str,
                                   Union[CategoricalComponent, NumericalComponent]] = {}
        self._max_index: int = 0
        self.reset()

    def add_categorical(self,
                        component_name: str,
                        possible_values: Tuple[Tuple[Categorical, ...], ...],
                        categories: Tuple[Categorical, ...]) -> None:
        '''
        Adds a categorical component.

        Arguments
-----------
        component_name: name of the component.

        possible_values: a list of lists of categorical values.

        categories: a list of all possible categories.

        Raises `KeyError` if `component_name` is duplicate.

        Example:
-----------
        >>> AG = ActionGenerator()
        >>> AG.add_categorical(
        ...     component_name='compass_directions',
        ...     possible_values=(('N',), ('E', 'W'), ('N', 'S')),
        ...     categories=('N', 'S', 'E', 'W'))
        '''
        if component_name in self._components:
            raise KeyError(f'Key {component_name} already exists.')

        self._components[component_name] = CategoricalComponent(component_name,
                                                                possible_values, categories)

        self._max_index = max(self._max_index,
                              len(self._components[component_name].possible_values))

    def add_numerical(self,
                      component_name: str,
                      possible_values: Tuple[Tuple[Numerical, ...]],
                      lower: Numerical,
                      upper: Numerical) -> None:
        '''
        Adds a numerical component.

        Arguments
-----------
        component_name: name of the component.

        possible_values: a list of lists of numerical values.

        lower: minimum value possible.

        upper: maximum value possible.

        Raises `KeyError` if `component_name` is duplicate.

        Example:
-----------
        >>> AG = ActionGenerator()
        >>> AG.add_numerical(component_name='odds',
        ...     possible_values=((1,), (1, 3)),
        ...     lower=1, upper=9)
        '''
        if component_name in self._components:
            raise KeyError(f'Key {component_name} already exists.')

        self._components[component_name] = NumericalComponent(
            component_name, possible_values, lower, upper)

        self._max_index = max(self._max_index,
                              len(self._components[component_name].possible_values))

    def possible_actions(self,
                         state: Optional[ReilData] = None) -> Tuple[ReilData, ...]:
        '''
        Generates and returns a list of possible actions.

        In this implementation, an `index` keeps track of where on the list it
        is on each component, and each call of this method generates the product
        of component values and returns the result as a list of `ReilData`. The
        `index` is incremented by 1 unit. The last lits of a component is used
        if it is exhausted.

        Arguments
-----------
        state: an optional argument that provides the generator with the current
        state of a subject. Subclasses of `ActionGenerator` can use this argument
        to generate tailored actions.

        Example
-----------
        >>> AG = ActionGenerator()
        >>> AG.add_categorical(
        ...     component_name='compass_directions',
        ...     possible_values=(('N',), ('E', 'W'), ('N', 'S')),
        ...     categories=('N', 'S', 'E', 'W'))
        >>> AG.add_numerical(component_name='odds',
        ...     possible_values=((1,), (1, 3)),
        ...     lower=1, upper=9)
        >>> for i in ('1st', '2nd', '3rd', '4th'):
        ...     print(f'calling possible_actions for the {i} time:')
        ...     for action in AG.possible_actions():
        ...         print(action.value)
        calling possible_actions for the 1st time:
        {'compass_directions': 'N', 'odds': 1}
        calling possible_actions for the 2nd time:
        {'compass_directions': 'E', 'odds': 1}
        {'compass_directions': 'E', 'odds': 3}
        {'compass_directions': 'W', 'odds': 1}
        {'compass_directions': 'W', 'odds': 3}
        calling possible_actions for the 3rd time:
        {'compass_directions': 'N', 'odds': 1}
        {'compass_directions': 'N', 'odds': 3}
        {'compass_directions': 'S', 'odds': 1}
        {'compass_directions': 'S', 'odds': 3}
        calling possible_actions for the 4th time:
        {'compass_directions': 'N', 'odds': 1}
        {'compass_directions': 'N', 'odds': 3}
        {'compass_directions': 'S', 'odds': 1}
        {'compass_directions': 'S', 'odds': 3}
        '''
        if self._index >= self._max_index:  # avoid recreating actions
            result = self._recent_possible_actions
        else:
            actions = itertools.product(
                *[component.generate(self._index)
                  for component in self._components.values()])
            self._index += 1
            result = self._recent_possible_actions = tuple(ReilData(a)
                                                           for a in actions)

        return result

    @property
    def components(self):
        return self._components.keys()

    @property
    def lower(self) -> Dict[str, Numerical]:
        return dict((component.name, component.lower)  # type: ignore
                    for component in self._components.values()
                    if hasattr(component, 'lower'))

    @property
    def upper(self) -> Dict[str, Numerical]:
        return dict((component.name, component.upper)  # type: ignore
                    for component in self._components.values()
                    if hasattr(component, 'upper'))

    @property
    def categories(self) -> Dict[str, Tuple[Categorical, ...]]:
        return dict((component.name, component.categories)  # type: ignore
                    for component in self._components.values()
                    if hasattr(component, 'categories'))

    def reset(self) -> None:
        ''' Resets the generator.'''
        self._index: int = 0
        self._recent_possible_actions: Tuple[ReilData, ...] = ()


if __name__ == "__main__":
    AG = ActionGenerator()
    AG.add_categorical(
        component_name='compass_directions',
        possible_values=(('N',), ('E', 'W'), ('N', 'S')),
        categories=('N', 'S', 'E', 'W'))
    AG.add_numerical(component_name='odds',
                     possible_values=((1,), (1, 3)),  # type: ignore
                     lower=1, upper=9)
    for i in ('1st', '2nd', '3rd', '4th'):
        print(f'calling possible_actions for the {i} time:')
        for action in AG.possible_actions():
            print(action.value)
