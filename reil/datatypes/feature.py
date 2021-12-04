# -*- coding: utf-8 -*-
'''
Feature, FeatureGenerator, FeatureArray classes
===============================================

`FeatureArray` is The main datatype used to communicate `state`s, `action`s,
and `reward`s, between objects in `reil`. `FeatureArray` is basically a
dictionary that contains instances of `Feature`.
`FeatureGenerator` allows for generating new `Feature` instances. It can
`generate` a new value or turn an input into a `Feature`. It enforces
`categorical` and `numerical` constraints, and produces `normalized` value.
'''
from __future__ import annotations

import dataclasses
from functools import cached_property
import itertools
from typing import (Any, Callable, Dict, Iterable, Literal, Optional,
                    Tuple, Union)

MISSING = '__missing_feature__'

MissingType = Literal['__missing_feature__']


@dataclasses.dataclass(frozen=True)
class Feature:
    '''
    Attributes
    ----------
    name:
        Name of the data.

    value:
        Value of the data. Can be one item, or a tuple of items of the same
        type.

    is_numerical:
        Is the value numerical?

    normalized:
        The normal form of the value.

    categories:
        A tuple of categories that the value can take.

    lower:
        The lower limit for numerical values.

    upper:
        The upper limit for numerical values.
    '''
    name: str
    value: Optional[Union[Any, Tuple[Any, ...], MissingType]] = None
    is_numerical: Optional[bool] = dataclasses.field(
        default=None, repr=False, compare=False)
    categories: Optional[Tuple[Any, ...]] = dataclasses.field(
        default=None, repr=False, compare=False)
    lower: Optional[Any] = dataclasses.field(
        default=None, repr=False, compare=False)
    upper: Optional[Any] = dataclasses.field(
        default=None, repr=False, compare=False)
    normalized: Optional[
        Union[Tuple[float, ...], Tuple[int, ...]]] = dataclasses.field(
            default=None, repr=False, compare=False)
    dict_fields: Tuple[str, ...] = dataclasses.field(
        default=('name', 'value'), init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.is_numerical is None:
            return

        if self.is_numerical:
            if self.categories is not None:
                raise ValueError('Numerical type cannot have categories.')

            self.__dict__['dict_fields'] = ('name', 'value', 'lower', 'upper')
        else:
            if self.lower is not None or self.upper is not None:
                raise ValueError(
                    'Categorical type cannot have lower and upper.')

            self.__dict__['dict_fields'] = ('name', 'value', 'categories')

    @classmethod
    def numerical(
            cls, name: str,
            value: Optional[Union[Any, Tuple[Any, ...]]] = None,
            lower: Optional[Any] = None, upper: Optional[Any] = None,
            normalized: Optional[Tuple[float, ...]] = None):
        '''Create a numerical instance of `Feature`.'''
        return cls(
            name=name, value=value, is_numerical=True,
            lower=lower, upper=upper, normalized=normalized)

    @classmethod
    def categorical(
            cls, name: str,
            value: Optional[Union[Any, Tuple[Any, ...], MissingType]] = None,
            categories: Optional[Tuple[Any, ...]] = None,
            normalized: Optional[Tuple[float, ...]] = None):
        '''Create a categorical instance of `Feature`.'''
        return cls(
            name=name, value=value, is_numerical=False,
            categories=categories, normalized=normalized)

    @cached_property
    def as_dict(self):
        '''
        Return the data as a dictionary.

        Returns
        -------
        :
            The data as a dictionary.
        '''
        return {field: self.__dict__[field] for field in self.dict_fields}

    def __add__(self, other: Any):
        my_type = type(self)
        if type(other) != my_type:
            raise TypeError(
                "unsupported operand type(s) for +: "
                f"'{my_type}' and '{type(other)}'")

        for k, v in self.__dict__.items():
            if k not in ('value', 'normalized'):
                if other.__dict__[k] != v:
                    raise TypeError(
                        f'Different {k} values: {v} != {other.__dict__[k]}.')

        new_value = self.__dict__['value'] + other.value
        if self.is_numerical:
            return my_type.numerical(
                name=self.name, value=new_value,
                lower=self.__dict__.get('lower'),
                upper=self.__dict__.get('upper'))
        else:
            return my_type.categorical(
                name=self.name, value=new_value,
                categories=self.__dict__.get('categories'))


@dataclasses.dataclass(frozen=True)
class FeatureGenerator:
    '''
    A class to generate `Feature`s.

    Attributes
    ----------
    name:
        Name of the data.

    is_numerical:
        Is the feature to be generated numerical?

    categories:
        A tuple of categories that the value can take.

    probabilities:
        A tuple of probabilities corresponding with each category. This can
        be used to generate new random `Feature` instances.

    mean:
        The mean for numerical values. This can
        be used to generate new random `Feature` instances.

    stdev:
        The standard deviation for numerical values. This can
        be used to generate new random `Feature` instances.

    lower:
        The lower limit for numerical values.

    upper:
        The upper limit for numerical values.

    normalizer:
        For a categorical `FeatureGenerator`, normalizer is a dictionary of
        categories as keys and their corresponding one-hot encodings as values.
        For a numerical `FeatureGenerator`, normalizer is a function that
        accepts the value and returns its normalized value.

    randomized:
        Determines for the `generator`, whether the new `Feature` should be
        randomly generated.

    generator:
        A function that accepts a `FeatureGenerator` instance, and produces
        a value for the new `Feature`.

    allow_missing:
        If `True`, a categorical generator can generate a `MISSING` instance.
        Also the normalized form will have one more categories to account for
        `MISSING`.
    '''
    name: str
    is_numerical: bool = dataclasses.field(
        default=False, repr=False, compare=False)
    categories: Optional[Tuple[Any, ...]] = None
    probabilities: Optional[Tuple[float, ...]] = None
    mean: Optional[Any] = None
    stdev: Optional[Any] = None
    lower: Optional[Any] = None
    upper: Optional[Any] = None
    normalizer: Optional[Any] = dataclasses.field(
        default=None, init=False, repr=False, compare=False)
    randomized: Optional[bool] = True
    generator: Optional[
        Callable[[FeatureGenerator], Union[Any, Tuple[Any, ...]]]] = None
    allow_missing: bool = False
    recent_value: Tuple[Any, Feature] = dataclasses.field(
        default=(None, Feature('')), init=False, repr=False, compare=False)

    @classmethod
    def numerical(
            cls, name: str,
            mean: Optional[float] = None, stdev: Optional[float] = None,
            lower: Optional[float] = None, upper: Optional[float] = None,
            generator: Optional[Callable[
                [FeatureGenerator], Union[Any, Tuple[Any, ...]]]] = None,
            randomized: Optional[bool] = None):
        return cls(
            name=name, is_numerical=True, lower=lower, upper=upper, mean=mean,
            stdev=stdev, generator=generator, randomized=randomized,
            allow_missing=False)

    @classmethod
    def categorical(
            cls, name: str,
            categories: Optional[Tuple[Any, ...]] = None,
            probabilities: Optional[Tuple[float, ...]] = None,
            generator: Optional[Callable[
                [FeatureGenerator], Union[Any, Tuple[Any, ...]]]] = None,
            randomized: Optional[bool] = None, allow_missing: bool = False):
        return cls(
            name=name, is_numerical=False,
            categories=categories, probabilities=probabilities,
            generator=generator, randomized=randomized,
            allow_missing=allow_missing)

    def __post_init__(self):
        self.__dict__['recent_value'] = (None, None)

        if self.is_numerical:
            if self.categories is not None:
                raise ValueError('Numerical type cannot have categories.')
            if self.allow_missing:
                raise TypeError(
                    'Only categorical type can accept missing values.')

            self._process_numerical()
        else:
            if self.lower is not None or self.upper is not None:
                raise ValueError(
                    'Categorical type cannot have lower and upper.')

            probabilities = self.probabilities
            categories = self.categories
            if probabilities is not None:
                if abs(sum(probabilities) - 1.0) > 1e-6:
                    raise ValueError(
                        'probabilities should add up to 1.0.'
                        f'Got {sum(probabilities)}')
                if categories is None:
                    raise ValueError(
                        'probabilities cannot be set for None categories.')
                if len(probabilities) != len(categories):
                    raise ValueError(
                        'Size mismatch. '
                        f'{len(categories)} categories vs. '
                        f'{len(probabilities)} probabilities')

            self._process_categorical()

    def __call__(
        self, value: Optional[Union[Any, Tuple[Any, ...], MissingType]] = None
    ) -> Feature:
        if value is None:
            if (gen := self.generator) is None:
                raise RuntimeError('generator not found.')
            _value = gen(self)
        else:
            _value = value

        if _value == self.recent_value[0]:
            return self.recent_value[1]

        if self.is_numerical:
            if _value == MISSING:
                raise ValueError('Numerical feature cannot accept MISSING.')

            return_value = self._call_numerical(_value)  # type: ignore
        else:
            return_value = self._call_categorical(_value)  # type: ignore

        self.__dict__['recent_value'] = (_value, return_value)

        return return_value

    def _process_categorical(self):
        cats = self.categories
        allow_missing = self.allow_missing
        if cats is None:
            return

        allow_missing_offset = int(allow_missing)
        cat_count = len(cats) - 1 + allow_missing_offset
        normalizer = {}
        for i, c in enumerate(cats[:-1]):
            temp = [0] * cat_count
            temp[i] = 1
            normalizer[c] = tuple(temp)

        temp = [0] * cat_count
        temp[-1] = allow_missing_offset
        normalizer[cats[-1]] = tuple(temp)

        if allow_missing:
            normalizer[MISSING] = tuple([0] * cat_count)

        self.__dict__['normalizer'] = normalizer

    def _process_numerical(self):
        lower: Optional[float] = self.lower  # type: ignore
        upper: Optional[float] = self.upper  # type: ignore

        if lower is None or upper is None or upper == lower:
            self.__dict__['normalizer'] = lambda _: None  # type: ignore
        else:
            if lower > upper:
                raise ValueError(
                    f'lower ({lower}) cannot be '
                    f'greater than upper ({upper}).')

            denominator = upper - lower

            def normalizer(x: float) -> float:
                return (x - lower)/denominator  # type: ignore

            self.__dict__['normalizer'] = normalizer

    def _call_categorical(
            self, value: Union[Any, Tuple[Any, ...], MissingType]
    ) -> Feature:
        normalizer = self.normalizer
        categories = self.categories or ()

        if normalizer is None:
            normalized = None
        elif value in categories:
            normalized = normalizer[value]
        elif self.allow_missing and value == MISSING:
            normalized = normalizer[value]
        else:
            try:
                normalized = tuple(
                    itertools.chain(
                        *(normalizer[d]
                          for d in value)))  # type: ignore
            except KeyError:
                raise ValueError(
                    f'{value} is not '
                    f'in the categories={categories}.')

        instance = Feature.categorical(
            name=self.name, value=value, categories=self.categories,
            normalized=normalized)

        return instance

    def _call_numerical(
            self, value: Union[Any, Tuple[Any, ...]]
    ) -> Feature:
        normalizer = self.normalizer
        lower = self.lower
        upper = self.upper

        if value is None:
            normalized = None

        elif isinstance(value, tuple):
            if (lower is not None
                    and min(value) < lower):  # type: ignore
                raise ValueError(f'Lower bound ({lower}) violated:\n {value}')

            if (upper is not None
                    and max(value) > upper):  # type: ignore
                raise ValueError(f'Upper bound ({upper}) violated:\n {value}')

            normalized = tuple(normalizer(d) for d in value)  # type: ignore

        else:
            if (lower is not None
                    and value < lower):  # type: ignore
                raise ValueError(f'Lower bound ({lower}) violated:\n {value}')

            if (upper is not None
                    and value > upper):  # type: ignore
                raise ValueError(f'Upper bound ({upper}) violated:\n {value}')

            normalized = normalizer(value)  # type: ignore

        instance: Feature = \
            Feature.numerical(
                name=self.name, value=value,   # type: ignore
                lower=lower, upper=upper,
                normalized=normalized)

        return instance


class FeatureArray:
    '''
    The main datatype used to communicate `state`s, `action`s, and `reward`s,
    between objects in `reil`.
    '''

    def __init__(self, data: Union[Feature, Iterable[Feature]]):
        '''
        Arguments
        ---------
        data:
            One or a sequence of `Feature`s.
        '''
        temp: Dict[str, Feature] = {}
        _data: Iterable[Any] = (
            data if hasattr(data, '__iter__') else [data])  # type: ignore

        for d in _data:
            if isinstance(d, Feature):
                name = d.name
                if name in temp:
                    raise KeyError(f'Duplicate name ({name}).')

                temp[name] = d
            else:
                raise TypeError(f'Unknown input type {type(d)} for item: {d}')

        self._data = temp

    @cached_property
    def value(self):
        '''
        Return a dictionary with elements' names as keys and
        their respective values as values.

        Returns
        -------
        :
            Names of the elements and their values.
        '''
        return {name: v.value for name, v in self._data.items()}

    @cached_property
    def lower(self):
        '''
        Return all `lower` attributes.

        Returns
        -------
        :
            `lower` attribute of all `NumericalData` variables with their names
            as keys.
        '''
        return {name: v.lower for name, v in self._data.items()}

    @cached_property
    def upper(self):
        '''
        Return all `upper` attributes.

        Returns
        -------
        :
            `upper` attribute of all `NumericalData` variables with their names
            as keys.
        '''
        return {name: v.upper for name, v in self._data.items()}

    @cached_property
    def categories(self):
        '''
        Return all `categories` attributes.

        Returns
        -------
        :
            `categories` attribute of all `CategoricalData` variables with
            their names as keys.
        '''
        return {name: v.categories for name, v in self._data.items()}

    @cached_property
    def normalized(self):
        '''
        Normalize all items in the instance.

        Returns
        -------
        :
            A `FeatureArray` of the normalized values of all the items in the
            instance, in the form of numerical `Feature`s.
        '''
        return FeatureArray(
            FeatureGenerator.numerical(name=name, lower=0, upper=1)
            (v.normalized)  # type: ignore
            for name, v in self._data.items())

    @cached_property
    def flattened(self):
        """Combine values of all items in the instance.

        Returns
        -------
        :
            A list that contains all the values of all the items.
        """
        def make_iterable(x: Any) -> Iterable[Any]:
            return x if hasattr(x, '__iter__') else [x]

        return list(
            itertools.chain(
                *[make_iterable(sublist) for sublist in self.value.values()]))

    def split(self):
        """Split the `FeatureArray` into a list of `FeatureArray`s.

        Returns
        -------
        :
            All items in the instance as separate `FeatureArray` instances.
        """
        if len(self) == 1:
            d = next(iter(self._data.values()))
            if not isinstance(d.value, (list, tuple)):
                splitted_list = FeatureArray(d)
            else:
                temp = d.as_dict
                cls = type(d)
                value = temp['value']
                del temp['value']
                if 'is_numerical' in temp:
                    del temp['is_numerical']

                splitted_list = [
                    FeatureArray(cls(value=v, **temp))
                    for v in value]

        else:
            splitted_list = list(FeatureArray(v) for v in self._data.values())

        return splitted_list

    def __iter__(self):
        return iter(self._data.values())

    def __getitem__(self, k: str):
        return self._data.__getitem__(k)

    def __len__(self):
        return self._data.__len__()

    def __hash__(self):
        return hash(tuple(self._data.items()))

    def __eq__(self, other: Any):
        return isinstance(other, type(self)) and (self._data == other._data)

    def __add__(self, other: Any):
        if not isinstance(other, FeatureArray):
            new_data = FeatureArray(other)
        else:
            new_data = other

        # if not isinstance(new_data, FeatureArray):
        #     raise TypeError(
        #         'Concatenation of type FeatureArray'
        #         f' and {type(other)} not implemented!')

        overlaps = set(new_data._data).intersection(self._data)
        if overlaps:
            raise ValueError(f'Objects already exist: {overlaps}.')

        return FeatureArray(itertools.chain(
            self._data.values(), new_data._data.values()))

    def __neg__(self):
        temp = [v
                for v in self._data.values()]
        for item in temp:
            if hasattr(item.value, '__neg__'):
                neg_value = -item.value  # type: ignore
                lower: Any = item.__dict__.get('lower') or neg_value
                upper: Any = item.__dict__.get('upper') or neg_value
                if lower <= neg_value <= upper:
                    object.__setattr__(item, 'value', neg_value)
                else:
                    raise ValueError(
                        f'Bounds violated: lower: {lower}, '
                        f'upper: {upper}, '
                        f'negative value: {neg_value}')

        return FeatureArray(temp)  # type: ignore

    def __repr__(self):
        return f'[{super().__repr__()} -> {self._data}]'

    def __str__(self):
        return f"[{', '.join((d.__str__() for d in self._data.items()))}]"


def change_to_missing(feature: Feature) -> Feature:
    if feature.is_numerical:
        raise TypeError('Only categorical features can have missing.')
    categories = feature.categories
    normalized = feature.normalized
    if categories is None:
        raise ValueError('No categories defined!')
    if normalized is None:
        raise ValueError(
            'Cannot generate normal form for a feature '
            'without the normal form.')

    if len(categories) != len(normalized):
        raise TypeError('Feature is not allowed to have MISSING')

    return FeatureGenerator.categorical(  # type: ignore
        name=feature.name, categories=categories, allow_missing=True)(MISSING)


def change_array_to_missing(
        features: FeatureArray, suppress_error: bool = True) -> FeatureArray:
    def try_to_change(feature: Feature) -> Feature:
        try:
            return change_to_missing(feature)
        except (TypeError, ValueError):
            if suppress_error:
                return feature
            raise

    return FeatureArray(try_to_change(f) for f in features)
