import argparse
import collections
import dataclasses
import pathlib
from distutils.util import strtobool
from typing import Any, Callable, Generic, Literal, TypeVar

from ruamel.yaml import YAML

from reil.utils.yaml_tools import parse_yaml

T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')


@dataclasses.dataclass()
class CommandlineArgument(Generic[T]):
    name: str
    type: Callable[[str], T]
    default: T | None
    const: T | None = None
    nargs: int | Literal['+', '*', '?'] | None = None


def str_to_tuple(
        s: str, _type: T = float,
        error: Literal['raise', 'suppress'] = 'raise') -> tuple[T, ...]:
    if error == 'suppress':
        def t(x: str) -> T | str | None:
            try:
                return _type(x)  # type: ignore
            except ValueError:
                if 'none' in x.strip().lower():
                    return None
                return x.strip()
    else:
        t = _type  # type: ignore
    return tuple(map(t, s.strip('[]() ').split(',')))  # type: ignore


def _parse_str_for_dict(s: str):
    block_chars = {
        '(': ')', '{': '}', '[': ']'
    }
    quote_chars: tuple[str, ...] = ('"', "'")

    start: int = 0
    index: int = 0
    len_s = len(s)
    wait_for: str = ','
    result: list[str] = []
    for index in range(len_s):
        match s[index]:
            case c if c in quote_chars:
                if c in wait_for:
                    wait_for = wait_for[:wait_for.index(c)]
                else:
                    wait_for += c
            case c if c in block_chars:
                # if wait_for == ',':
                #     start = index
                wait_for += block_chars[c]
            case c if c in block_chars.values():
                if c != wait_for[-1] and any(q in wait_for for q in quote_chars):
                    raise ValueError(
                        f'Expected {wait_for[-1]} in position {index}, received {c}.')
                wait_for = wait_for[:-1]
            case c if c == wait_for:
                result.append(s[start: index].strip())
                start = index + 1

    if wait_for != ',':
        raise RuntimeError(f'Closing characters not found: {wait_for[1:].split()}')

    result.append(s[start: index + 1].strip())

    return result


def _split_list(input_list: list[Any]) -> tuple[list[Any], list[Any]]:
    return input_list[0::2], input_list[1::2]


def str_to_dict(
        s: str, value_type: VT, key_type: KT = str,
        error: Literal['raise', 'suppress'] = 'raise') -> dict[KT, VT]:
    if error == 'suppress':
        def kt(x: str) -> KT | str:
            try:
                return key_type(x)  # type: ignore
            except ValueError:
                return x.strip()

        def vt(x: str) -> KT | str | None:
            try:
                return value_type(x)  # type: ignore
            except ValueError:
                if 'none' in x.strip().lower():
                    return None
                return x.strip()
    else:
        kt, vt = key_type, value_type  # type: ignore
    keys, values = _split_list(_parse_str_for_dict(s))

    return dict(zip(map(kt, keys), map(vt, values)))


def boolean(x: str) -> bool:
    return bool(strtobool(x))


def optional_arg(x: Callable[[str], T]) -> Callable[[str], T | None]:
    def optional_fx(s: str) -> T | None:
        if s.lower() == 'none':
            return None
        return x(s)

    return optional_fx


class CommandlineParser:
    def __init__(
            self,
            cmd_args: list[CommandlineArgument],
            extra_args: dict[str, Any] | None = None) -> None:

        self.parsed_args = vars(self._parse_cmd_args(cmd_args, extra_args))

    @staticmethod
    def _parse_cmd_args(
            cmd_args: list[CommandlineArgument],
            extra_args: dict[str, str] | None = None) -> argparse.Namespace:
        '''
        Parse command line arguments, and add `extra_args`.
        '''
        arg_parser = argparse.ArgumentParser()

        for arg in cmd_args:
            temp = {'type': arg.type, 'default': arg.default}
            if arg.const is not None:
                temp['const'] = arg.const
            if arg.nargs is not None:
                temp['nargs'] = arg.nargs
            elif isinstance(arg.default, (list, tuple)):
                temp['nargs'] = '+'
            arg_parser.add_argument(f'--{arg.name}', **temp)

        parsed_args = arg_parser.parse_args()

        if extra_args is not None:
            parsed_args = arg_parser.parse_args(
                list(b
                     for a in extra_args.items()
                     for b in a),
                namespace=parsed_args)

        return parsed_args


class ConfigParser:
    def __init__(
            self,
            config_filenames: dict[str, str],
            config_path: pathlib.Path | str | None = None,
            vars_dict: dict[str, str] | None = None) -> None:

        self.config: dict[str, Any]
        if config_filenames:
            self.config = {
                key: self._load_config_file(value, config_path, vars_dict)
                for key, value in config_filenames.items()
            }
        else:
            self.config = collections.defaultdict()

    @staticmethod
    def _load_config_file(
            filename: str,
            path: pathlib.Path | str | None = None,
            vars_dict: dict[str, str] | None = None) -> Any:

        _path = pathlib.Path(path or '.')
        _filename = filename if filename.endswith((
            '.yaml', '.yml')) else f'{filename}.yaml'

        with open(_path / _filename, 'r') as f:
            temp = f.read()
            # temp = YAML().load(f)

        if vars_dict:
            for name, value in vars_dict.items():
                temp = temp.replace(f'${name}$', str(value))

        return YAML().load(temp)  # type: ignore

    def extract(
            self, root_name: str, branch_name: str, as_object: bool = False
    ) -> Any:
        conf = self.config[root_name][branch_name]
        if as_object:
            return parse_yaml(conf)

        return conf

    def contains(self, root_name: str, branch_name: str) -> bool:
        return branch_name in self.config[root_name]
