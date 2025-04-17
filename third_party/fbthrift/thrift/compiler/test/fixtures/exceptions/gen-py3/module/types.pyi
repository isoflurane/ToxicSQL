#
# Autogenerated by Thrift
#
# DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
#  @generated
#

import folly.iobuf as _fbthrift_iobuf
import thrift.py3.types
import thrift.py3.exceptions
from thrift.py3.types import __NotSet, NOTSET
import typing as _typing
from typing_extensions import Final

import sys
import itertools


__property__ = property


class Banal(thrift.py3.exceptions.GeneratedError, _typing.Hashable, _typing.Iterable[_typing.Tuple[str, _typing.Any]]):
    class __fbthrift_IsSet:
        pass

    def __init__(
        self, 
    ) -> None: ...

    def __iter__(self) -> _typing.Iterator[_typing.Tuple[str, _typing.Any]]: ...
    def __hash__(self) -> int: ...
    def __lt__(self, other: 'Banal') -> bool: ...
    def __gt__(self, other: 'Banal') -> bool: ...
    def __le__(self, other: 'Banal') -> bool: ...
    def __ge__(self, other: 'Banal') -> bool: ...


class Fiery(thrift.py3.exceptions.GeneratedError, _typing.Hashable, _typing.Iterable[_typing.Tuple[str, _typing.Any]]):
    class __fbthrift_IsSet:
        message: bool
        pass

    message: Final[str] = ...

    def __init__(
        self, *,
        message: _typing.Optional[str]=None
    ) -> None: ...

    def __iter__(self) -> _typing.Iterator[_typing.Tuple[str, _typing.Any]]: ...
    def __hash__(self) -> int: ...
    def __lt__(self, other: 'Fiery') -> bool: ...
    def __gt__(self, other: 'Fiery') -> bool: ...
    def __le__(self, other: 'Fiery') -> bool: ...
    def __ge__(self, other: 'Fiery') -> bool: ...


class Serious(thrift.py3.exceptions.GeneratedError, _typing.Hashable, _typing.Iterable[_typing.Tuple[str, _typing.Any]]):
    class __fbthrift_IsSet:
        sonnet: bool
        pass

    sonnet: Final[_typing.Optional[str]] = ...

    def __init__(
        self, *,
        sonnet: _typing.Optional[str]=None
    ) -> None: ...

    def __iter__(self) -> _typing.Iterator[_typing.Tuple[str, _typing.Any]]: ...
    def __hash__(self) -> int: ...
    def __lt__(self, other: 'Serious') -> bool: ...
    def __gt__(self, other: 'Serious') -> bool: ...
    def __le__(self, other: 'Serious') -> bool: ...
    def __ge__(self, other: 'Serious') -> bool: ...


class ComplexFieldNames(thrift.py3.exceptions.GeneratedError, _typing.Hashable, _typing.Iterable[_typing.Tuple[str, _typing.Any]]):
    class __fbthrift_IsSet:
        error_message: bool
        internal_error_message: bool
        pass

    error_message: Final[str] = ...

    internal_error_message: Final[str] = ...

    def __init__(
        self, *,
        error_message: _typing.Optional[str]=None,
        internal_error_message: _typing.Optional[str]=None
    ) -> None: ...

    def __iter__(self) -> _typing.Iterator[_typing.Tuple[str, _typing.Any]]: ...
    def __hash__(self) -> int: ...
    def __lt__(self, other: 'ComplexFieldNames') -> bool: ...
    def __gt__(self, other: 'ComplexFieldNames') -> bool: ...
    def __le__(self, other: 'ComplexFieldNames') -> bool: ...
    def __ge__(self, other: 'ComplexFieldNames') -> bool: ...


class CustomFieldNames(thrift.py3.exceptions.GeneratedError, _typing.Hashable, _typing.Iterable[_typing.Tuple[str, _typing.Any]]):
    class __fbthrift_IsSet:
        error_message: bool
        internal_error_message: bool
        pass

    error_message: Final[str] = ...

    internal_error_message: Final[str] = ...

    def __init__(
        self, *,
        error_message: _typing.Optional[str]=None,
        internal_error_message: _typing.Optional[str]=None
    ) -> None: ...

    def __iter__(self) -> _typing.Iterator[_typing.Tuple[str, _typing.Any]]: ...
    def __hash__(self) -> int: ...
    def __lt__(self, other: 'CustomFieldNames') -> bool: ...
    def __gt__(self, other: 'CustomFieldNames') -> bool: ...
    def __le__(self, other: 'CustomFieldNames') -> bool: ...
    def __ge__(self, other: 'CustomFieldNames') -> bool: ...


