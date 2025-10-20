from collections.abc import Callable
from enum import Enum
from typing import Any


class EnumBase(str, Enum):
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Enum):
            return self.value == other.value
        return False

    __hash__ = Enum.__hash__


def extend_enum(parent_enum: EnumBase) -> Callable[[EnumBase], EnumBase]:
    """Decorator function that extends an enum class with values from another
    enum class."""

    def wrapper(extended_enum: EnumBase) -> EnumBase:
        joined = {}
        for item in parent_enum:
            joined[item.name] = item.value
        for item in extended_enum:
            joined[item.name] = item.value
        return EnumBase(extended_enum.__name__, joined)

    return wrapper
