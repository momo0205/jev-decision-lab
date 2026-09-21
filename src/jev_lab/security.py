from __future__ import annotations

import re
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_DENIED_KEYS = re.compile(
    r"(^|_)(api_key|authorization|token|secret|request_id|raw_response|trace)($|_)", re.IGNORECASE
)
_SECRET_VALUE = re.compile(r"(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9._-]{8,})", re.IGNORECASE)


class PublicationSafetyError(ValueError):
    pass


def assert_public_safe(value: JSONValue, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if _DENIED_KEYS.search(key):
                raise PublicationSafetyError(f"denied public field at {path}.{key}")
            assert_public_safe(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_public_safe(nested, f"{path}[{index}]")
    elif isinstance(value, str) and _SECRET_VALUE.search(value):
        raise PublicationSafetyError(f"secret-like value at {path}")


def sanitize_public(value: JSONValue) -> JSONValue:
    assert_public_safe(value)
    if isinstance(value, dict):
        return {key: sanitize_public(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [sanitize_public(nested) for nested in value]
    return value
