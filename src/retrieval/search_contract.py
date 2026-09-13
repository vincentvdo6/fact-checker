"""Keep research queries within shared character and word limits."""

from __future__ import annotations


def valid_query(value: object) -> bool:
    if not isinstance(value, str) or not value.strip() or len(value) > 600 or len(value.split()) > 75:
        return False
    return not any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in value)
