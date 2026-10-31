"""Reject invalid numeric data before clipping or updating persistent state."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Number

import numpy as np


class NonFiniteError(ValueError):
    """A numeric fault, with the originating stage and field for diagnostics."""


def require_finite(source: str, **values) -> None:
    for name, value in values.items():
        field = f"{source}.{name}"
        if isinstance(value, Mapping):
            require_finite(field, **value)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                require_finite(field, **{str(i): item})
        elif isinstance(value, (np.ndarray, Number)):
            if not np.isfinite(value).all():
                raise NonFiniteError(field)
