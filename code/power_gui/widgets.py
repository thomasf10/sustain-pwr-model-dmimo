"""Dataclass-to-Streamlit-widget helpers.

The models are configured through dataclasses (``PowerParams``,
``OperatingPoint``, ``DMIMOConfig``). Rather than hand-wire a widget per field,
:func:`edit_dataclass` reflects over a dataclass instance and emits an
appropriate widget for each field, returning a new instance with the edited
values. This keeps the GUI in sync when a config field is added or renamed.

Dispatch is on the runtime value type (``from __future__ import annotations``
turns ``field.type`` into a string, so it cannot be trusted): ``bool`` ->
checkbox, ``Enum`` -> selectbox, ``int`` -> integer input, ``float`` -> numeric
input. Derived ``@property`` values are not dataclass fields and are therefore
skipped automatically.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Iterable, Optional, TypeVar

import streamlit as st

T = TypeVar("T")


def humanize(name: str) -> str:
    """Turn a field name into a readable label (``M_ant`` -> ``M ant``)."""
    return name.replace("_", " ")


def _widget_for(name: str, value, key: str):
    """Render one widget for ``value`` and return the (possibly new) value."""
    label = humanize(name)
    # bool must be tested before int (bool is a subclass of int).
    if isinstance(value, bool):
        return st.checkbox(label, value=value, key=key)
    if isinstance(value, Enum):
        options = list(type(value))
        return st.selectbox(label, options, index=options.index(value),
                            format_func=lambda e: e.value, key=key)
    if isinstance(value, int):
        return int(st.number_input(label, value=int(value), step=1, key=key))
    if isinstance(value, float):
        return float(st.number_input(label, value=float(value), format="%g",
                                     key=key))
    # Anything else (str, None, ...) is shown read-only and left untouched.
    st.text_input(label, value=str(value), key=key, disabled=True)
    return value


def edit_dataclass(obj: T, *, key_prefix: str = "",
                   include: Optional[Iterable[str]] = None,
                   skip: Optional[Iterable[str]] = None,
                   columns: int = 2) -> T:
    """Render editable widgets for the fields of a dataclass instance.

    Args:
        obj: A dataclass instance to edit.
        key_prefix: Prefix for Streamlit widget keys (needed when the same
            dataclass type is shown more than once on a page).
        include: If given, only these field names are shown, in this order.
        skip: Field names to omit (ignored when ``include`` is given).
        columns: Number of side-by-side columns to lay the widgets out in.

    Returns:
        A new dataclass instance with the edited values (``obj`` is unchanged).
    """
    skip = set(skip or ())
    fields = [f for f in dataclasses.fields(obj) if f.init]
    if include is not None:
        by_name = {f.name: f for f in fields}
        fields = [by_name[n] for n in include if n in by_name]
    else:
        fields = [f for f in fields if f.name not in skip]

    cols = st.columns(columns)
    updates = {}
    for i, f in enumerate(fields):
        with cols[i % columns]:
            updates[f.name] = _widget_for(
                f.name, getattr(obj, f.name), key=f"{key_prefix}{f.name}")
    return dataclasses.replace(obj, **updates)
