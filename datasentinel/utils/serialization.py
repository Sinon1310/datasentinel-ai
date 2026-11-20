"""Conversion of engine results into JSON-safe Python builtins."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


def to_builtin(obj: Any) -> Any:
    """Recursively convert numpy / pandas / dataclass / enum values to JSON-safe builtins.

    ``NaN``, ``NaT`` and infinities become ``None`` so the output is valid JSON.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_builtin(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if obj is pd.NaT:
        return None
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return None if np.isnat(obj) else pd.Timestamp(obj).isoformat()
    if isinstance(obj, np.ndarray):
        return [to_builtin(v) for v in obj.tolist()]
    if obj is pd.NA:
        return None
    return obj
