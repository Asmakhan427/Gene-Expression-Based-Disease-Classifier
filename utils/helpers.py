"""
utils/helpers.py
================
Helper utilities.
"""

from __future__ import annotations

import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def timer(func: Callable) -> Callable:
    """Decorator that logs the wall-clock time of a function call."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"⏱  {func.__qualname__} completed in {elapsed:.2f} s")
        return result
    return wrapper


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Return the memory usage of a DataFrame in megabytes."""
    return df.memory_usage(deep=True).sum() / 1_048_576


def set_global_seed(seed: int = 42) -> None:
    """Set NumPy random seed for reproducibility."""
    np.random.seed(seed)