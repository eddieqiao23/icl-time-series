"""Small statistical helpers with pool-level uncertainty."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def mean_ci(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": math.nan, "stderr": math.nan, "ci95_low": math.nan,
                "ci95_high": math.nan, "n": 0}
    mean = float(array.mean())
    stderr = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return {"mean": mean, "stderr": stderr, "ci95_low": mean - 1.96 * stderr,
            "ci95_high": mean + 1.96 * stderr, "n": len(array)}
