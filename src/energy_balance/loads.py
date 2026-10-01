"""Load schedule evaluator.

Each ``LoadConfig`` is turned into a time-series power demand (W) at each step,
based on its mode:
- ``always``: constant power.
- ``sunlit_only``: power scaled by the illumination fraction.
- ``eclipse_only``: power scaled by (1 - illumination fraction).
- ``duty_cycle``: constant power multiplied by ``duty_percent / 100``.

Returns per-load and total load power time series.
"""

from __future__ import annotations

import numpy as np

from .config import LoadConfig, LoadMode


def total_load_watts(
    loads: list[LoadConfig], illumination: np.ndarray
) -> dict[str, np.ndarray]:
    n_pts = illumination.shape[0]
    per_load = np.zeros((len(loads), n_pts))
    for idx, load in enumerate(loads):
        if load.mode is LoadMode.ALWAYS:
            per_load[idx] = load.power_w
        elif load.mode is LoadMode.SUNLIT:
            per_load[idx] = load.power_w * illumination
        elif load.mode is LoadMode.ECLIPSE:
            per_load[idx] = load.power_w * (1.0 - illumination)
        elif load.mode is LoadMode.DUTY_CYCLE:
            per_load[idx] = load.power_w * (load.duty_percent / 100.0)
    total = np.sum(per_load, axis=0)
    names = [load.name for load in loads]
    return {"total": total, "per_load": per_load, "load_names": names}
