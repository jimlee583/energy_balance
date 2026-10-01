"""Battery state-of-charge integrator.

Takes per-step generated power and load power (both in W), and integrates the
energy stored in the battery over time:

- Net to battery = generated - load, after a power-path efficiency.
- If net is positive, the battery charges at ``charge_efficiency``, up to
  the ``max_charge_rate_w`` and the capacity ceiling. Excess is shunted
  (wasted) and reported.
- If net is negative, the battery supplies the deficit at ``discharge_efficiency``;
  if the battery is empty, the deficit is reported as unmet load.

Returned columns:
- ``soc_wh``: battery energy stored (Wh)
- ``soc_percent``: 0..100
- ``shunt_w``: power shunted / wasted at each step (W)
- ``unmet_w``: unmet load at each step (W)
"""

from __future__ import annotations

import numpy as np

from .config import BatteryConfig, PowerSystemConfig


def simulate_battery(
    battery: BatteryConfig,
    power_system: PowerSystemConfig,
    gen_w: np.ndarray,
    load_w: np.ndarray,
    time_step_s: float,
) -> dict[str, np.ndarray]:
    gen = gen_w * power_system.path_efficiency  # delivered to bus
    net_w = gen - load_w
    n = gen.shape[0]
    soc_wh = np.zeros(n)
    shunt_w = np.zeros(n)
    unmet_w = np.zeros(n)

    cap = battery.capacity_wh
    soc = cap * battery.initial_soc_percent / 100.0
    dt_h = time_step_s / 3600.0

    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency
    max_charge_w = battery.max_charge_rate_w

    for i in range(n):
        p = net_w[i]
        if p >= 0.0:
            # Attempt to charge; may be throttled by rate or capacity.
            charge_w = min(p, max_charge_w)
            stored_w = charge_w * eta_c
            room_wh = cap - soc
            max_stored_w = room_wh / dt_h if dt_h > 0 else 0.0
            if stored_w > max_stored_w:
                stored_w = max_stored_w
                # Scale back the charge draw accordingly.
                charge_w = stored_w / eta_c if eta_c > 0 else 0.0
            soc = soc + stored_w * dt_h
            shunt_w[i] = p - charge_w
        else:
            # Discharge to meet the load deficit.
            needed_w = -p
            # Available discharge limited by stored energy.
            max_discharge_w = soc / dt_h * eta_d if dt_h > 0 else 0.0
            discharge_bus_w = min(needed_w, max_discharge_w)
            soc = soc - discharge_bus_w / eta_d * dt_h if eta_d > 0 else soc
            unmet_w[i] = needed_w - discharge_bus_w
        soc_wh[i] = max(min(soc, cap), 0.0)

    soc_percent = soc_wh / cap * 100.0 if cap > 0 else np.zeros_like(soc_wh)
    return {
        "soc_wh": soc_wh,
        "soc_percent": soc_percent,
        "shunt_w": shunt_w,
        "unmet_w": unmet_w,
    }
