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
- ``bus_voltage_v``: battery-dominated bus voltage (NaN for a regulated bus)
- ``battery_current_a``: positive while charging, negative while discharging
- ``battery_power_w``: terminal power into the battery (negative on discharge)
- ``voltage_limited_w``: load power not served because of the minimum bus voltage
"""

from __future__ import annotations

import numpy as np

from .config import BatteryConfig, BusArchitecture, PowerSystemConfig


def _charge_current(power_w: float, ocv_v: float, resistance_ohm: float) -> float:
    """Return current for terminal charge power P = I * (Voc + I*R)."""
    if power_w <= 0.0:
        return 0.0
    if resistance_ohm <= 0.0:
        return power_w / ocv_v
    root = np.sqrt(ocv_v**2 + 4.0 * resistance_ohm * power_w)
    return 2.0 * power_w / (ocv_v + root)


def _discharge_current(power_w: float, ocv_v: float, resistance_ohm: float) -> float:
    """Return the low-current root for terminal power P = I * (Voc - I*R)."""
    if power_w <= 0.0:
        return 0.0
    if resistance_ohm <= 0.0:
        return power_w / ocv_v
    discriminant = max(ocv_v**2 - 4.0 * resistance_ohm * power_w, 0.0)
    root = np.sqrt(discriminant)
    return 2.0 * power_w / (ocv_v + root)


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
    bus_voltage_v = np.full(n, np.nan)
    battery_current_a = np.full(n, np.nan)
    battery_power_w = np.zeros(n)
    voltage_limited_w = np.zeros(n)

    cap = battery.capacity_wh
    soc = cap * battery.initial_soc_percent / 100.0
    dt_h = time_step_s / 3600.0

    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency
    max_charge_w = battery.max_charge_rate_w
    battery_dominated = (
        power_system.bus_architecture is BusArchitecture.BATTERY_DOMINATED
    )

    for i in range(n):
        p = net_w[i]
        if battery_dominated:
            soc_fraction = np.clip(soc / cap, 0.0, 1.0)
            ocv_v = battery.empty_ocv_v + soc_fraction * (
                battery.full_ocv_v - battery.empty_ocv_v
            )
            resistance = battery.internal_resistance_ohm
            bus_voltage_v[i] = ocv_v
            battery_current_a[i] = 0.0

        if p >= 0.0:
            # Attempt to charge; may be throttled by rate or capacity.
            charge_w = min(p, max_charge_w)
            if battery_dominated:
                # The full-pack OCV is also the charge-voltage ceiling, causing
                # current to taper as SOC approaches 100%.
                if resistance > 0.0:
                    voltage_current_a = max(
                        (battery.full_ocv_v - ocv_v) / resistance,
                        0.0,
                    )
                    voltage_charge_w = (
                        battery.full_ocv_v * voltage_current_a
                    )
                    charge_w = min(charge_w, voltage_charge_w)
                elif ocv_v >= battery.full_ocv_v:
                    charge_w = 0.0

                charge_current_a = _charge_current(charge_w, ocv_v, resistance)
                stored_w = ocv_v * charge_current_a * eta_c
                room_wh = cap - soc
                max_stored_w = room_wh / dt_h if dt_h > 0 else 0.0
                if stored_w > max_stored_w:
                    stored_w = max_stored_w
                    charge_current_a = (
                        stored_w / (ocv_v * eta_c)
                        if ocv_v > 0.0 and eta_c > 0.0
                        else 0.0
                    )
                    charge_w = (
                        ocv_v + charge_current_a * resistance
                    ) * charge_current_a
                bus_voltage_v[i] = ocv_v + charge_current_a * resistance
                battery_current_a[i] = charge_current_a
            else:
                stored_w = charge_w * eta_c
                room_wh = cap - soc
                max_stored_w = room_wh / dt_h if dt_h > 0 else 0.0
                if stored_w > max_stored_w:
                    stored_w = max_stored_w
                    # Scale back the charge draw accordingly.
                    charge_w = stored_w / eta_c if eta_c > 0 else 0.0
            soc = soc + stored_w * dt_h
            shunt_w[i] = p - charge_w
            battery_power_w[i] = charge_w
        else:
            # Discharge to meet the load deficit.
            needed_w = -p
            if battery_dominated:
                if resistance > 0.0 and ocv_v > battery.minimum_bus_voltage_v:
                    voltage_current_a = (
                        ocv_v - battery.minimum_bus_voltage_v
                    ) / resistance
                    voltage_power_w = (
                        battery.minimum_bus_voltage_v * voltage_current_a
                    )
                elif resistance <= 0.0 and ocv_v >= battery.minimum_bus_voltage_v:
                    voltage_current_a = np.inf
                    voltage_power_w = np.inf
                else:
                    voltage_current_a = 0.0
                    voltage_power_w = 0.0

                voltage_limited_w[i] = max(needed_w - voltage_power_w, 0.0)
                energy_current_a = (
                    soc * eta_d / (ocv_v * dt_h)
                    if dt_h > 0.0 and ocv_v > 0.0
                    else 0.0
                )
                available_current_a = min(voltage_current_a, energy_current_a)
                available_voltage_v = max(
                    ocv_v - available_current_a * resistance,
                    0.0,
                )
                max_discharge_w = available_voltage_v * available_current_a
                discharge_bus_w = min(needed_w, max_discharge_w)
                discharge_current_a = _discharge_current(
                    discharge_bus_w,
                    ocv_v,
                    resistance,
                )
                chemical_draw_w = (
                    ocv_v * discharge_current_a / eta_d
                    if eta_d > 0.0
                    else 0.0
                )
                soc -= chemical_draw_w * dt_h
                bus_voltage_v[i] = ocv_v - discharge_current_a * resistance
                battery_current_a[i] = -discharge_current_a
            else:
                # Available discharge limited by stored energy.
                max_discharge_w = soc / dt_h * eta_d if dt_h > 0 else 0.0
                discharge_bus_w = min(needed_w, max_discharge_w)
                soc = (
                    soc - discharge_bus_w / eta_d * dt_h
                    if eta_d > 0
                    else soc
                )
            unmet_w[i] = needed_w - discharge_bus_w
            battery_power_w[i] = -discharge_bus_w
        soc_wh[i] = max(min(soc, cap), 0.0)

    soc_percent = soc_wh / cap * 100.0 if cap > 0 else np.zeros_like(soc_wh)
    return {
        "soc_wh": soc_wh,
        "soc_percent": soc_percent,
        "shunt_w": shunt_w,
        "unmet_w": unmet_w,
        "bus_voltage_v": bus_voltage_v,
        "battery_current_a": battery_current_a,
        "battery_power_w": battery_power_w,
        "voltage_limited_w": voltage_limited_w,
    }
