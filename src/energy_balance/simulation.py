"""Top-level simulation entry point.

Given a ``SimulationConfig``, it:
1. Resolves orbit mean elements (handling sun-synchronous shortcut).
2. Propagates the orbit over the requested time span.
3. Computes the Sun direction at each step.
4. Computes the illumination fraction with a conical shadow model.
5. Builds the body-to-ECI rotation per step for the chosen attitude.
6. Computes per-panel generated power.
7. Builds the per-step load total.
8. Integrates the battery state of charge.

Returns a tidy pandas DataFrame indexed by time.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .attitude import body_to_eci
from .battery import simulate_battery
from .config import SimulationConfig
from .eclipse import illumination_fraction_conical
from .loads import total_load_watts
from .orbit import OrbitState, propagate_orbit
from .solar_array import array_power_watts
from .sun import beta_angle_deg, sun_vector_eci


def _time_grid(duration_days: float, time_step_s: float) -> np.ndarray:
    total_s = duration_days * 86_400.0
    n = int(np.floor(total_s / time_step_s)) + 1
    return np.arange(n) * time_step_s


def run_simulation(config: SimulationConfig) -> pd.DataFrame:
    """Run the full simulation and return a tidy DataFrame."""
    epoch = config.mission.epoch
    times_s = _time_grid(config.mission.duration_days, config.mission.time_step_s)

    orbit_state = OrbitState(config.orbit, epoch)
    orbit = propagate_orbit(orbit_state, times_s)

    sun = sun_vector_eci(epoch, times_s)
    illum = illumination_fraction_conical(orbit["r_eci_km"], sun["r_sun_eci_km"])

    r_to_eci = body_to_eci(
        config.attitude,
        orbit["r_eci_km"],
        orbit["v_eci_km_s"],
        sun["u_sun_eci"],
    )

    array = array_power_watts(
        config.solar_array,
        r_to_eci,
        sun["u_sun_eci"],
        sun["distance_au"],
        illum,
    )
    loads = total_load_watts(config.loads, illum)

    battery = simulate_battery(
        config.battery,
        config.power_system,
        array["total"],
        loads["total"],
        config.mission.time_step_s,
    )

    # Beta angle time series: compute h vector from r x v.
    h_eci = np.cross(orbit["r_eci_km"], orbit["v_eci_km_s"])
    beta_deg = beta_angle_deg(h_eci, sun["u_sun_eci"])

    orbit_number = (times_s / orbit_state.period_s).astype(int)
    timestamps = [epoch + timedelta(seconds=float(t)) for t in times_s]
    altitude_km = np.linalg.norm(orbit["r_eci_km"], axis=1) - 6378.137

    df = pd.DataFrame(
        {
            "time_s": times_s,
            "timestamp_utc": timestamps,
            "orbit_number": orbit_number,
            "altitude_km": altitude_km,
            "illumination": illum,
            "in_eclipse": illum < 0.01,
            "beta_deg": beta_deg,
            "gen_w": array["total"],
            "load_w": loads["total"],
            "net_w": array["total"] - loads["total"],
            "soc_wh": battery["soc_wh"],
            "soc_percent": battery["soc_percent"],
            "shunt_w": battery["shunt_w"],
            "unmet_w": battery["unmet_w"],
        }
    )

    # Attach panel-wise power and orbit-plane ECI components as attributes.
    df.attrs["panel_names"] = array["panel_names"]
    df.attrs["panel_power_w"] = array["per_panel"]
    df.attrs["load_names"] = loads["load_names"]
    df.attrs["load_power_w"] = loads["per_load"]
    df.attrs["r_eci_km"] = orbit["r_eci_km"]
    df.attrs["v_eci_km_s"] = orbit["v_eci_km_s"]
    df.attrs["sun_eci_km"] = sun["r_sun_eci_km"]
    df.attrs["period_s"] = orbit_state.period_s
    df.attrs["inclination_deg"] = orbit_state.inclination_deg
    df.attrs["raan_deg"] = orbit_state.raan_deg
    df.attrs["raan_dot_deg_per_day"] = float(np.rad2deg(orbit_state.raan_dot) * 86400.0)
    df.attrs["semi_major_axis_km"] = orbit_state.a
    df.attrs["eccentricity"] = orbit_state.e
    df.attrs["config"] = config
    return df


__all__ = ["run_simulation"]
