"""Metrics, verdict, and sizing hints derived from a simulation DataFrame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimulationMetrics:
    period_s: float
    inclination_deg: float
    raan_deg: float
    raan_dot_deg_per_day: float
    mean_eclipse_fraction: float
    mean_eclipse_duration_s: float
    orbit_summary: pd.DataFrame
    energy_in_wh: float
    energy_out_wh: float
    energy_shunted_wh: float
    energy_unmet_wh: float
    min_soc_percent: float
    max_dod_percent_observed: float
    worst_orbit: int
    beta_min_deg: float
    beta_max_deg: float
    verdict: dict
    sizing: dict


def _orbit_summary(df: pd.DataFrame, dt_s: float) -> pd.DataFrame:
    g = df.groupby("orbit_number")
    gen = g["gen_w"].mean() * (dt_s * g["gen_w"].count()) / 3600.0
    load = g["load_w"].mean() * (dt_s * g["load_w"].count()) / 3600.0
    net = gen - load
    min_soc = g["soc_percent"].min()
    eclipse_frac = 1.0 - g["illumination"].mean()
    return pd.DataFrame(
        {
            "energy_in_wh": gen,
            "energy_out_wh": load,
            "net_wh": net,
            "min_soc_percent": min_soc,
            "eclipse_fraction": eclipse_frac,
            "n_steps": g["gen_w"].count(),
        }
    )


def compute_metrics(df: pd.DataFrame) -> SimulationMetrics:
    config = df.attrs["config"]
    dt_s = config.mission.time_step_s
    dt_h = dt_s / 3600.0

    energy_in = float(np.sum(df["gen_w"]) * dt_h)
    energy_out = float(np.sum(df["load_w"]) * dt_h)
    energy_shunt = float(np.sum(df["shunt_w"]) * dt_h)
    energy_unmet = float(np.sum(df["unmet_w"]) * dt_h)

    orbit_summary = _orbit_summary(df, dt_s)
    # Discard a partial last orbit if present.
    period_s = df.attrs["period_s"]
    full_steps = round(period_s / dt_s)
    full_orbits = orbit_summary[orbit_summary["n_steps"] >= 0.9 * full_steps]

    worst_orbit = (
        int(full_orbits["min_soc_percent"].idxmin()) if len(full_orbits) else 0
    )

    min_soc = float(df["soc_percent"].min())
    max_dod = 100.0 - min_soc
    dod_limit = config.battery.max_dod_percent

    orbit_recovers = bool(
        full_orbits.get("min_soc_percent", pd.Series(dtype=float)).iloc[-1] > 95.0
    ) if len(full_orbits) else False

    power_budget_positive = bool(np.mean(df["gen_w"]) >= np.mean(df["load_w"]))

    verdict = {
        "power_budget_positive": power_budget_positive,
        "dod_within_limit": max_dod <= dod_limit,
        "no_unmet_load": energy_unmet < 1e-6,
        "battery_recovers_each_orbit": orbit_recovers,
        "overall_pass": (
            power_budget_positive
            and max_dod <= dod_limit
            and energy_unmet < 1e-6
        ),
    }
    sizing = sizing_hints(df)

    return SimulationMetrics(
        period_s=period_s,
        inclination_deg=df.attrs["inclination_deg"],
        raan_deg=df.attrs["raan_deg"],
        raan_dot_deg_per_day=df.attrs["raan_dot_deg_per_day"],
        mean_eclipse_fraction=float(1.0 - df["illumination"].mean()),
        mean_eclipse_duration_s=float((1.0 - df["illumination"].mean()) * period_s),
        orbit_summary=orbit_summary,
        energy_in_wh=energy_in,
        energy_out_wh=energy_out,
        energy_shunted_wh=energy_shunt,
        energy_unmet_wh=energy_unmet,
        min_soc_percent=min_soc,
        max_dod_percent_observed=max_dod,
        worst_orbit=worst_orbit,
        beta_min_deg=float(df["beta_deg"].min()),
        beta_max_deg=float(df["beta_deg"].max()),
        verdict=verdict,
        sizing=sizing,
    )


def sizing_hints(df: pd.DataFrame) -> dict:
    """Rough minimum-array-area and minimum-battery-capacity needed to close the budget.

    These are first-order estimates computed from the worst orbit (highest eclipse
    fraction / largest energy deficit).
    """
    config = df.attrs["config"]
    dt_s = config.mission.time_step_s
    dt_h = dt_s / 3600.0
    period_s = df.attrs["period_s"]
    full_steps = round(period_s / dt_s)

    orbit_summary = _orbit_summary(df, dt_s)
    full = orbit_summary[orbit_summary["n_steps"] >= 0.9 * full_steps]
    if len(full) == 0:
        return {}
    worst = full.loc[full["net_wh"].idxmin()]

    # Minimum array area scaling: scale so that net_wh of the worst orbit >= 0.
    current_area_total = sum(p.area_m2 for p in config.solar_array.panels)
    needed_scale = worst["energy_out_wh"] / max(worst["energy_in_wh"], 1e-9)
    min_area_m2 = current_area_total * max(needed_scale, 1.0)

    # Minimum battery capacity: maximum Wh pulled from the battery in any orbit.
    deficit_wh = max(worst["energy_out_wh"] - worst["energy_in_wh"], 0.0)
    # Also consider the within-orbit swing (eclipse-only consumption).
    worst_rows = df[df["orbit_number"] == worst.name]
    eclipse_load_wh = float(np.sum(np.maximum(-worst_rows["net_w"], 0.0)) * dt_h)
    needed_capacity_wh = max(deficit_wh, eclipse_load_wh)
    # Enforce the configured DoD limit.
    dod_frac = max(config.battery.max_dod_percent / 100.0, 1e-6)
    min_capacity_wh = needed_capacity_wh / dod_frac

    return {
        "worst_orbit": int(worst.name),
        "worst_orbit_eclipse_fraction": float(worst["eclipse_fraction"]),
        "worst_orbit_energy_in_wh": float(worst["energy_in_wh"]),
        "worst_orbit_energy_out_wh": float(worst["energy_out_wh"]),
        "worst_orbit_deficit_wh": float(deficit_wh),
        "min_array_area_m2": float(min_area_m2),
        "current_array_area_m2": float(current_area_total),
        "min_battery_capacity_wh": float(min_capacity_wh),
        "current_battery_capacity_wh": float(config.battery.capacity_wh),
    }


__all__ = ["SimulationMetrics", "compute_metrics", "sizing_hints"]
