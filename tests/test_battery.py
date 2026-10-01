"""Battery integrator sanity tests."""

from __future__ import annotations

import numpy as np
import pytest

from energy_balance.battery import simulate_battery
from energy_balance.config import BatteryConfig, PowerSystemConfig


def test_soc_stays_within_bounds():
    rng = np.random.default_rng(0)
    gen = rng.uniform(0, 200, size=500)
    load = rng.uniform(0, 150, size=500)
    battery = BatteryConfig(capacity_wh=100.0, initial_soc_percent=50.0, max_charge_rate_w=500.0)
    power = PowerSystemConfig()
    out = simulate_battery(battery, power, gen, load, time_step_s=60.0)
    assert np.all(out["soc_percent"] >= 0.0)
    assert np.all(out["soc_percent"] <= 100.0)


def test_steady_charge_fills_battery():
    gen = np.full(1000, 100.0)
    load = np.full(1000, 20.0)
    battery = BatteryConfig(
        capacity_wh=100.0,
        initial_soc_percent=0.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        max_charge_rate_w=500.0,
    )
    power = PowerSystemConfig(path_efficiency=1.0)
    out = simulate_battery(battery, power, gen, load, time_step_s=60.0)
    assert out["soc_percent"][-1] == 100.0
    # Shunt should start being nonzero once battery hits the ceiling.
    assert np.any(out["shunt_w"] > 0.0)


def test_steady_discharge_drains_battery_and_reports_unmet():
    gen = np.zeros(600)
    load = np.full(600, 50.0)
    battery = BatteryConfig(
        capacity_wh=10.0,
        initial_soc_percent=100.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        max_charge_rate_w=500.0,
    )
    power = PowerSystemConfig(path_efficiency=1.0)
    out = simulate_battery(battery, power, gen, load, time_step_s=60.0)
    assert out["soc_percent"][-1] == 0.0
    assert np.any(out["unmet_w"] > 0.0)


def test_energy_conservation_closed_system():
    gen = np.full(100, 100.0)
    load = np.full(100, 60.0)
    battery = BatteryConfig(
        capacity_wh=10_000.0,
        initial_soc_percent=50.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        max_charge_rate_w=10_000.0,
    )
    power = PowerSystemConfig(path_efficiency=1.0)
    dt = 60.0
    out = simulate_battery(battery, power, gen, load, time_step_s=dt)
    net_in_wh = (gen - load).sum() * dt / 3600.0
    delta_soc_wh = out["soc_wh"][-1] - battery.capacity_wh * battery.initial_soc_percent / 100.0
    assert delta_soc_wh == pytest.approx(net_in_wh, abs=1e-6)
