"""Battery integrator sanity tests."""

from __future__ import annotations

import numpy as np
import pytest

from energy_balance.battery import simulate_battery
from energy_balance.config import (
    BatteryConfig,
    BusArchitecture,
    PowerSystemConfig,
)


def _regulated_power() -> PowerSystemConfig:
    return PowerSystemConfig(
        bus_architecture=BusArchitecture.REGULATED,
        path_efficiency=1.0,
    )


def test_soc_stays_within_bounds():
    rng = np.random.default_rng(0)
    gen = rng.uniform(0, 200, size=500)
    load = rng.uniform(0, 150, size=500)
    battery = BatteryConfig(capacity_wh=100.0, initial_soc_percent=50.0, max_charge_rate_w=500.0)
    power = _regulated_power()
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
    power = _regulated_power()
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
    power = _regulated_power()
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
    power = _regulated_power()
    dt = 60.0
    out = simulate_battery(battery, power, gen, load, time_step_s=dt)
    net_in_wh = (gen - load).sum() * dt / 3600.0
    delta_soc_wh = out["soc_wh"][-1] - battery.capacity_wh * battery.initial_soc_percent / 100.0
    assert delta_soc_wh == pytest.approx(net_in_wh, abs=1e-6)


def test_default_pack_is_8s_lithium_ion_battery_dominated_bus():
    battery = BatteryConfig()
    power = PowerSystemConfig()

    assert battery.empty_ocv_v == 24.0
    assert battery.full_ocv_v == 33.6
    assert battery.internal_resistance_ohm == 0.08
    assert battery.minimum_bus_voltage_v == 24.0
    assert power.bus_architecture is BusArchitecture.BATTERY_DOMINATED


@pytest.mark.parametrize(
    ("soc_percent", "expected_voltage"),
    [(0.0, 24.0), (50.0, 28.8), (100.0, 33.6)],
)
def test_battery_dominated_open_circuit_voltage_follows_soc(
    soc_percent,
    expected_voltage,
):
    battery = BatteryConfig(
        capacity_wh=1000.0,
        initial_soc_percent=soc_percent,
    )
    out = simulate_battery(
        battery,
        PowerSystemConfig(path_efficiency=1.0),
        np.array([0.0]),
        np.array([0.0]),
        time_step_s=60.0,
    )

    assert out["bus_voltage_v"][0] == pytest.approx(expected_voltage)
    assert out["battery_current_a"][0] == pytest.approx(0.0)


def test_battery_dominated_discharge_includes_resistive_voltage_drop():
    battery = BatteryConfig(
        capacity_wh=1000.0,
        initial_soc_percent=50.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
    )
    out = simulate_battery(
        battery,
        PowerSystemConfig(path_efficiency=1.0),
        np.array([0.0]),
        np.array([100.0]),
        time_step_s=60.0,
    )

    ocv = 28.8
    resistance = 0.08
    expected_current = 2.0 * 100.0 / (
        ocv + np.sqrt(ocv**2 - 4.0 * resistance * 100.0)
    )
    assert out["battery_current_a"][0] == pytest.approx(-expected_current)
    assert out["bus_voltage_v"][0] == pytest.approx(
        ocv - expected_current * resistance
    )
    assert out["battery_power_w"][0] == pytest.approx(-100.0)
    chemical_draw_wh = 500.0 - out["soc_wh"][0]
    assert chemical_draw_wh > 100.0 / 60.0


def test_battery_dominated_charge_includes_resistive_voltage_rise():
    battery = BatteryConfig(
        capacity_wh=1000.0,
        initial_soc_percent=50.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        max_charge_rate_w=500.0,
    )
    out = simulate_battery(
        battery,
        PowerSystemConfig(path_efficiency=1.0),
        np.array([100.0]),
        np.array([0.0]),
        time_step_s=60.0,
    )

    assert out["battery_current_a"][0] > 0.0
    assert out["bus_voltage_v"][0] > 28.8
    assert out["battery_power_w"][0] == pytest.approx(100.0)


def test_battery_dominated_bus_sheds_load_at_minimum_voltage():
    battery = BatteryConfig(
        capacity_wh=1000.0,
        initial_soc_percent=50.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        minimum_bus_voltage_v=28.0,
    )
    out = simulate_battery(
        battery,
        PowerSystemConfig(path_efficiency=1.0),
        np.array([0.0]),
        np.array([400.0]),
        time_step_s=60.0,
    )

    assert out["bus_voltage_v"][0] == pytest.approx(28.0)
    assert out["unmet_w"][0] == pytest.approx(120.0)
    assert out["voltage_limited_w"][0] == pytest.approx(120.0)
