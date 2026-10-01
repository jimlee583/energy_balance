"""End-to-end simulation tests via presets."""

from __future__ import annotations

from datetime import UTC

import numpy as np

from energy_balance import (
    AttitudeConfig,
    AttitudeMode,
    BatteryConfig,
    LoadConfig,
    LoadMode,
    MissionConfig,
    OrbitConfig,
    PanelConfig,
    PanelMounting,
    PowerSystemConfig,
    SimulationConfig,
    SolarArrayConfig,
    compute_metrics,
    load_config,
    preset_configs,
    run_simulation,
    save_config,
)


def test_all_presets_run_and_produce_metrics():
    for name, cfg in preset_configs().items():
        df = run_simulation(cfg)
        m = compute_metrics(df)
        assert m.period_s > 0, name
        assert len(df) > 10, name
        assert np.all(df["illumination"] >= 0.0) and np.all(df["illumination"] <= 1.0)
        assert np.all(df["soc_percent"] >= 0.0) and np.all(df["soc_percent"] <= 100.0)


def test_sso_preset_is_viable():
    cfg = preset_configs()["Sun-synchronous 550 km, 10:30 LTAN"]
    m = compute_metrics(run_simulation(cfg))
    assert m.verdict["dod_within_limit"]
    assert m.verdict["no_unmet_load"]


def test_config_round_trip(tmp_path):
    cfg = preset_configs()["ISS-like LEO (420 km, 51.6 deg)"]
    path = tmp_path / "cfg.json"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.model_dump() == cfg.model_dump()


def test_two_axis_tracking_panel_always_full_power_in_sunlight():
    cfg = SimulationConfig(
        name="tracking-check",
        mission=MissionConfig(duration_days=0.1, time_step_s=60.0),
        orbit=OrbitConfig(perigee_altitude_km=600, apogee_altitude_km=600, inclination_deg=0.0),
        attitude=AttitudeConfig(mode=AttitudeMode.INERTIAL),
        solar_array=SolarArrayConfig(
            panels=[
                PanelConfig(
                    name="tracker",
                    area_m2=1.0,
                    mounting=PanelMounting.TWO_AXIS,
                    cell_efficiency=0.3,
                    packing_factor=1.0,
                    inherent_degradation=1.0,
                    degradation_per_year=0.0,
                    temperature_loss=0.0,
                )
            ]
        ),
        loads=[LoadConfig(name="bus", power_w=0.0, mode=LoadMode.ALWAYS)],
        battery=BatteryConfig(capacity_wh=10000.0, max_charge_rate_w=10000.0),
        power_system=PowerSystemConfig(path_efficiency=1.0),
    )
    df = run_simulation(cfg)
    sunlit = df[df["illumination"] > 0.99]
    # With solar constant ~1361, area 1 m^2, eff 0.3, expect ~408 W full on.
    assert sunlit["gen_w"].mean() > 400.0
    assert sunlit["gen_w"].mean() < 420.0


def test_mission_epoch_is_tz_aware():
    cfg = preset_configs()["ISS-like LEO (420 km, 51.6 deg)"]
    assert cfg.mission.epoch.tzinfo is not None
    # And after a round trip through save/load.
    import json
    data = json.loads(cfg.model_dump_json())
    reloaded = SimulationConfig.model_validate(data)
    assert reloaded.mission.epoch.tzinfo is not None
    assert reloaded.mission.epoch.tzinfo.utcoffset(reloaded.mission.epoch) == UTC.utcoffset(reloaded.mission.epoch)
