"""Pydantic input models for the energy-balance simulator.

This module describes everything the user can enter in the UI, with validation,
sensible defaults, a set of named presets, and JSON save/load helpers.

The physics modules only depend on these models; they never import Streamlit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# --- Enumerations ----------------------------------------------------------------------


class AttitudeMode(str, Enum):
    SUN_POINTING = "sun_pointing"
    NADIR_POINTING = "nadir_pointing"
    INERTIAL = "inertial"


class PanelMounting(str, Enum):
    FIXED = "fixed"
    ONE_AXIS = "one_axis_tracking"
    TWO_AXIS = "two_axis_tracking"


class LoadMode(str, Enum):
    ALWAYS = "always"
    SUNLIT = "sunlit_only"
    ECLIPSE = "eclipse_only"
    DUTY_CYCLE = "duty_cycle"


class Regulation(str, Enum):
    MPPT = "mppt"
    DET = "direct_energy_transfer"


class OrbitSpec(str, Enum):
    ALTITUDES = "perigee_apogee_altitudes"
    SEMI_LATUS = "semi_major_axis_eccentricity"


# --- Named body-frame directions -------------------------------------------------------

BODY_FACES: dict[str, tuple[float, float, float]] = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}


# --- Config models ---------------------------------------------------------------------


class OrbitConfig(BaseModel):
    """Keplerian orbit, plus flags for sun-synchronous setup."""

    spec: OrbitSpec = OrbitSpec.ALTITUDES
    perigee_altitude_km: float = Field(500.0, ge=120.0, le=100_000.0)
    apogee_altitude_km: float = Field(500.0, ge=120.0, le=400_000.0)
    semi_major_axis_km: float = Field(6878.137, ge=6500.0, le=500_000.0)
    eccentricity: float = Field(0.0, ge=0.0, lt=1.0)
    inclination_deg: float = Field(51.6, ge=0.0, le=180.0)
    raan_deg: float = Field(0.0, ge=0.0, le=360.0)
    arg_perigee_deg: float = Field(0.0, ge=0.0, le=360.0)
    true_anomaly_deg: float = Field(0.0, ge=0.0, le=360.0)
    sun_synchronous: bool = False
    ltan_hours: float = Field(10.5, ge=0.0, le=24.0, description="Local time of ascending node")

    @model_validator(mode="after")
    def _check_altitudes(self) -> OrbitConfig:
        if self.spec is OrbitSpec.ALTITUDES and self.apogee_altitude_km < self.perigee_altitude_km:
            raise ValueError("apogee altitude must be >= perigee altitude")
        return self


class AttitudeConfig(BaseModel):
    mode: AttitudeMode = AttitudeMode.NADIR_POINTING
    nadir_yaw_axis: Literal["+X", "-X", "+Y", "-Y"] = "+X"
    sun_pointing_axis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"] = "+Z"


class PanelConfig(BaseModel):
    name: str = "Panel"
    area_m2: float = Field(1.0, gt=0.0)
    normal_face: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z", "custom"] = "+Z"
    normal_body: tuple[float, float, float] = (0.0, 0.0, 1.0)
    mounting: PanelMounting = PanelMounting.FIXED
    rotation_axis_body: tuple[float, float, float] = (0.0, 1.0, 0.0)
    cell_efficiency: float = Field(0.30, gt=0.0, le=1.0)
    packing_factor: float = Field(0.85, gt=0.0, le=1.0)
    inherent_degradation: float = Field(0.77, gt=0.0, le=1.0)
    degradation_per_year: float = Field(0.025, ge=0.0, le=1.0)
    mission_year: float = Field(0.0, ge=0.0, le=25.0)
    temperature_loss: float = Field(0.10, ge=0.0, le=1.0, description="Fractional loss, 0..1")

    @model_validator(mode="after")
    def _apply_face(self) -> PanelConfig:
        if self.normal_face != "custom":
            object.__setattr__(self, "normal_body", BODY_FACES[self.normal_face])
        return self


class SolarArrayConfig(BaseModel):
    panels: list[PanelConfig] = Field(
        default_factory=lambda: [
            PanelConfig(name="Starboard (+Y)", normal_face="+Y"),
            PanelConfig(name="Port (-Y)", normal_face="-Y"),
        ]
    )
    solar_constant_w_m2: float = Field(1361.0, gt=0.0)


class LoadConfig(BaseModel):
    name: str = "Bus"
    power_w: float = Field(20.0, ge=0.0)
    mode: LoadMode = LoadMode.ALWAYS
    duty_percent: float = Field(100.0, ge=0.0, le=100.0)


class BatteryConfig(BaseModel):
    capacity_wh: float = Field(100.0, gt=0.0)
    initial_soc_percent: float = Field(100.0, ge=0.0, le=100.0)
    max_dod_percent: float = Field(30.0, ge=0.0, le=100.0, description="Allowed depth of discharge")
    charge_efficiency: float = Field(0.95, gt=0.0, le=1.0)
    discharge_efficiency: float = Field(0.98, gt=0.0, le=1.0)
    max_charge_rate_w: float = Field(200.0, gt=0.0)


class PowerSystemConfig(BaseModel):
    regulation: Regulation = Regulation.MPPT
    path_efficiency: float = Field(0.90, gt=0.0, le=1.0)


class MissionConfig(BaseModel):
    epoch: datetime = Field(default_factory=lambda: datetime(2026, 3, 20, 12, 0, 0, tzinfo=UTC))
    duration_days: float = Field(1.0, gt=0.0, le=365.0)
    time_step_s: float = Field(30.0, gt=0.1, le=600.0)

    @field_validator("epoch")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class SimulationConfig(BaseModel):
    """Full input to a simulation run."""

    name: str = "Custom"
    mission: MissionConfig = Field(default_factory=MissionConfig)
    orbit: OrbitConfig = Field(default_factory=OrbitConfig)
    attitude: AttitudeConfig = Field(default_factory=AttitudeConfig)
    solar_array: SolarArrayConfig = Field(default_factory=SolarArrayConfig)
    loads: list[LoadConfig] = Field(default_factory=lambda: [LoadConfig()])
    battery: BatteryConfig = Field(default_factory=BatteryConfig)
    power_system: PowerSystemConfig = Field(default_factory=PowerSystemConfig)


# --- Save/load -------------------------------------------------------------------------


def save_config(config: SimulationConfig, path: str | Path) -> None:
    Path(path).write_text(config.model_dump_json(indent=2))


def load_config(path: str | Path) -> SimulationConfig:
    data = json.loads(Path(path).read_text())
    return SimulationConfig.model_validate(data)


# --- Presets ---------------------------------------------------------------------------


def _iss_like() -> SimulationConfig:
    return SimulationConfig(
        name="ISS-like LEO (420 km, 51.6 deg)",
        orbit=OrbitConfig(
            perigee_altitude_km=420.0,
            apogee_altitude_km=420.0,
            inclination_deg=51.6,
        ),
        solar_array=SolarArrayConfig(
            panels=[
                PanelConfig(name="Starboard (+Y)", normal_face="+Y", area_m2=2.0),
                PanelConfig(name="Port (-Y)", normal_face="-Y", area_m2=2.0),
            ]
        ),
        loads=[
            LoadConfig(name="Bus", power_w=40.0, mode=LoadMode.ALWAYS),
            LoadConfig(name="Payload", power_w=25.0, mode=LoadMode.SUNLIT),
            LoadConfig(name="Heater", power_w=15.0, mode=LoadMode.ECLIPSE),
        ],
        battery=BatteryConfig(capacity_wh=200.0, max_dod_percent=30.0, max_charge_rate_w=400.0),
    )


def _sso_550() -> SimulationConfig:
    cfg = SimulationConfig(
        name="Sun-synchronous 550 km, 10:30 LTAN",
        orbit=OrbitConfig(
            perigee_altitude_km=550.0,
            apogee_altitude_km=550.0,
            inclination_deg=97.6,
            sun_synchronous=True,
            ltan_hours=10.5,
        ),
        attitude=AttitudeConfig(mode=AttitudeMode.NADIR_POINTING, nadir_yaw_axis="+X"),
        solar_array=SolarArrayConfig(
            panels=[
                PanelConfig(
                    name="Deployed wing",
                    normal_face="custom",
                    normal_body=(0.0, 0.0, 1.0),
                    mounting=PanelMounting.ONE_AXIS,
                    rotation_axis_body=(0.0, 1.0, 0.0),
                    area_m2=3.0,
                )
            ]
        ),
        loads=[
            LoadConfig(name="Bus", power_w=35.0, mode=LoadMode.ALWAYS),
            LoadConfig(name="Imager", power_w=60.0, mode=LoadMode.SUNLIT, duty_percent=20.0),
        ],
        battery=BatteryConfig(capacity_wh=150.0, max_dod_percent=25.0, max_charge_rate_w=300.0),
    )
    return cfg


def _geo() -> SimulationConfig:
    return SimulationConfig(
        name="GEO (35,786 km, equatorial)",
        orbit=OrbitConfig(
            perigee_altitude_km=35_786.0,
            apogee_altitude_km=35_786.0,
            inclination_deg=0.1,
        ),
        mission=MissionConfig(duration_days=1.0, time_step_s=120.0),
        attitude=AttitudeConfig(mode=AttitudeMode.SUN_POINTING),
        solar_array=SolarArrayConfig(
            panels=[
                PanelConfig(
                    name="Sun-tracking wing",
                    normal_face="+Z",
                    mounting=PanelMounting.TWO_AXIS,
                    area_m2=8.0,
                )
            ]
        ),
        loads=[
            LoadConfig(name="Bus", power_w=400.0, mode=LoadMode.ALWAYS),
            LoadConfig(name="Payload", power_w=800.0, mode=LoadMode.ALWAYS),
        ],
        battery=BatteryConfig(capacity_wh=2500.0, max_dod_percent=80.0, max_charge_rate_w=2000.0),
    )


def preset_configs() -> dict[str, SimulationConfig]:
    """Return built-in example configurations keyed by name."""
    presets = [_iss_like(), _sso_550(), _geo()]
    return {p.name: p for p in presets}
