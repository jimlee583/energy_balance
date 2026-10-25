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


class BusArchitecture(str, Enum):
    REGULATED = "regulated"
    BATTERY_DOMINATED = "battery_dominated"


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
    perigee_altitude_km: float = Field(1000.0, ge=120.0, le=100_000.0)
    apogee_altitude_km: float = Field(1000.0, ge=120.0, le=400_000.0)
    semi_major_axis_km: float = Field(7378.137, ge=6500.0, le=500_000.0)
    eccentricity: float = Field(0.0, ge=0.0, lt=1.0)
    inclination_deg: float = Field(80.0, ge=0.0, le=180.0)
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
    area_m2: float = Field(4.0, gt=0.0)
    # Rest normal is +Z so it stays perpendicular to the default ±X wing axis.
    normal_face: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z", "custom"] = "+Z"
    normal_body: tuple[float, float, float] = (0.0, 0.0, 1.0)
    mounting: PanelMounting = PanelMounting.TWO_AXIS
    rotation_axis_body: tuple[float, float, float] = (1.0, 0.0, 0.0)
    gimbal_min_deg: float | None = None
    gimbal_max_deg: float | None = None
    gimbal2_min_deg: float | None = None
    gimbal2_max_deg: float | None = None
    max_slew_rate_deg_s: float = Field(1.0, gt=0.0)
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

        # Primary-axis gimbal limit: legal for 1-axis and 2-axis panels. A span
        # >= 360 deg means the gimbal can always rewind to a 360-deg-equivalent
        # angle and keep tracking the Sun exactly (cable-wrap mode); a span
        # < 360 deg is a hard mechanical stop where the Sun can go out of reach
        # and incidence degrades.
        limits = (self.gimbal_min_deg, self.gimbal_max_deg)
        if (limits[0] is None) != (limits[1] is None):
            raise ValueError("gimbal minimum and maximum must be supplied together")
        if limits[0] is not None and limits[1] is not None:
            if self.mounting not in (PanelMounting.ONE_AXIS, PanelMounting.TWO_AXIS):
                raise ValueError("gimbal limits are only supported for 1-axis and 2-axis panels")
            if not limits[0] <= 0.0 <= limits[1]:
                raise ValueError("gimbal limits must contain the 0 deg rest position")
            if limits[1] <= limits[0]:
                raise ValueError("gimbal maximum must exceed gimbal minimum")

        # Secondary-axis gimbal limit: only meaningful for 2-axis panels. The
        # secondary angle is bounded to +/-90 deg by the tracking geometry
        # (it is always a hard stop; a cable-wrap rewind mode is impossible).
        limits2 = (self.gimbal2_min_deg, self.gimbal2_max_deg)
        if (limits2[0] is None) != (limits2[1] is None):
            raise ValueError("secondary gimbal minimum and maximum must be supplied together")
        if limits2[0] is not None and limits2[1] is not None:
            if self.mounting is not PanelMounting.TWO_AXIS:
                raise ValueError("secondary gimbal limits are only supported for 2-axis panels")
            if not (-90.0 <= limits2[0] <= 0.0 <= limits2[1] <= 90.0):
                raise ValueError(
                    "secondary gimbal limits must contain 0 deg and lie within +/-90 deg"
                )
            if limits2[1] <= limits2[0]:
                raise ValueError("secondary gimbal maximum must exceed secondary gimbal minimum")
        return self


class SolarArrayConfig(BaseModel):
    """One or two solar arrays; two is the default, on the +X and -X sides."""

    panels: list[PanelConfig] = Field(
        default_factory=lambda: [
            # Wings on the ±X booms, both tracking on two axes.
            PanelConfig(name="Wing (+X)"),
            PanelConfig(name="Wing (-X)"),
        ],
        min_length=1,
        max_length=2,
    )
    solar_constant_w_m2: float = Field(1361.0, gt=0.0)


class LoadConfig(BaseModel):
    name: str = "Payload"
    power_w: float = Field(800.0, ge=0.0)
    mode: LoadMode = LoadMode.ALWAYS
    duty_percent: float = Field(100.0, ge=0.0, le=100.0)


class BatteryConfig(BaseModel):
    capacity_wh: float = Field(1000.0, gt=0.0)
    initial_soc_percent: float = Field(100.0, ge=0.0, le=100.0)
    max_dod_percent: float = Field(30.0, ge=0.0, le=100.0, description="Allowed depth of discharge")
    charge_efficiency: float = Field(0.95, gt=0.0, le=1.0)
    discharge_efficiency: float = Field(0.98, gt=0.0, le=1.0)
    max_charge_rate_w: float = Field(500.0, gt=0.0)
    empty_ocv_v: float = Field(
        24.0,
        gt=0.0,
        description="Pack open-circuit voltage at 0% SOC (8s Li-ion default)",
    )
    full_ocv_v: float = Field(
        33.6,
        gt=0.0,
        description="Pack open-circuit voltage at 100% SOC (8s Li-ion default)",
    )
    internal_resistance_ohm: float = Field(
        0.08,
        ge=0.0,
        description="Equivalent pack series resistance",
    )
    minimum_bus_voltage_v: float = Field(
        24.0,
        gt=0.0,
        description="Minimum usable bus voltage before load shedding",
    )

    @model_validator(mode="after")
    def _check_pack_voltages(self) -> BatteryConfig:
        if self.full_ocv_v <= self.empty_ocv_v:
            raise ValueError("full battery OCV must be greater than empty battery OCV")
        if self.minimum_bus_voltage_v > self.full_ocv_v:
            raise ValueError("minimum bus voltage must not exceed full battery OCV")
        return self


class PowerSystemConfig(BaseModel):
    regulation: Regulation = Regulation.MPPT
    bus_architecture: BusArchitecture = BusArchitecture.BATTERY_DOMINATED
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


def _leo_1000() -> SimulationConfig:
    """Opening configuration: 1000 km circular orbit at 80 deg inclination."""
    return SimulationConfig(name="LEO 1000 km, 80 deg")


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
                PanelConfig(
                    name="Starboard (+Y)",
                    area_m2=2.0,
                    mounting=PanelMounting.ONE_AXIS,
                    rotation_axis_body=(0.0, 1.0, 0.0),
                ),
                PanelConfig(
                    name="Port (-Y)",
                    area_m2=2.0,
                    mounting=PanelMounting.ONE_AXIS,
                    rotation_axis_body=(0.0, 1.0, 0.0),
                ),
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
                # Both wings share the +Y drive axis, so they are drawn on the
                # +Y and -Y booms. The rest normal must stay perpendicular to
                # that axis for the gimbal angle to be meaningful.
                PanelConfig(
                    name="Starboard wing (+Y)",
                    normal_face="custom",
                    normal_body=(0.0, 0.0, 1.0),
                    mounting=PanelMounting.ONE_AXIS,
                    rotation_axis_body=(0.0, 1.0, 0.0),
                    area_m2=1.5,
                ),
                PanelConfig(
                    name="Port wing (-Y)",
                    normal_face="custom",
                    normal_body=(0.0, 0.0, 1.0),
                    mounting=PanelMounting.ONE_AXIS,
                    rotation_axis_body=(0.0, 1.0, 0.0),
                    area_m2=1.5,
                ),
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
                    name="Starboard wing (+Y)",
                    normal_face="+Z",
                    mounting=PanelMounting.TWO_AXIS,
                    rotation_axis_body=(0.0, 1.0, 0.0),
                    area_m2=4.0,
                ),
                PanelConfig(
                    name="Port wing (-Y)",
                    normal_face="+Z",
                    mounting=PanelMounting.TWO_AXIS,
                    rotation_axis_body=(0.0, 1.0, 0.0),
                    area_m2=4.0,
                ),
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
    presets = [_leo_1000(), _iss_like(), _sso_550(), _geo()]
    return {p.name: p for p in presets}
