"""Streamlit UI for the satellite energy balance simulator.

Run with::

    uv run streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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
    Regulation,
    SimulationConfig,
    SolarArrayConfig,
    compute_metrics,
    panel_frames_eci,
    preset_configs,
    run_simulation,
)
from energy_balance.attitude import body_to_eci
from energy_balance.config import BODY_FACES, OrbitSpec
from energy_balance.eclipse import R_EARTH_KM
from energy_balance.orbit import OrbitState, propagate_orbit, sun_sync_inclination_deg
from energy_balance.sun import beta_angle_deg, seasonal_sweep

st.set_page_config(page_title="Satellite Energy Balance", layout="wide")


# --------------------------------------------------------------------------------------
# Session-state helpers
# --------------------------------------------------------------------------------------


def _init_session() -> None:
    if "config" not in st.session_state:
        st.session_state.config = next(iter(preset_configs().values())).model_copy(deep=True)
    if "preset_name" not in st.session_state:
        st.session_state.preset_name = next(iter(preset_configs().keys()))
    if "n_arrays" not in st.session_state:
        st.session_state.n_arrays = len(st.session_state.config.solar_array.panels)


_PANEL_WIDGET_PREFIXES = (
    "pname", "parea", "pface", "pnx", "pny", "pnz", "pmount",
    "pax", "pay", "paz", "peff", "ppack", "pinh", "pdeg", "pyear", "ptemp",
)


def _apply_preset(name: str) -> None:
    st.session_state.config = preset_configs()[name].model_copy(deep=True)
    st.session_state.preset_name = name
    st.session_state.n_arrays = len(st.session_state.config.solar_array.panels)
    # Keyed widgets own their value once created, so clear the per-panel and
    # load keys to let the preset repopulate them.
    stale = [k for k in st.session_state if k.startswith(_PANEL_WIDGET_PREFIXES)]
    for key in stale:
        del st.session_state[key]
    st.session_state.pop("loads_editor", None)


# --------------------------------------------------------------------------------------
# Sidebar inputs
# --------------------------------------------------------------------------------------


def _mission_inputs(cfg: SimulationConfig) -> MissionConfig:
    with st.sidebar.expander("Mission", expanded=False):
        default_dt = cfg.mission.epoch.astimezone(UTC)
        date_val = st.date_input("Epoch date (UTC)", default_dt.date())
        time_val = st.time_input("Epoch time (UTC)", default_dt.time())
        if isinstance(date_val, tuple):
            date_val = date_val[0]
        if not isinstance(time_val, time):
            time_val = default_dt.time()
        epoch = datetime.combine(date_val, time_val).replace(tzinfo=UTC)
        duration = st.number_input(
            "Simulation duration (days)",
            min_value=0.05,
            max_value=365.0,
            value=float(cfg.mission.duration_days),
            step=0.1,
        )
        step = st.number_input(
            "Time step (s)",
            min_value=1.0,
            max_value=600.0,
            value=float(cfg.mission.time_step_s),
            step=1.0,
        )
    return MissionConfig(epoch=epoch, duration_days=duration, time_step_s=step)


def _orbit_inputs(cfg: SimulationConfig) -> OrbitConfig:
    with st.sidebar.expander("Orbit", expanded=True):
        spec_name = st.radio(
            "Specify orbit by",
            options=["Altitudes (perigee / apogee)", "Semi-major axis + eccentricity"],
            index=0 if cfg.orbit.spec is OrbitSpec.ALTITUDES else 1,
            horizontal=False,
        )
        spec = (
            OrbitSpec.ALTITUDES
            if spec_name.startswith("Altitudes")
            else OrbitSpec.SEMI_LATUS
        )

        perigee = cfg.orbit.perigee_altitude_km
        apogee = cfg.orbit.apogee_altitude_km
        sma = cfg.orbit.semi_major_axis_km
        ecc = cfg.orbit.eccentricity
        if spec is OrbitSpec.ALTITUDES:
            perigee = st.number_input(
                "Perigee altitude (km)", 120.0, 100_000.0, float(perigee), step=10.0
            )
            apogee = st.number_input(
                "Apogee altitude (km)", 120.0, 400_000.0, float(apogee), step=10.0
            )
        else:
            sma = st.number_input(
                "Semi-major axis (km)", 6500.0, 500_000.0, float(sma), step=10.0
            )
            ecc = st.number_input(
                "Eccentricity", 0.0, 0.95, float(ecc), step=0.001, format="%.4f"
            )

        sun_sync = st.checkbox(
            "Sun-synchronous (auto inclination from altitude)",
            value=cfg.orbit.sun_synchronous,
        )
        if sun_sync:
            a_for_sso = (
                sma if spec is OrbitSpec.SEMI_LATUS else R_EARTH_KM + 0.5 * (perigee + apogee)
            )
            e_for_sso = ecc if spec is OrbitSpec.SEMI_LATUS else 0.0
            inclination = sun_sync_inclination_deg(a_for_sso, e_for_sso)
            st.caption(f"Sun-synchronous inclination = {inclination:.3f} deg")
            ltan = st.number_input(
                "LTAN (hours)", 0.0, 24.0, float(cfg.orbit.ltan_hours), step=0.5
            )
            raan = cfg.orbit.raan_deg
        else:
            inclination = st.number_input(
                "Inclination (deg)", 0.0, 180.0, float(cfg.orbit.inclination_deg), step=0.1
            )
            raan = st.number_input(
                "RAAN (deg)", 0.0, 360.0, float(cfg.orbit.raan_deg), step=1.0
            )
            ltan = cfg.orbit.ltan_hours

        argp = st.number_input(
            "Argument of perigee (deg)", 0.0, 360.0, float(cfg.orbit.arg_perigee_deg), step=1.0
        )
        true_anomaly = st.number_input(
            "True anomaly (deg)", 0.0, 360.0, float(cfg.orbit.true_anomaly_deg), step=1.0
        )

    return OrbitConfig(
        spec=spec,
        perigee_altitude_km=perigee,
        apogee_altitude_km=apogee,
        semi_major_axis_km=sma,
        eccentricity=ecc,
        inclination_deg=inclination,
        raan_deg=raan,
        arg_perigee_deg=argp,
        true_anomaly_deg=true_anomaly,
        sun_synchronous=sun_sync,
        ltan_hours=ltan,
    )


def _attitude_inputs(cfg: SimulationConfig) -> AttitudeConfig:
    with st.sidebar.expander("Attitude", expanded=False):
        mode_label = {
            AttitudeMode.NADIR_POINTING: "Nadir pointing (+Z toward Earth)",
            AttitudeMode.SUN_POINTING: "Sun pointing (chosen axis toward Sun)",
            AttitudeMode.INERTIAL: "Inertially fixed (body == ECI)",
        }
        mode = st.selectbox(
            "Mode",
            list(mode_label.keys()),
            format_func=lambda m: mode_label[m],
            index=list(mode_label.keys()).index(cfg.attitude.mode),
        )
        nadir_yaw = cfg.attitude.nadir_yaw_axis
        sun_axis = cfg.attitude.sun_pointing_axis
        if mode is AttitudeMode.NADIR_POINTING:
            nadir_yaw = st.selectbox(
                "Yaw axis (points along velocity)",
                ["+X", "-X", "+Y", "-Y"],
                index=["+X", "-X", "+Y", "-Y"].index(cfg.attitude.nadir_yaw_axis),
            )
        if mode is AttitudeMode.SUN_POINTING:
            sun_axis = st.selectbox(
                "Body axis that points at the Sun",
                list(BODY_FACES.keys()),
                index=list(BODY_FACES.keys()).index(cfg.attitude.sun_pointing_axis),
            )
    return AttitudeConfig(mode=mode, nadir_yaw_axis=nadir_yaw, sun_pointing_axis=sun_axis)


def _panels_inputs(cfg: SimulationConfig) -> SolarArrayConfig:
    with st.sidebar.expander("Solar arrays", expanded=False):
        solar_const = st.number_input(
            "Solar constant (W/m^2)",
            1000.0,
            1500.0,
            float(cfg.solar_array.solar_constant_w_m2),
            step=1.0,
        )
        # Keyed (not `default=`) so the choice survives its own rerun: a
        # cfg-derived default changes the widget id and drops the new value.
        n_panels = st.segmented_control(
            "Number of solar arrays",
            [1, 2],
            required=True,
            format_func=lambda n: "One array" if n == 1 else "Two arrays",
            key="n_arrays",
        )
        # Slot 0 is starboard (+Y), slot 1 is port (-Y). A new array uses the
        # PanelConfig default: 1-axis about +Y with a +Z rest normal.
        sides = ["Starboard (+Y)", "Port (-Y)"]
        panels = []
        for i in range(n_panels):
            side_name = sides[i]
            default = (
                cfg.solar_array.panels[i]
                if i < len(cfg.solar_array.panels)
                else PanelConfig(name=side_name)
            )
            with st.container(border=True):
                st.markdown(f"**Array {i + 1}**")
                name = st.text_input(f"Name##p{i}", default.name, key=f"pname{i}")
                area = st.number_input(
                    f"Area m^2##p{i}", 0.01, 100.0, float(default.area_m2), 0.1, key=f"parea{i}"
                )
                face = st.selectbox(
                    f"Body-frame normal##p{i}",
                    list(BODY_FACES.keys()) + ["custom"],
                    index=(list(BODY_FACES.keys()) + ["custom"]).index(default.normal_face),
                    key=f"pface{i}",
                )
                normal_body = default.normal_body
                if face == "custom":
                    cols = st.columns(3)
                    nx = cols[0].number_input(f"nx##p{i}", -1.0, 1.0, float(default.normal_body[0]), 0.05, key=f"pnx{i}")
                    ny = cols[1].number_input(f"ny##p{i}", -1.0, 1.0, float(default.normal_body[1]), 0.05, key=f"pny{i}")
                    nz = cols[2].number_input(f"nz##p{i}", -1.0, 1.0, float(default.normal_body[2]), 0.05, key=f"pnz{i}")
                    normal_body = (nx, ny, nz)
                mounting = st.selectbox(
                    f"Mounting##p{i}",
                    list(PanelMounting),
                    format_func=lambda m: {
                        PanelMounting.FIXED: "Fixed",
                        PanelMounting.ONE_AXIS: "1-axis tracking",
                        PanelMounting.TWO_AXIS: "2-axis tracking",
                    }[m],
                    index=list(PanelMounting).index(default.mounting),
                    key=f"pmount{i}",
                )
                if mounting is PanelMounting.ONE_AXIS:
                    cols = st.columns(3)
                    ax = cols[0].number_input(f"ax##p{i}", -1.0, 1.0, float(default.rotation_axis_body[0]), 0.05, key=f"pax{i}")
                    ay = cols[1].number_input(f"ay##p{i}", -1.0, 1.0, float(default.rotation_axis_body[1]), 0.05, key=f"pay{i}")
                    az = cols[2].number_input(f"az##p{i}", -1.0, 1.0, float(default.rotation_axis_body[2]), 0.05, key=f"paz{i}")
                    rotation_axis = (ax, ay, az)
                else:
                    rotation_axis = default.rotation_axis_body
                eff = st.number_input(
                    f"Cell efficiency##p{i}",
                    0.05,
                    0.5,
                    float(default.cell_efficiency),
                    0.01,
                    key=f"peff{i}",
                )
                pack = st.number_input(
                    f"Packing factor##p{i}",
                    0.1,
                    1.0,
                    float(default.packing_factor),
                    0.05,
                    key=f"ppack{i}",
                )
                inherent = st.number_input(
                    f"Inherent degradation (I_d)##p{i}",
                    0.3,
                    1.0,
                    float(default.inherent_degradation),
                    0.01,
                    key=f"pinh{i}",
                )
                deg_year = st.number_input(
                    f"Degradation per year##p{i}",
                    0.0,
                    0.2,
                    float(default.degradation_per_year),
                    0.005,
                    key=f"pdeg{i}",
                )
                mission_year = st.number_input(
                    f"Mission year##p{i}",
                    0.0,
                    20.0,
                    float(default.mission_year),
                    0.5,
                    key=f"pyear{i}",
                )
                temp_loss = st.number_input(
                    f"Temperature loss (fraction)##p{i}",
                    0.0,
                    0.5,
                    float(default.temperature_loss),
                    0.01,
                    key=f"ptemp{i}",
                )
                panels.append(
                    PanelConfig(
                        name=name,
                        area_m2=area,
                        normal_face=face,
                        normal_body=normal_body,
                        mounting=mounting,
                        rotation_axis_body=rotation_axis,
                        cell_efficiency=eff,
                        packing_factor=pack,
                        inherent_degradation=inherent,
                        degradation_per_year=deg_year,
                        mission_year=mission_year,
                        temperature_loss=temp_loss,
                    )
                )
    return SolarArrayConfig(panels=panels, solar_constant_w_m2=solar_const)


def _loads_inputs(cfg: SimulationConfig) -> list[LoadConfig]:
    with st.sidebar.expander("Loads", expanded=False):
        mode_labels = {m.value: m for m in LoadMode}
        data = pd.DataFrame(
            [
                {
                    "name": load.name,
                    "power_w": load.power_w,
                    "mode": load.mode.value,
                    "duty_percent": load.duty_percent,
                }
                for load in cfg.loads
            ]
        )
        edited = st.data_editor(
            data,
            num_rows="dynamic",
            width='stretch',
            column_config={
                "mode": st.column_config.SelectboxColumn(
                    "mode", options=list(mode_labels.keys())
                ),
                "power_w": st.column_config.NumberColumn(min_value=0.0, format="%.1f"),
                "duty_percent": st.column_config.NumberColumn(min_value=0.0, max_value=100.0),
            },
            hide_index=True,
            key="loads_editor",
        )
        loads: list[LoadConfig] = []
        for _, row in edited.iterrows():
            if pd.isna(row.get("name")):
                continue
            try:
                loads.append(
                    LoadConfig(
                        name=str(row["name"]),
                        power_w=float(row["power_w"]),
                        mode=mode_labels.get(str(row["mode"]), LoadMode.ALWAYS),
                        duty_percent=float(row.get("duty_percent", 100.0) or 100.0),
                    )
                )
            except Exception as exc:  # pragma: no cover - UI validation
                st.warning(f"Skipping invalid load row: {exc}")
        if not loads:
            loads = [LoadConfig()]
    return loads


def _battery_inputs(cfg: SimulationConfig) -> BatteryConfig:
    with st.sidebar.expander("Battery", expanded=False):
        capacity = st.number_input(
            "Capacity (Wh)", 1.0, 100_000.0, float(cfg.battery.capacity_wh), step=10.0
        )
        soc0 = st.number_input(
            "Initial SOC (%)", 0.0, 100.0, float(cfg.battery.initial_soc_percent), step=5.0
        )
        dod = st.number_input(
            "Max depth-of-discharge (%)", 0.0, 100.0, float(cfg.battery.max_dod_percent), step=5.0
        )
        eta_c = st.number_input(
            "Charge efficiency", 0.5, 1.0, float(cfg.battery.charge_efficiency), 0.01
        )
        eta_d = st.number_input(
            "Discharge efficiency", 0.5, 1.0, float(cfg.battery.discharge_efficiency), 0.01
        )
        max_rate = st.number_input(
            "Max charge rate (W)", 1.0, 50_000.0, float(cfg.battery.max_charge_rate_w), step=10.0
        )
    return BatteryConfig(
        capacity_wh=capacity,
        initial_soc_percent=soc0,
        max_dod_percent=dod,
        charge_efficiency=eta_c,
        discharge_efficiency=eta_d,
        max_charge_rate_w=max_rate,
    )


def _power_system_inputs(cfg: SimulationConfig) -> PowerSystemConfig:
    with st.sidebar.expander("Power system", expanded=False):
        reg = st.selectbox(
            "Regulation",
            list(Regulation),
            format_func=lambda r: {
                Regulation.MPPT: "MPPT",
                Regulation.DET: "Direct Energy Transfer",
            }[r],
            index=list(Regulation).index(cfg.power_system.regulation),
        )
        eff = st.number_input(
            "Path efficiency", 0.3, 1.0, float(cfg.power_system.path_efficiency), 0.01
        )
    return PowerSystemConfig(regulation=reg, path_efficiency=eff)


def _sidebar() -> SimulationConfig:
    with st.sidebar:
        st.title("Satellite Energy Balance")
        presets = preset_configs()
        chosen = st.selectbox(
            "Preset",
            list(presets.keys()),
            index=list(presets.keys()).index(st.session_state.preset_name)
            if st.session_state.preset_name in presets
            else 0,
        )
        if chosen != st.session_state.preset_name:
            _apply_preset(chosen)
            st.rerun()
        if st.button("Reset preset"):
            _apply_preset(chosen)
            st.rerun()
        uploaded = st.file_uploader("Load config (JSON)", type=["json"])
        if uploaded is not None:
            try:
                data = json.loads(uploaded.read())
                st.session_state.config = SimulationConfig.model_validate(data)
                st.session_state.n_arrays = len(st.session_state.config.solar_array.panels)
                st.success("Config loaded.")
            except Exception as exc:
                st.error(f"Failed to parse config: {exc}")

    cfg = st.session_state.config
    mission = _mission_inputs(cfg)
    orbit = _orbit_inputs(cfg)
    attitude = _attitude_inputs(cfg)
    arrays = _panels_inputs(cfg)
    loads = _loads_inputs(cfg)
    battery = _battery_inputs(cfg)
    power_system = _power_system_inputs(cfg)

    new_cfg = SimulationConfig(
        name=cfg.name,
        mission=mission,
        orbit=orbit,
        attitude=attitude,
        solar_array=arrays,
        loads=loads,
        battery=battery,
        power_system=power_system,
    )
    st.session_state.config = new_cfg

    with st.sidebar:
        st.download_button(
            "Download config (JSON)",
            data=new_cfg.model_dump_json(indent=2),
            file_name="energy_balance_config.json",
            mime="application/json",
        )
    return new_cfg


# --------------------------------------------------------------------------------------
# Cached simulation
# --------------------------------------------------------------------------------------


@st.cache_data(show_spinner="Running simulation...")
def _cached_simulate(config_json: str) -> pd.DataFrame:
    cfg = SimulationConfig.model_validate_json(config_json)
    return run_simulation(cfg)


# --------------------------------------------------------------------------------------
# Tabs / plots
# --------------------------------------------------------------------------------------


def _render_summary(df: pd.DataFrame) -> None:
    m = compute_metrics(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Period (min)", f"{m.period_s / 60:.2f}")
    c2.metric("Inclination (deg)", f"{m.inclination_deg:.2f}")
    c3.metric("Mean eclipse (min)", f"{m.mean_eclipse_duration_s / 60:.2f}")
    c4.metric("RAAN drift (deg/day)", f"{m.raan_dot_deg_per_day:.4f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Energy in (Wh)", f"{m.energy_in_wh:,.1f}")
    c2.metric("Energy out (Wh)", f"{m.energy_out_wh:,.1f}")
    c3.metric("Shunted (Wh)", f"{m.energy_shunted_wh:,.1f}")
    c4.metric("Unmet (Wh)", f"{m.energy_unmet_wh:,.1f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min SOC (%)", f"{m.min_soc_percent:.1f}")
    c2.metric("Max observed DOD (%)", f"{m.max_dod_percent_observed:.1f}")
    c3.metric("Beta min (deg)", f"{m.beta_min_deg:.2f}")
    c4.metric("Beta max (deg)", f"{m.beta_max_deg:.2f}")

    verdict = m.verdict
    overall = "PASS" if verdict["overall_pass"] else "FAIL"
    color = "green" if verdict["overall_pass"] else "red"
    st.markdown(
        f"### Verdict: :{color}[{overall}]"
    )
    cols = st.columns(4)
    cols[0].write("Power budget >= 0: " + ("yes" if verdict["power_budget_positive"] else "NO"))
    cols[1].write("DoD within limit: " + ("yes" if verdict["dod_within_limit"] else "NO"))
    cols[2].write("No unmet load: " + ("yes" if verdict["no_unmet_load"] else "NO"))
    cols[3].write(
        "Battery recovers each orbit: " + ("yes" if verdict["battery_recovers_each_orbit"] else "NO")
    )

    st.subheader("Per-orbit energy")
    st.dataframe(m.orbit_summary, width='stretch')

    st.subheader("Sizing hints")
    sizing = m.sizing
    if sizing:
        s1, s2 = st.columns(2)
        s1.metric(
            "Minimum array area (m^2)",
            f"{sizing['min_array_area_m2']:.2f}",
            delta=f"{sizing['min_array_area_m2'] - sizing['current_array_area_m2']:+.2f} vs current",
        )
        s2.metric(
            "Minimum battery capacity (Wh)",
            f"{sizing['min_battery_capacity_wh']:.1f}",
            delta=f"{sizing['min_battery_capacity_wh'] - sizing['current_battery_capacity_wh']:+.1f} vs current",
        )
        st.caption(
            f"Worst orbit: #{sizing['worst_orbit']}, eclipse fraction "
            f"{sizing['worst_orbit_eclipse_fraction']:.3f}; "
            f"in {sizing['worst_orbit_energy_in_wh']:.1f} Wh, out "
            f"{sizing['worst_orbit_energy_out_wh']:.1f} Wh."
        )


def _render_power(df: pd.DataFrame) -> None:
    panel_names = df.attrs.get("panel_names", [])
    panel_power = df.attrs.get("panel_power_w", np.empty((0, 0)))
    load_names = df.attrs.get("load_names", [])
    load_power = df.attrs.get("load_power_w", np.empty((0, 0)))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Power (W)", "Illumination"])
    fig.add_trace(
        go.Scatter(x=df["timestamp_utc"], y=df["gen_w"], name="Generated", line=dict(color="orange")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp_utc"], y=df["load_w"], name="Load", line=dict(color="red")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp_utc"], y=df["net_w"], name="Net", line=dict(color="green", dash="dot")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp_utc"], y=df["illumination"], name="Illumination", fill="tozeroy"),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="W", row=1, col=1)
    fig.update_yaxes(title_text="0..1", row=2, col=1)
    fig.update_layout(height=500, legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch')

    with st.expander("Per-panel generated power"):
        if panel_power.size:
            per_panel_df = pd.DataFrame(
                panel_power.T,
                columns=panel_names,
                index=df["timestamp_utc"],
            )
            fig_pp = px.area(per_panel_df, title="Per-panel power (W)")
            fig_pp.update_layout(height=400, legend=dict(orientation="h"))
            st.plotly_chart(fig_pp, width='stretch')

    with st.expander("Per-load power"):
        if load_power.size:
            per_load_df = pd.DataFrame(
                load_power.T,
                columns=load_names,
                index=df["timestamp_utc"],
            )
            fig_pl = px.area(per_load_df, title="Per-load power (W)")
            fig_pl.update_layout(height=400, legend=dict(orientation="h"))
            st.plotly_chart(fig_pl, width='stretch')


def _render_battery(df: pd.DataFrame) -> None:
    cfg: SimulationConfig = df.attrs["config"]
    dod_floor = 100.0 - cfg.battery.max_dod_percent
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Battery SOC (%)", "Shunt and unmet (W)"])
    fig.add_trace(
        go.Scatter(x=df["timestamp_utc"], y=df["soc_percent"], name="SOC", line=dict(color="royalblue")),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=dod_floor,
        line=dict(color="red", dash="dash"),
        row=1,
        col=1,
        annotation_text=f"DoD limit ({dod_floor:.0f}%)",
        annotation_position="bottom right",
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp_utc"], y=df["shunt_w"], name="Shunt", fill="tozeroy", line=dict(color="goldenrod")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp_utc"], y=df["unmet_w"], name="Unmet load", fill="tozeroy", line=dict(color="firebrick")),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="%", row=1, col=1, range=[0, 101])
    fig.update_yaxes(title_text="W", row=2, col=1)
    fig.update_layout(height=550, legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch')

    st.subheader("Beta angle")
    fig_b = px.line(df, x="timestamp_utc", y="beta_deg", title="Beta angle (deg)")
    fig_b.update_layout(height=300)
    st.plotly_chart(fig_b, width='stretch')


def _render_orbit(df: pd.DataFrame) -> None:
    r = df.attrs.get("r_eci_km")
    if r is None:
        st.info("No ECI positions available.")
        return
    r_eci = np.asarray(r)
    r_sun = df.attrs.get("sun_eci_km")

    # 3D orbit view (first orbit only to keep plot light)
    period_s = df.attrs.get("period_s", 90 * 60)
    one_orbit_mask = df["time_s"] <= period_s * 1.05
    r_show = r_eci[one_orbit_mask.values]

    fig3d = go.Figure()
    # Earth sphere.
    phi = np.linspace(0, np.pi, 25)
    theta = np.linspace(0, 2 * np.pi, 40)
    px_grid, py_grid = np.meshgrid(theta, phi)
    xe = R_EARTH_KM * np.sin(py_grid) * np.cos(px_grid)
    ye = R_EARTH_KM * np.sin(py_grid) * np.sin(px_grid)
    ze = R_EARTH_KM * np.cos(py_grid)
    fig3d.add_trace(
        go.Surface(x=xe, y=ye, z=ze, colorscale="Blues", opacity=0.4, showscale=False, name="Earth")
    )
    fig3d.add_trace(
        go.Scatter3d(
            x=r_show[:, 0],
            y=r_show[:, 1],
            z=r_show[:, 2],
            mode="lines",
            line=dict(color="orange", width=4),
            name="Orbit (1 revolution)",
        )
    )
    if r_sun is not None and len(r_sun) > 0:
        sun_dir = np.asarray(r_sun[0])
        sun_dir = sun_dir / np.linalg.norm(sun_dir) * 2.5 * R_EARTH_KM
        fig3d.add_trace(
            go.Scatter3d(
                x=[0, sun_dir[0]],
                y=[0, sun_dir[1]],
                z=[0, sun_dir[2]],
                mode="lines+markers",
                line=dict(color="yellow", width=6),
                marker=dict(size=[2, 6], color="yellow"),
                name="Sun direction (not to scale)",
            )
        )
    fig3d.update_layout(
        scene=dict(aspectmode="data"),
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig3d, width='stretch')

    # Ground track (approximate: subtract Earth's rotation using GMST).
    from energy_balance.orbit import gmst_rad

    cfg: SimulationConfig = df.attrs["config"]
    epoch = cfg.mission.epoch
    gmst0 = gmst_rad(epoch)
    omega_earth = 2 * np.pi / 86164.0905
    theta_g = gmst0 + omega_earth * df["time_s"].values
    cos_g, sin_g = np.cos(theta_g), np.sin(theta_g)
    r_ecef_x = cos_g * r_eci[:, 0] + sin_g * r_eci[:, 1]
    r_ecef_y = -sin_g * r_eci[:, 0] + cos_g * r_eci[:, 1]
    r_ecef_z = r_eci[:, 2]
    lon = np.degrees(np.arctan2(r_ecef_y, r_ecef_x))
    lat = np.degrees(np.arcsin(r_ecef_z / np.linalg.norm(r_eci, axis=1)))
    gt = pd.DataFrame({"lon": lon, "lat": lat, "time_s": df["time_s"]})
    gt = gt[gt["time_s"] <= period_s * 3]  # show up to 3 revolutions
    fig_gt = px.scatter_geo(gt, lon="lon", lat="lat", title="Ground track (first ~3 orbits)")
    fig_gt.update_traces(marker=dict(size=3))
    fig_gt.update_layout(height=400, geo=dict(showcoastlines=True, projection_type="natural earth"))
    st.plotly_chart(fig_gt, width='stretch')


def _render_seasonal(cfg: SimulationConfig) -> None:
    st.caption(
        "Beta angle and eclipse fraction over one year, computed from the orbit's "
        "angular momentum vector and the Sun's position. Useful for finding worst-case "
        "cold (large eclipse) and hot (full sun) seasons."
    )
    epoch = cfg.mission.epoch
    orbit_state = OrbitState(cfg.orbit, epoch)
    # Need h_eci. For circular orbits, h is roughly constant in direction. Use r x v at t=0.
    out = propagate_orbit(orbit_state, np.array([0.0]))
    h0 = np.cross(out["r_eci_km"][0], out["v_eci_km_s"][0])
    h0 = h0 / np.linalg.norm(h0)

    times_s, u_sun = seasonal_sweep(epoch, step_days=5.0, duration_days=365.0)
    betas = beta_angle_deg(np.tile(h0, (u_sun.shape[0], 1)), u_sun)
    from energy_balance.eclipse import analytic_eclipse_fraction

    altitude_km = orbit_state.a - R_EARTH_KM
    eclipse_fracs = np.array([analytic_eclipse_fraction(altitude_km, b) for b in betas])
    days = times_s / 86400.0

    seasonal_df = pd.DataFrame(
        {
            "day_of_year": days,
            "beta_deg": betas,
            "eclipse_fraction": eclipse_fracs,
        }
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=seasonal_df["day_of_year"], y=seasonal_df["beta_deg"], name="Beta (deg)")
    )
    fig.add_trace(
        go.Scatter(
            x=seasonal_df["day_of_year"],
            y=seasonal_df["eclipse_fraction"],
            name="Eclipse fraction",
            yaxis="y2",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"Seasonal sweep (altitude {altitude_km:.0f} km, {orbit_state.inclination_deg:.2f} deg incl.)",
        xaxis_title="Days from epoch",
        yaxis_title="Beta angle (deg)",
        yaxis2_title="Eclipse fraction (0..1)",
        height=400,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')
    st.dataframe(seasonal_df, width='stretch')


# --------------------------------------------------------------------------------------
# Frames tab helpers
# --------------------------------------------------------------------------------------


def _unit_cube() -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, triangle indices) for a unit cube centered at the origin."""
    v = np.array(
        [
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ],
        dtype=float,
    ) * 0.5
    tris = np.array(
        [
            [0, 1, 2], [0, 2, 3],   # -Z
            [4, 6, 5], [4, 7, 6],   # +Z
            [0, 3, 7], [0, 7, 4],   # -X
            [1, 5, 6], [1, 6, 2],   # +X
            [0, 4, 5], [0, 5, 1],   # -Y
            [2, 6, 7], [2, 7, 3],   # +Y
        ]
    )
    return v, tris


def _box_mesh_in_frame(
    center: np.ndarray,
    rot: np.ndarray,
    size: tuple[float, float, float],
    offset: tuple[float, float, float],
    color: str,
    name: str,
    scene: str,
    *,
    showlegend: bool = False,
    opacity: float = 1.0,
) -> go.Mesh3d:
    """Axis-aligned box in a local frame, rotated by ``rot`` and placed at ``center``."""
    verts, tris = _unit_cube()
    verts_local = verts * np.asarray(size, dtype=float) + np.asarray(offset, dtype=float)
    verts_world = (rot @ verts_local.T).T + np.asarray(center, dtype=float)
    return go.Mesh3d(
        x=verts_world[:, 0], y=verts_world[:, 1], z=verts_world[:, 2],
        i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
        color=color, opacity=opacity, flatshading=True,
        name=name, hoverinfo="name", showlegend=showlegend, scene=scene,
    )


def _plate_mesh(
    center: np.ndarray,
    edge: np.ndarray,
    width: np.ndarray,
    normal: np.ndarray,
    size_edge: float,
    size_width: float,
    thickness: float,
    color: str,
    name: str,
    scene: str,
    *,
    opacity: float = 1.0,
    showlegend: bool = False,
) -> go.Mesh3d:
    """Build an oriented flat plate as a Mesh3d. (edge, width, normal) is an orthonormal basis."""
    verts, tris = _unit_cube()
    scale = np.array([size_edge, size_width, thickness], dtype=float)
    frame = np.stack(
        [
            np.asarray(edge, dtype=float),
            np.asarray(width, dtype=float),
            np.asarray(normal, dtype=float),
        ],
        axis=-1,
    )
    verts_world = (frame @ (verts * scale).T).T + np.asarray(center, dtype=float)
    return go.Mesh3d(
        x=verts_world[:, 0], y=verts_world[:, 1], z=verts_world[:, 2],
        i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
        color=color, opacity=opacity, flatshading=True,
        name=name, hoverinfo="name", showlegend=showlegend, scene=scene,
    )


def _plate_color(brightness: float) -> str:
    """Map brightness in [0, 1] to an rgb string from dark navy to bright blue."""
    b = max(0.0, min(1.0, float(brightness)))
    r = int(10 + 55 * b)
    g = int(25 + 110 * b)
    bb = int(60 + 185 * b)
    return f"rgb({r},{g},{bb})"


def _rectangle_grid(
    center: np.ndarray,
    edge: np.ndarray,
    width: np.ndarray,
    half_edge: float,
    half_width: float,
    *,
    n_edge: int = 4,
    n_width: int = 2,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Line segments for a rectangular outline plus a solar-cell grid in that plane."""
    c = np.asarray(center, dtype=float)
    e = np.asarray(edge, dtype=float)
    w = np.asarray(width, dtype=float)
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []

    def _seg(a: np.ndarray, b: np.ndarray) -> None:
        xs.extend((float(a[0]), float(b[0]), None))
        ys.extend((float(a[1]), float(b[1]), None))
        zs.extend((float(a[2]), float(b[2]), None))

    for i in range(n_edge + 1):
        t = -1.0 + 2.0 * i / n_edge
        _seg(c + e * (half_edge * t) + w * half_width, c + e * (half_edge * t) - w * half_width)
    for j in range(n_width + 1):
        t = -1.0 + 2.0 * j / n_width
        _seg(c + w * (half_width * t) + e * half_edge, c + w * (half_width * t) - e * half_edge)
    return xs, ys, zs


def _arrow_pair(
    start: np.ndarray,
    tip: np.ndarray,
    color: str,
    label: str,
    scene: str,
    *,
    dashed: bool = False,
    showlegend: bool = True,
) -> list[go.BaseTraceType]:
    """Return [line, cone] for a single arrow in the given scene."""
    s = np.asarray(start, dtype=float)
    t = np.asarray(tip, dtype=float)
    d = t - s
    norm = float(np.linalg.norm(d))
    d_unit = np.array([0.0, 0.0, 1.0]) if norm < 1e-12 else d / norm
    line = go.Scatter3d(
        x=[s[0], t[0]], y=[s[1], t[1]], z=[s[2], t[2]],
        mode="lines",
        line=dict(color=color, width=6, dash="dash" if dashed else "solid"),
        name=label, hoverinfo="name",
        showlegend=showlegend, scene=scene,
    )
    cone = go.Cone(
        x=[t[0]], y=[t[1]], z=[t[2]],
        u=[d_unit[0]], v=[d_unit[1]], w=[d_unit[2]],
        sizemode="absolute", sizeref=max(norm * 0.2, 1e-3), anchor="tip",
        colorscale=[[0, color], [1, color]],
        showscale=False, showlegend=False, hoverinfo="skip", scene=scene,
    )
    return [line, cone]


def _satellite_frame_traces(
    center: np.ndarray,
    rot_display: np.ndarray,
    sat_scale: float,
    cfg: SimulationConfig,
    panel_normal_body: np.ndarray,
    panel_edge_body: np.ndarray,
    panel_cos: np.ndarray,
    illum: float,
    sun_body: np.ndarray,
    vel_body: np.ndarray,
    nadir_body: np.ndarray,
    hhat_body: np.ndarray,
    arrow_len: float,
    sun_arrow_len: float,
    flags: dict[str, bool],
    scene: str,
    panel_plate_scale: float,
    boom_width: float,
    trail_display: np.ndarray | None = None,
    draw_solar_rays: bool = False,
) -> list[go.BaseTraceType]:
    """Build the full satellite + arrows + optional trail for one animation frame.

    All ``*_body`` inputs are expressed in the body frame; ``rot_display`` maps
    body-coords to the display frame (identity for a body-fixed close-up, the
    satellite's body-to-ECI matrix otherwise). The resulting trace list has a
    deterministic order and length so each frame produces matching data.

    When ``draw_solar_rays`` is set, each panel also gets a rectangular solar-ray
    graphic in its plane and a yellow arrow along the sun-facing normal.
    """
    show_legend_here = scene == "scene"
    traces: list[go.BaseTraceType] = []
    s = float(sat_scale)
    center = np.asarray(center, dtype=float)
    sun_d = rot_display @ np.asarray(sun_body, dtype=float)
    sun_norm = float(np.linalg.norm(sun_d))
    sun_d = np.array([1.0, 0.0, 0.0]) if sun_norm < 1e-12 else sun_d / sun_norm

    # Bus (cube of side 2*s).
    traces.append(
        _box_mesh_in_frame(
            center=center, rot=rot_display,
            size=(2 * s, 2 * s, 2 * s), offset=(0.0, 0.0, 0.0),
            color="#8d99a6", name="Bus", scene=scene, showlegend=show_legend_here,
        )
    )

    # Plates and booms.
    boom_xs: list[float | None] = []
    boom_ys: list[float | None] = []
    boom_zs: list[float | None] = []
    plate_centers_display: list[np.ndarray] = []
    axis_side: dict[tuple[float, float, float], int] = {}

    for i, panel in enumerate(cfg.solar_array.panels):
        area = max(float(panel.area_m2), 0.01)
        l_edge = max(s * 2.4 * np.sqrt(area / 4.0), s * 1.2) * panel_plate_scale
        l_width = max(s * 1.4 * panel_plate_scale, l_edge * 0.5)
        thickness = max(s * 0.08, 1e-6)

        normal_b = np.asarray(panel_normal_body[i], dtype=float)
        normal_b = normal_b / max(np.linalg.norm(normal_b), 1e-12)
        edge_b = np.asarray(panel_edge_body[i], dtype=float)
        edge_b = edge_b - np.dot(edge_b, normal_b) * normal_b
        edge_norm = np.linalg.norm(edge_b)
        if edge_norm < 1e-9:
            # Deterministic fallback perpendicular axis.
            ref = np.array([0.0, 0.0, 1.0]) if abs(normal_b[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            edge_b = ref - np.dot(ref, normal_b) * normal_b
            edge_norm = np.linalg.norm(edge_b)
        edge_b = edge_b / edge_norm
        width_b = np.cross(normal_b, edge_b)

        normal_d = rot_display @ normal_b
        edge_d = rot_display @ edge_b
        width_d = rot_display @ width_b

        if panel.mounting is PanelMounting.FIXED:
            plate_center_body = normal_b * (s + thickness * 0.5)
        else:
            axis_body = np.array(panel.rotation_axis_body, dtype=float)
            if np.linalg.norm(axis_body) < 1e-9:
                axis_body = np.array([0.0, 1.0, 0.0])
            axis_body = axis_body / np.linalg.norm(axis_body)
            key = tuple(np.round(axis_body, 6))
            sign = axis_side.get(key, +1)
            axis_side[key] = -sign
            boom_len = s + l_edge * 0.5
            plate_center_body = sign * axis_body * boom_len
            bs_d = rot_display @ (sign * axis_body * s) + center
            be_d = rot_display @ plate_center_body + center
            boom_xs += [bs_d[0], be_d[0], None]
            boom_ys += [bs_d[1], be_d[1], None]
            boom_zs += [bs_d[2], be_d[2], None]

        plate_center_display = rot_display @ plate_center_body + center
        plate_centers_display.append(plate_center_display)
        color = _plate_color(illum * float(panel_cos[i]))
        traces.append(
            _plate_mesh(
                center=plate_center_display,
                edge=edge_d, width=width_d, normal=normal_d,
                size_edge=l_edge, size_width=l_width, thickness=thickness,
                color=color, name=panel.name, scene=scene,
                opacity=0.35 if draw_solar_rays else 0.95,
                showlegend=show_legend_here,
            )
        )

        # Rectangular solar ray coming out this panel's plus or minus side.
        # The rectangle is perpendicular to the Sun, so its arrow is the
        # normal and points straight at the Sun.
        if draw_solar_rays:
            outward = plate_center_display - center
            outward_norm = float(np.linalg.norm(outward))
            if outward_norm < 1e-9:
                outward = normal_d if float(np.dot(normal_d, sun_d)) >= 0.0 else -normal_d
            else:
                outward = outward / outward_norm
            face = sun_d
            edge = outward - np.dot(outward, face) * face
            edge_norm = float(np.linalg.norm(edge))
            if edge_norm < 0.2:
                fallback = edge_d if abs(float(np.dot(edge_d, face))) < 0.9 else width_d
                edge = fallback - np.dot(fallback, face) * face
                edge_norm = float(np.linalg.norm(edge))
            edge = edge / max(edge_norm, 1e-12)
            width = np.cross(face, edge)
            ray_edge = max(l_edge * 1.15, s * 2.6)
            ray_width = max(l_width, s * 1.5)
            ray_center = plate_center_display + outward * max(0.55 * s, 0.18 * ray_edge)
            ray_center = ray_center + face * (thickness * 0.5)
            lit = illum >= 0.5
            grid_x, grid_y, grid_z = _rectangle_grid(
                ray_center, edge, width, 0.5 * ray_edge, 0.5 * ray_width,
            )
            traces.append(
                _plate_mesh(
                    center=ray_center,
                    edge=edge, width=width, normal=face,
                    size_edge=ray_edge, size_width=ray_width, thickness=thickness * 0.35,
                    color="#f6d56a" if lit else "#8a7d58",
                    name="Solar rays", scene=scene,
                    opacity=0.55 if lit else 0.28,
                    showlegend=(i == 0),
                )
            )
            traces.append(
                go.Scatter3d(
                    x=grid_x, y=grid_y, z=grid_z, mode="lines",
                    line=dict(color="#ffe38a" if lit else "#b3a57a", width=3),
                    name="Solar rays", hoverinfo="skip",
                    showlegend=False, scene=scene,
                )
            )
            traces.extend(
                _arrow_pair(
                    ray_center, ray_center + face * sun_arrow_len * 0.72,
                    "yellow", "Sun normal", scene,
                    dashed=not lit, showlegend=(i == 0),
                )
            )

    # Booms (single combined line trace, always present).
    traces.append(
        go.Scatter3d(
            x=boom_xs, y=boom_ys, z=boom_zs, mode="lines",
            line=dict(color="#5c6a7a", width=max(boom_width, 1)),
            name="Booms", hoverinfo="skip",
            showlegend=False, scene=scene,
        )
    )

    # Body axes (one combined pair of traces).
    if flags.get("body_axes"):
        axes_body = np.eye(3)
        colors = ["#d62728", "#2ca02c", "#1f77b4"]
        for dir_body, c, label in zip(
            axes_body, colors, ["+X body", "+Y body", "+Z body"], strict=True,
        ):
            dir_disp = rot_display @ dir_body
            traces.extend(
                _arrow_pair(center, center + dir_disp * arrow_len, c, label, scene,
                            showlegend=show_legend_here)
            )

    # Local-frame arrows.
    if flags.get("velocity"):
        traces.extend(
            _arrow_pair(center, center + (rot_display @ vel_body) * arrow_len,
                        "#17becf", "Velocity", scene, showlegend=show_legend_here)
        )
    if flags.get("nadir"):
        traces.extend(
            _arrow_pair(center, center + (rot_display @ nadir_body) * arrow_len,
                        "#8c564b", "Nadir", scene, showlegend=show_legend_here)
        )
    if flags.get("orbit_normal"):
        traces.extend(
            _arrow_pair(center, center + (rot_display @ hhat_body) * arrow_len,
                        "#9467bd", "Orbit normal", scene, showlegend=show_legend_here)
        )

    # Panel normals, one arrow per array. A single combined cone trace
    # sizes every head from the gap between arrows, which turns the
    # kilometer-scale Earth view into one enormous cone.
    if flags.get("panel_normals") and not draw_solar_rays:
        for i in range(len(cfg.solar_array.panels)):
            normal_b = np.asarray(panel_normal_body[i], dtype=float)
            normal_b = normal_b / max(np.linalg.norm(normal_b), 1e-12)
            normal_d = rot_display @ normal_b
            start = plate_centers_display[i]
            traces.extend(
                _arrow_pair(
                    start, start + normal_d * arrow_len * 0.55,
                    "yellow", "Panel normals", scene,
                    showlegend=show_legend_here and i == 0,
                )
            )

    # Sun arrow at the satellite.
    if flags.get("sun_arrow"):
        sun_d = rot_display @ sun_body
        traces.extend(
            _arrow_pair(
                center, center + sun_d * sun_arrow_len,
                "goldenrod", "Sun", scene,
                dashed=(illum < 0.5), showlegend=show_legend_here,
            )
        )

    # Trail (scene 1 only). Always add the trace so scenes have fixed trace counts;
    # when ``trail_display`` is None we add an invisible placeholder.
    if flags.get("trail"):
        if trail_display is not None and len(trail_display) > 1:
            traces.append(
                go.Scatter3d(
                    x=trail_display[:, 0], y=trail_display[:, 1], z=trail_display[:, 2],
                    mode="lines",
                    line=dict(color="rgba(255,170,100,0.6)", width=3),
                    name="Trail", hoverinfo="skip",
                    showlegend=False, scene=scene,
                )
            )
        else:
            traces.append(
                go.Scatter3d(
                    x=[], y=[], z=[], mode="lines",
                    line=dict(color="rgba(255,170,100,0.6)", width=3),
                    name="Trail", hoverinfo="skip",
                    showlegend=False, scene=scene, visible=False,
                )
            )

    return traces


def _shade_eclipse(fig: go.Figure, x_values: np.ndarray, in_eclipse: np.ndarray) -> None:
    """Add a translucent vertical rectangle for each contiguous eclipse span."""
    if not np.any(in_eclipse):
        return
    starts: list[float] = []
    ends: list[float] = []
    inside = False
    for i, flag in enumerate(in_eclipse):
        if flag and not inside:
            starts.append(float(x_values[i]))
            inside = True
        elif not flag and inside:
            ends.append(float(x_values[i]))
            inside = False
    if inside:
        ends.append(float(x_values[-1]))
    for i, (s, e) in enumerate(zip(starts, ends, strict=False)):
        fig.add_vrect(
            x0=s, x1=e, fillcolor="gray", opacity=0.15, line_width=0, layer="below",
            annotation_text="eclipse" if i == 0 else None,
            annotation_position="top left",
        )


def _render_frames(df: pd.DataFrame) -> None:
    cfg: SimulationConfig = df.attrs["config"]
    st.caption(
        "Earth-centered orbit view (left) and a satellite close-up (right), animated across the "
        "chosen orbit. Each configured panel is drawn with its tracked normal: fixed panels sit "
        "flush on their bus face, 1-axis arrays rotate about their gimbal axis to maximise sun "
        "incidence, and 2-axis arrays always face the Sun. Plates brighten as they face the Sun "
        "and darken in eclipse. In body-fixed close-up mode the Sun arrow sweeps around the "
        "satellite over the orbit; in inertial mode the satellite tumbles while the arrays stay "
        "pointed at the Sun. The inertial close-up draws a rectangular solar ray out the plus "
        "and minus side of each array, with a yellow arrow normal to that rectangle and pointed "
        "at the Sun. 1-axis "
        "arrays continue to track the geometric Sun direction while in eclipse (no rest-mode "
        "command is modeled)."
    )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    orbit_numbers = df["orbit_number"].to_numpy()
    max_orbit = int(orbit_numbers.max())

    c1, c2, c3, c4 = st.columns(4)
    orbit_sel = int(
        c1.number_input("Orbit to animate", min_value=0, max_value=max_orbit, value=0, step=1)
    )
    frames_target = int(c2.slider("Frames per orbit", min_value=20, max_value=240, value=90, step=5))
    frame_ms = int(c3.slider("Frame duration (ms)", min_value=20, max_value=400, value=80, step=10))
    close_up_mode = c4.radio(
        "Close-up frame", ["Body-fixed", "Inertial"], index=0, horizontal=True,
        help="Body-fixed: satellite is stationary and the Sun arrow sweeps. "
             "Inertial: satellite rotates in ECI while arrays stay on the Sun.",
    )

    ck = st.columns(4)
    show_trail = ck[0].checkbox("Orbit trail (left scene)", value=True)
    show_axes = ck[1].checkbox("Body axes", value=True)
    show_sun = ck[2].checkbox("Sun arrow at satellite", value=True)
    show_panel_normals = ck[3].checkbox("Panel normals", value=True)
    ck2 = st.columns(3)
    show_velocity = ck2[0].checkbox("Velocity", value=True)
    show_nadir = ck2[1].checkbox("Nadir", value=False)
    show_normal = ck2[2].checkbox("Orbit normal", value=False)

    # ------------------------------------------------------------------
    # Simulation slice for the chosen orbit
    # ------------------------------------------------------------------
    orbit_indices = np.where(orbit_numbers == orbit_sel)[0]
    if len(orbit_indices) == 0:
        st.info("No samples for the selected orbit.")
        return
    K = min(frames_target, len(orbit_indices))
    if len(orbit_indices) < frames_target:
        st.caption(
            f"Chosen orbit has {len(orbit_indices)} simulation steps; the animation uses all of "
            "them (shorten the time step in the Mission panel for a smoother animation)."
        )
    frame_indices = orbit_indices[np.linspace(0, len(orbit_indices) - 1, K).astype(int)]

    r_all = np.asarray(df.attrs["r_eci_km"])
    v_all = np.asarray(df.attrs["v_eci_km_s"])
    sun_all = np.asarray(df.attrs["sun_eci_km"])
    u_sun_all = sun_all / np.linalg.norm(sun_all, axis=-1, keepdims=True)

    r_k = r_all[frame_indices]
    v_k = v_all[frame_indices]
    u_sun_k = u_sun_all[frame_indices]
    illum_k = df["illumination"].to_numpy()[frame_indices]
    time_k = df["time_s"].to_numpy()[frame_indices]
    timestamp_k = df["timestamp_utc"].to_numpy()[frame_indices]

    rot_k = body_to_eci(cfg.attitude, r_k, v_k, u_sun_k)

    # Local-frame unit vectors in ECI.
    nadir_eci_k = -r_k / np.linalg.norm(r_k, axis=-1, keepdims=True)
    v_hat_eci_k = v_k / np.linalg.norm(v_k, axis=-1, keepdims=True)
    h_eci_k = np.cross(r_k, v_k)
    h_hat_eci_k = h_eci_k / np.linalg.norm(h_eci_k, axis=-1, keepdims=True)

    # Panel geometry per time step (ECI first, then body-frame via R^T).
    n_panels = len(cfg.solar_array.panels)
    normal_eci_k = np.zeros((n_panels, K, 3))
    edge_eci_k = np.zeros((n_panels, K, 3))
    cos_k = np.zeros((n_panels, K))
    angle_k = np.zeros((n_panels, K))
    for i, panel in enumerate(cfg.solar_array.panels):
        pf = panel_frames_eci(panel, rot_k, u_sun_k)
        normal_eci_k[i] = pf["normal_eci"]
        edge_eci_k[i] = pf["edge_eci"]
        cos_k[i] = pf["cos_incidence"]
        angle_k[i] = pf["angle_deg"]
    normal_body_k = np.einsum("nji,pnj->pni", rot_k, normal_eci_k)
    edge_body_k = np.einsum("nji,pnj->pni", rot_k, edge_eci_k)

    # Local-frame vectors in body coords (used by the body-fixed close-up).
    vel_body_k = np.einsum("nji,nj->ni", rot_k, v_hat_eci_k)
    nadir_body_k = np.einsum("nji,nj->ni", rot_k, nadir_eci_k)
    hhat_body_k = np.einsum("nji,nj->ni", rot_k, h_hat_eci_k)
    sun_body_k = np.einsum("nji,nj->ni", rot_k, u_sun_k)

    # ------------------------------------------------------------------
    # Scales
    # ------------------------------------------------------------------
    r_mag_all = np.linalg.norm(r_all, axis=-1)
    orbit_radius = float(np.max(r_mag_all))
    orbit_min = float(np.min(r_mag_all))
    sat_scale_s1 = max(0.04 * orbit_radius, 250.0)
    arrow_len_s1 = 0.35 * orbit_radius
    sun_distance_s1 = 1.45 * orbit_radius
    sun_arrow_len_s1 = 0.18 * orbit_radius
    earth_visual_km = min(float(R_EARTH_KM), 0.40 * orbit_min)

    sat_scale_s2 = 1.0
    arrow_len_s2 = 3.0
    sun_arrow_len_s2 = 4.5

    # ------------------------------------------------------------------
    # Static scene 1 content (Earth, full-orbit path, static Sun marker)
    # ------------------------------------------------------------------
    phi = np.linspace(0, np.pi, 25)
    theta = np.linspace(0, 2 * np.pi, 40)
    th_grid, ph_grid = np.meshgrid(theta, phi)
    xe = earth_visual_km * np.sin(ph_grid) * np.cos(th_grid)
    ye = earth_visual_km * np.sin(ph_grid) * np.sin(th_grid)
    ze = earth_visual_km * np.cos(ph_grid)
    static_s1: list[go.BaseTraceType] = [
        go.Surface(
            x=xe, y=ye, z=ze, colorscale="Blues", opacity=0.6, showscale=False,
            name="Earth (not to scale)", hoverinfo="name", scene="scene",
        )
    ]
    r_orbit_full = r_all[orbit_numbers == orbit_sel]
    if len(r_orbit_full) > 1:
        static_s1.append(
            go.Scatter3d(
                x=r_orbit_full[:, 0], y=r_orbit_full[:, 1], z=r_orbit_full[:, 2],
                mode="lines", line=dict(color="orange", width=3),
                name=f"Orbit {orbit_sel}", hoverinfo="name", scene="scene",
            )
        )
    sun_mid = u_sun_k[K // 2]
    sun_pos = sun_mid * sun_distance_s1
    static_s1.append(
        go.Scatter3d(
            x=[0.0, sun_pos[0]], y=[0.0, sun_pos[1]], z=[0.0, sun_pos[2]],
            mode="lines", line=dict(color="goldenrod", width=3, dash="dot"),
            name="Sun direction (not to scale)", hoverinfo="name", scene="scene",
        )
    )
    sun_r = sat_scale_s1 * 2.5
    xs_sun = sun_pos[0] + sun_r * np.sin(ph_grid) * np.cos(th_grid)
    ys_sun = sun_pos[1] + sun_r * np.sin(ph_grid) * np.sin(th_grid)
    zs_sun = sun_pos[2] + sun_r * np.cos(ph_grid)
    static_s1.append(
        go.Surface(
            x=xs_sun, y=ys_sun, z=zs_sun,
            colorscale=[[0, "gold"], [1, "yellow"]], opacity=1.0, showscale=False,
            name="Sun (not to scale)", hoverinfo="name", scene="scene",
        )
    )

    # ------------------------------------------------------------------
    # Build per-frame animated traces
    # ------------------------------------------------------------------
    flags_s1 = {
        "body_axes": show_axes, "velocity": show_velocity, "nadir": show_nadir,
        "orbit_normal": show_normal, "panel_normals": show_panel_normals,
        "sun_arrow": show_sun, "trail": show_trail,
    }
    flags_s2 = {**flags_s1, "trail": False}
    trail_window = max(1, K // 8)

    def _frame_traces(k: int) -> tuple[list[go.BaseTraceType], list[go.BaseTraceType]]:
        trail = r_k[max(0, k - trail_window):k + 1] if show_trail else None
        s1 = _satellite_frame_traces(
            center=r_k[k],
            rot_display=rot_k[k],
            sat_scale=sat_scale_s1,
            cfg=cfg,
            panel_normal_body=normal_body_k[:, k, :],
            panel_edge_body=edge_body_k[:, k, :],
            panel_cos=cos_k[:, k],
            illum=float(illum_k[k]),
            sun_body=sun_body_k[k],
            vel_body=vel_body_k[k],
            nadir_body=nadir_body_k[k],
            hhat_body=hhat_body_k[k],
            arrow_len=arrow_len_s1,
            sun_arrow_len=sun_arrow_len_s1,
            flags=flags_s1,
            scene="scene",
            panel_plate_scale=1.0,
            boom_width=4,
            trail_display=trail,
        )
        rot_display = np.eye(3) if close_up_mode == "Body-fixed" else rot_k[k]
        s2 = _satellite_frame_traces(
            center=np.zeros(3),
            rot_display=rot_display,
            sat_scale=sat_scale_s2,
            cfg=cfg,
            panel_normal_body=normal_body_k[:, k, :],
            panel_edge_body=edge_body_k[:, k, :],
            panel_cos=cos_k[:, k],
            illum=float(illum_k[k]),
            sun_body=sun_body_k[k],
            vel_body=vel_body_k[k],
            nadir_body=nadir_body_k[k],
            hhat_body=hhat_body_k[k],
            arrow_len=arrow_len_s2,
            sun_arrow_len=sun_arrow_len_s2,
            flags=flags_s2,
            scene="scene2",
            panel_plate_scale=1.0,
            boom_width=4,
            trail_display=None,
            draw_solar_rays=(close_up_mode == "Inertial"),
        )
        return s1, s2

    s1_0, s2_0 = _frame_traces(0)
    initial_data = static_s1 + s1_0 + s2_0
    animated_start = len(static_s1)
    animated_indices = list(range(animated_start, animated_start + len(s1_0) + len(s2_0)))

    def _title_for(k: int) -> str:
        ts = pd.Timestamp(timestamp_k[k])
        minutes = (float(time_k[k]) - float(time_k[0])) / 60.0
        return (
            f"Orbit {orbit_sel} · {ts}  (t+{minutes:5.1f} min) · "
            f"illumination {float(illum_k[k]):.2f}"
        )

    def _figure_title(k: int) -> dict:
        # Sit at the top of the paper, above the Play/Pause row.
        return dict(
            text=_title_for(k),
            x=0.5,
            xanchor="center",
            xref="paper",
            y=1.0,
            yanchor="top",
            yref="paper",
            automargin=False,
        )

    frames: list[go.Frame] = []
    for k in range(K):
        s1, s2 = _frame_traces(k)
        frames.append(
            go.Frame(
                name=str(k),
                data=s1 + s2,
                traces=animated_indices,
                layout=go.Layout(title=_figure_title(k)),
            )
        )

    # ------------------------------------------------------------------
    # Compose the two-scene figure with Play/Pause and a time slider
    # ------------------------------------------------------------------
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            "Earth-centered inertial",
            f"Satellite close-up ({close_up_mode.lower()})",
        ],
        horizontal_spacing=0.02,
    )
    for tr in initial_data:
        fig.add_trace(tr)
    fig.frames = tuple(frames)

    # Reserve a header inside the figure: title, then Play/Pause, then the
    # subplot labels. The buttons used to share the title line and covered it.
    scene_top = 0.82
    for ann in fig.layout.annotations:
        ann.update(y=scene_top, yanchor="bottom")

    axis_half_s1 = sun_distance_s1 * 1.15
    blank_axis_s1 = dict(
        title="", range=[-axis_half_s1, axis_half_s1],
        showticklabels=False, showgrid=False, zeroline=False,
        showbackground=False, showspikes=False, showline=False, ticks="",
    )
    scene_half_s2 = max(arrow_len_s2, sun_arrow_len_s2) * 1.15
    blank_axis_s2 = dict(
        title="", range=[-scene_half_s2, scene_half_s2],
        showticklabels=False, showgrid=False, zeroline=False,
        showbackground=False, showspikes=False, showline=False, ticks="",
    )

    play_args = [None, dict(
        frame=dict(duration=frame_ms, redraw=True),
        transition=dict(duration=0), fromcurrent=True, mode="immediate",
    )]
    pause_args = [[None], dict(
        frame=dict(duration=0, redraw=False),
        transition=dict(duration=0), mode="immediate",
    )]

    fig.update_layout(
        title=_figure_title(0),
        height=720,
        margin=dict(l=0, r=0, t=28, b=10),
        legend=dict(orientation="h", y=-0.04),
        scene=dict(
            aspectmode="cube", bgcolor="rgba(0,0,0,0)",
            domain=dict(y=[0.0, scene_top]),
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.95), center=dict(x=0, y=0, z=0)),
            xaxis=blank_axis_s1, yaxis=blank_axis_s1, zaxis=blank_axis_s1,
        ),
        scene2=dict(
            aspectmode="cube", bgcolor="rgba(0,0,0,0)",
            domain=dict(y=[0.0, scene_top]),
            camera=dict(eye=dict(x=2.0, y=2.0, z=1.5), center=dict(x=0, y=0, z=0)),
            xaxis=blank_axis_s2, yaxis=blank_axis_s2, zaxis=blank_axis_s2,
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            showactive=False,
            x=0.0,
            y=0.948,
            xanchor="left",
            yanchor="top",
            pad=dict(t=0, b=0, l=0, r=8),
            buttons=[
                dict(label="Play", method="animate", args=play_args),
                dict(label="Pause", method="animate", args=pause_args),
            ],
        )],
        sliders=[dict(
            active=0, pad=dict(t=40, b=10), x=0.08, len=0.9,
            currentvalue=dict(prefix="t = ", font=dict(size=12)),
            steps=[dict(
                method="animate",
                args=[[str(k)], dict(
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0), mode="immediate",
                )],
                label=f"{(float(time_k[k]) - float(time_k[0]))/60:.1f} min",
            ) for k in range(K)],
        )],
    )
    st.plotly_chart(fig, width='stretch')

    # ------------------------------------------------------------------
    # Tracking charts over the chosen orbit
    # ------------------------------------------------------------------
    minutes_k = (time_k - time_k[0]) / 60.0
    in_eclipse_k = illum_k < 0.01
    panel_names = [p.name for p in cfg.solar_array.panels]

    cos_df = pd.DataFrame(
        {name: cos_k[i] for i, name in enumerate(panel_names)},
        index=pd.Index(minutes_k, name="Minutes since orbit start"),
    )
    fig_cos = px.line(
        cos_df,
        title="cos(incidence angle) per panel",
        labels={"value": "cos(incidence)", "variable": "Panel"},
    )
    _shade_eclipse(fig_cos, minutes_k, in_eclipse_k)
    fig_cos.update_layout(height=320, legend=dict(orientation="h"))
    st.plotly_chart(fig_cos, width='stretch')

    tracking_pairs = [
        (p.name, angle_k[i])
        for i, p in enumerate(cfg.solar_array.panels)
        if p.mounting is not PanelMounting.FIXED
    ]
    if tracking_pairs:
        gimbal_df = pd.DataFrame(
            {name: angles for name, angles in tracking_pairs},
            index=pd.Index(minutes_k, name="Minutes since orbit start"),
        )
        fig_g = px.line(
            gimbal_df,
            title="Gimbal / slew angle per tracking panel (deg)",
            labels={"value": "Angle (deg)", "variable": "Panel"},
        )
        _shade_eclipse(fig_g, minutes_k, in_eclipse_k)
        fig_g.update_layout(height=300, legend=dict(orientation="h"))
        st.plotly_chart(fig_g, width='stretch')

    # ------------------------------------------------------------------
    # Inspect-frame slider feeding the numeric table
    # ------------------------------------------------------------------
    st.subheader("Inspect frame")
    insp = st.slider("Frame", min_value=0, max_value=K - 1, value=0, step=1, key="frames_inspect")
    rot_i = rot_k[insp]
    body_x = rot_i[:, 0]
    body_y = rot_i[:, 1]
    body_z = rot_i[:, 2]
    u_sun_i = u_sun_k[insp]
    nadir_i = nadir_eci_k[insp]
    v_i = v_hat_eci_k[insp]
    h_i = h_hat_eci_k[insp]

    def _vec_row(label: str, vec: np.ndarray, body_desc: str) -> dict[str, object]:
        return {
            "Vector": label,
            "ECI X": round(float(vec[0]), 4),
            "ECI Y": round(float(vec[1]), 4),
            "ECI Z": round(float(vec[2]), 4),
            "Body-frame expression": body_desc,
        }

    rows: list[dict[str, object]] = [
        _vec_row("+X body", body_x, "+X"),
        _vec_row("+Y body", body_y, "+Y"),
        _vec_row("+Z body", body_z, "+Z"),
        _vec_row("Sun", u_sun_i, "(depends on attitude)"),
        _vec_row("Nadir", nadir_i, "(depends on attitude)"),
        _vec_row("Velocity", v_i, "(depends on attitude)"),
        _vec_row("Orbit normal", h_i, ""),
    ]
    for i_p, panel in enumerate(cfg.solar_array.panels):
        n_eci = normal_eci_k[i_p, insp]
        inc_deg = float(np.rad2deg(np.arccos(np.clip(cos_k[i_p, insp], -1.0, 1.0))))
        if panel.mounting is PanelMounting.FIXED:
            face_label = panel.normal_face if panel.normal_face != "custom" else (
                f"({panel.normal_body[0]:+.2f}, {panel.normal_body[1]:+.2f}, "
                f"{panel.normal_body[2]:+.2f})"
            )
            desc = f"fixed {face_label}, incidence {inc_deg:.1f} deg"
        elif panel.mounting is PanelMounting.ONE_AXIS:
            desc = (
                f"1-axis about {panel.rotation_axis_body}, incidence {inc_deg:.1f} deg, "
                f"gimbal {float(angle_k[i_p, insp]):+.1f} deg"
            )
        else:
            desc = f"2-axis -> Sun, slew {float(angle_k[i_p, insp]):.1f} deg"
        rows.append(_vec_row(f"{panel.name} normal", n_eci, desc))

    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    _init_session()
    cfg = _sidebar()

    st.title("Satellite Energy Balance")
    st.caption(
        "Simulate an orbit, sunlight and eclipse periods, panel power, loads, and battery "
        "state of charge, then see whether the power budget closes. Use the sidebar to edit "
        "the inputs; everything recomputes automatically."
    )

    df = _cached_simulate(cfg.model_dump_json())
    # Attach config for metrics (lost through JSON cache attrs).
    df.attrs["config"] = cfg

    tabs = st.tabs(["Summary", "Power", "Battery", "Orbit", "Frames", "Seasonal"])
    with tabs[0]:
        _render_summary(df)
    with tabs[1]:
        _render_power(df)
    with tabs[2]:
        _render_battery(df)
    with tabs[3]:
        _render_orbit(df)
    with tabs[4]:
        _render_frames(df)
    with tabs[5]:
        _render_seasonal(cfg)

    st.sidebar.download_button(
        "Download results (CSV)",
        data=df.drop(columns=["timestamp_utc"]).to_csv(index=False).encode("utf-8"),
        file_name="energy_balance_timeseries.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
