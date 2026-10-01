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


def _apply_preset(name: str) -> None:
    st.session_state.config = preset_configs()[name].model_copy(deep=True)
    st.session_state.preset_name = name


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
        n_panels = st.number_input(
            "Number of panels",
            1,
            12,
            len(cfg.solar_array.panels),
        )
        panels = []
        for i in range(n_panels):
            default = (
                cfg.solar_array.panels[i]
                if i < len(cfg.solar_array.panels)
                else PanelConfig(name=f"Panel {i + 1}")
            )
            with st.container(border=True):
                st.markdown(f"**Panel {i + 1}**")
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


def _arrow3d(
    label: str,
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    color: str,
    *,
    dashed: bool = False,
) -> list[go.BaseTraceType]:
    """Render a labeled 3D arrow of length ``length`` starting at ``start``.

    ``direction`` must be a unit vector. The returned traces are a line segment
    plus a cone arrowhead at the tip.
    """
    s = np.asarray(start, dtype=float)
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        return []
    d = d / norm
    tip = s + d * length
    line = go.Scatter3d(
        x=[s[0], tip[0]],
        y=[s[1], tip[1]],
        z=[s[2], tip[2]],
        mode="lines",
        line=dict(color=color, width=6, dash="dash" if dashed else "solid"),
        name=label,
        hovertemplate=f"<b>{label}</b><br>Dir: ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f})<extra></extra>",
        showlegend=True,
    )
    cone = go.Cone(
        x=[tip[0]], y=[tip[1]], z=[tip[2]],
        u=[d[0]], v=[d[1]], w=[d[2]],
        sizemode="absolute",
        sizeref=max(length * 0.2, 1e-3),
        anchor="tip",
        colorscale=[[0, color], [1, color]],
        showscale=False,
        showlegend=False,
        hoverinfo="skip",
    )
    return [line, cone]


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


def _box_mesh(
    center_eci: np.ndarray,
    rot_body_to_eci: np.ndarray,
    size_body: tuple[float, float, float],
    offset_body: tuple[float, float, float],
    color: str,
    name: str,
    *,
    showlegend: bool = False,
    opacity: float = 1.0,
) -> go.Mesh3d:
    """Build an arbitrarily sized box in body coords, rotated and translated to ECI."""
    verts, tris = _unit_cube()
    scale = np.array(size_body, dtype=float)
    offset = np.array(offset_body, dtype=float)
    verts_body = verts * scale + offset
    verts_eci = (rot_body_to_eci @ verts_body.T).T + np.asarray(center_eci, dtype=float)
    return go.Mesh3d(
        x=verts_eci[:, 0], y=verts_eci[:, 1], z=verts_eci[:, 2],
        i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
        color=color, opacity=opacity, flatshading=True,
        name=name, hoverinfo="name", showlegend=showlegend,
    )


def _satellite_mesh(
    center_eci: np.ndarray,
    rot_body_to_eci: np.ndarray,
    scale: float,
) -> list[go.BaseTraceType]:
    """Return a notional box-bus-plus-two-wings satellite rotated into ECI.

    ``scale`` is roughly the half-size of the bus in km. Wings extend from the +/-Y
    faces of the bus and are rendered as thin flat boxes.
    """
    traces: list[go.BaseTraceType] = []
    s = float(scale)
    # Bus: cube of side 2*s, centered on the satellite.
    traces.append(
        _box_mesh(
            center_eci, rot_body_to_eci,
            size_body=(2 * s, 2 * s, 2 * s), offset_body=(0.0, 0.0, 0.0),
            color="#8d99a6", name="Bus", showlegend=True,
        )
    )
    # Wings: thin flat boxes along +Y and -Y, each about 3x the bus width.
    wing_x = 4 * s     # along body X
    wing_y = 3 * s     # extent along Y past the bus face
    wing_z = 0.1 * s   # thin in Z
    wing_offset = s + wing_y / 2.0
    for side, label in [(+1, "Wing +Y"), (-1, "Wing -Y")]:
        traces.append(
            _box_mesh(
                center_eci, rot_body_to_eci,
                size_body=(wing_x, wing_y, wing_z),
                offset_body=(0.0, side * wing_offset, 0.0),
                color="#1f3a5f", name=label, opacity=0.95,
            )
        )
    return traces


def _pick_default_time_index(df: pd.DataFrame) -> int:
    """Return the first index that is comfortably in sunlight."""
    sunlit = df.index[df["illumination"] > 0.99]
    if len(sunlit):
        return int(sunlit[0])
    return int(df["illumination"].idxmax())


def _render_frames(df: pd.DataFrame) -> None:
    cfg: SimulationConfig = df.attrs["config"]
    st.caption(
        "Earth-centered scene showing the orbit, a notional satellite (box bus plus two wings), "
        "and the direction to the Sun. Earth stays at the origin but is drawn smaller than true "
        "scale, so a low orbit and the +X, +Y, and +Z body axes are not buried in the globe. "
        "The satellite geometry is schematic and not to scale; it rotates with the chosen "
        "attitude mode. The Sun marker is placed along the true Sun direction but at a "
        "dramatically reduced distance so everything is visible. Scrub the time slider to watch "
        "the satellite move along its orbit and see how its body axes line up with Earth and Sun."
    )

    # Time picker: quick preset + fine slider wired through session state so the
    # quick pick actually jumps the slider.
    n = len(df)
    quick = st.radio(
        "Quick pick",
        options=["Default (first sunlit)", "Start", "Mid-sunlit", "Mid-eclipse", "End"],
        horizontal=True,
        index=0,
        key="frames_quick",
    )

    def _resolve_default() -> int:
        if quick == "Start":
            return 0
        if quick == "End":
            return n - 1
        if quick == "Mid-sunlit":
            sunlit = df.index[df["illumination"] > 0.99]
            return int(sunlit[len(sunlit) // 2]) if len(sunlit) else _pick_default_time_index(df)
        if quick == "Mid-eclipse":
            eclipse = df.index[df["illumination"] < 0.01]
            return int(eclipse[len(eclipse) // 2]) if len(eclipse) else _pick_default_time_index(df)
        return _pick_default_time_index(df)

    # Keep the slider's session-state in sync with quick picks and simulation length.
    if "frames_idx" not in st.session_state:
        st.session_state.frames_idx = _resolve_default()
    if st.session_state.get("frames_quick_prev") != quick:
        st.session_state.frames_quick_prev = quick
        st.session_state.frames_idx = _resolve_default()
    # Clamp if the simulation length changed under us.
    st.session_state.frames_idx = int(min(max(st.session_state.frames_idx, 0), n - 1))

    idx = st.slider(
        "Time step",
        min_value=0,
        max_value=n - 1,
        step=1,
        key="frames_idx",
        help="Scrub through the simulation. The satellite and vectors update live.",
    )
    ts = df["timestamp_utc"].iloc[idx]
    illum = float(df["illumination"].iloc[idx])
    st.markdown(
        f"**Time:** {ts}  &middot;  **Step {idx} of {n - 1}**  &middot;  "
        f"**Illumination:** {illum:.2f}"
    )

    cols = st.columns(3)
    show_orbit = cols[0].checkbox("Orbit trajectory", value=True)
    show_sun = cols[0].checkbox("Sun marker and direction", value=True)
    show_body_axes = cols[1].checkbox("Body axes on satellite", value=True)
    show_nadir = cols[1].checkbox("Nadir arrow", value=False)
    show_velocity = cols[2].checkbox("Velocity arrow", value=False)
    show_normal = cols[2].checkbox("Orbit normal arrow", value=False)

    # Vectors and attitude at the chosen step.
    r_all = np.asarray(df.attrs["r_eci_km"])
    r_eci = r_all[idx]
    v_eci = np.asarray(df.attrs["v_eci_km_s"][idx])
    sun_eci = np.asarray(df.attrs["sun_eci_km"][idx])
    u_sun = sun_eci / np.linalg.norm(sun_eci)
    nadir = -r_eci / np.linalg.norm(r_eci)
    v_hat = v_eci / np.linalg.norm(v_eci)
    h_hat = np.cross(r_eci, v_eci)
    h_hat = h_hat / np.linalg.norm(h_hat)

    rot = body_to_eci(cfg.attitude, r_eci[None, :], v_eci[None, :], u_sun[None, :])[0]
    body_x = rot[:, 0]
    body_y = rot[:, 1]
    body_z = rot[:, 2]

    # Visual scales (km). Notional satellite size and Sun distance are chosen so
    # that everything stays visible for both LEO and GEO orbits.
    r_mag = np.linalg.norm(r_all, axis=1)
    orbit_radius = float(np.max(r_mag))
    orbit_min = float(np.min(r_mag))
    sat_scale = max(0.04 * orbit_radius, 250.0)
    arrow_len = 3.0 * sat_scale
    sun_distance = 2.0 * orbit_radius
    # Draw Earth smaller than true scale and centered at the origin. A real Earth
    # nearly fills a LEO orbit, which hides the trajectory and the body axes.
    # Cap at the true radius so higher orbits are not enlarged.
    earth_visual_km = min(float(R_EARTH_KM), 0.40 * orbit_min)

    traces: list[go.BaseTraceType] = []

    # Earth sphere (schematic, centered on the ECI origin).
    phi = np.linspace(0, np.pi, 25)
    theta = np.linspace(0, 2 * np.pi, 40)
    th_grid, ph_grid = np.meshgrid(theta, phi)
    xe = earth_visual_km * np.sin(ph_grid) * np.cos(th_grid)
    ye = earth_visual_km * np.sin(ph_grid) * np.sin(th_grid)
    ze = earth_visual_km * np.cos(ph_grid)
    traces.append(
        go.Surface(
            x=xe, y=ye, z=ze, colorscale="Blues", opacity=0.6, showscale=False,
            name="Earth (not to scale)", hoverinfo="name",
        )
    )

    # Orbit trajectory (one revolution).
    period_s = df.attrs.get("period_s", 90.0 * 60.0)
    orbit_mask = df["time_s"].values <= period_s * 1.02
    r_orbit = r_all[orbit_mask]
    if show_orbit and len(r_orbit) > 1:
        traces.append(
            go.Scatter3d(
                x=r_orbit[:, 0], y=r_orbit[:, 1], z=r_orbit[:, 2],
                mode="lines", line=dict(color="orange", width=4),
                name="Orbit (1 revolution)", hoverinfo="name",
            )
        )

    # Notional satellite geometry at the current step.
    traces += _satellite_mesh(r_eci, rot, sat_scale)

    # Body axes and (optional) local-frame arrows anchored on the satellite.
    if show_body_axes:
        traces += _arrow3d("+X body", r_eci, body_x, arrow_len, "#d62728")
        traces += _arrow3d("+Y body", r_eci, body_y, arrow_len, "#2ca02c")
        traces += _arrow3d("+Z body", r_eci, body_z, arrow_len, "#1f77b4")
    if show_nadir:
        traces += _arrow3d("Nadir", r_eci, nadir, arrow_len, "#8c564b")
    if show_velocity:
        traces += _arrow3d("Velocity", r_eci, v_hat, arrow_len, "#17becf")
    if show_normal:
        traces += _arrow3d("Orbit normal", r_eci, h_hat, arrow_len, "#9467bd")

    # Sun marker along the Sun direction (not to scale).
    if show_sun:
        sun_pos = u_sun * sun_distance
        traces.append(
            go.Scatter3d(
                x=[0.0, sun_pos[0]], y=[0.0, sun_pos[1]], z=[0.0, sun_pos[2]],
                mode="lines",
                line=dict(color="goldenrod", width=3, dash="dot"),
                name="Sun direction (not to scale)", hoverinfo="name",
            )
        )
        sun_r = sat_scale * 2.5
        xs = sun_pos[0] + sun_r * np.sin(ph_grid) * np.cos(th_grid)
        ys = sun_pos[1] + sun_r * np.sin(ph_grid) * np.sin(th_grid)
        zs = sun_pos[2] + sun_r * np.cos(ph_grid)
        traces.append(
            go.Surface(
                x=xs, y=ys, z=zs,
                colorscale=[[0, "gold"], [1, "yellow"]],
                opacity=1.0, showscale=False,
                name="Sun (not to scale)", hoverinfo="name",
            )
        )

    fig = go.Figure(data=traces)
    axis_half = sun_distance * 1.15 if show_sun else orbit_radius * 1.3
    fig.update_layout(
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title="ECI X (km)", range=[-axis_half, axis_half]),
            yaxis=dict(title="ECI Y (km)", range=[-axis_half, axis_half]),
            zaxis=dict(title="ECI Z (km)", range=[-axis_half, axis_half]),
        ),
        height=650,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')

    # Numeric values table (unit vectors in ECI).
    rows: list[dict[str, object]] = []

    def _row(label: str, vec: np.ndarray, body_frame: str = "") -> None:
        rows.append(
            {
                "Vector": label,
                "ECI X": round(float(vec[0]), 4),
                "ECI Y": round(float(vec[1]), 4),
                "ECI Z": round(float(vec[2]), 4),
                "Body-frame expression": body_frame,
            }
        )

    _row("+X body", body_x, "+X")
    _row("+Y body", body_y, "+Y")
    _row("+Z body", body_z, "+Z")
    _row("Sun", u_sun, "(depends on attitude)")
    _row("Nadir", nadir, "(depends on attitude)")
    _row("Velocity", v_hat, "(depends on attitude)")
    _row("Orbit normal", h_hat, "")
    for panel in cfg.solar_array.panels:
        if panel.mounting is PanelMounting.TWO_AXIS:
            _row(f"{panel.name} normal", u_sun, "2-axis tracking -> Sun")
        else:
            normal_body = np.asarray(panel.normal_body, dtype=float)
            normal_body = normal_body / np.linalg.norm(normal_body)
            normal_eci = rot @ normal_body
            face_label = panel.normal_face if panel.normal_face != "custom" else (
                f"({normal_body[0]:+.2f}, {normal_body[1]:+.2f}, {normal_body[2]:+.2f})"
            )
            _row(f"{panel.name} normal", normal_eci, face_label)

    st.subheader("Numeric values (unit vectors in ECI)")
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
