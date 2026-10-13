"""Solar array power model.

For each panel, instantaneous power is::

    P = flux * area * eff * packing * degradation * (1 - temp_loss) *
        max(cos(incidence_angle), 0) * illumination

The incidence angle is between the Sun direction and the panel normal, both in
the ECI frame. Fixed panels use the body-frame normal rotated by the attitude
matrix; 1-axis-tracking panels pick the rotation about their body-frame axis
that maximises the cosine of incidence; 2-axis tracking panels always face the
Sun directly (cos(incidence) = 1), as long as the satellite is in sunlight.

A per-panel degradation of ``(1 - degradation_per_year)**mission_year`` is
applied for end-of-life calculations.
"""

from __future__ import annotations

import numpy as np

from .config import PanelConfig, PanelMounting, SolarArrayConfig


def _panel_effective_flux_factor(panel: PanelConfig) -> float:
    """Fixed per-panel scalar: eff * packing * inherent * degradation * (1 - temp)."""
    degrade = (1.0 - panel.degradation_per_year) ** panel.mission_year
    return (
        panel.cell_efficiency
        * panel.packing_factor
        * panel.inherent_degradation
        * degrade
        * (1.0 - panel.temperature_loss)
    )


def _in_plane_axis_body(normal: np.ndarray) -> np.ndarray:
    """Deterministic unit vector perpendicular to ``normal`` in body coords."""
    n = normal / np.linalg.norm(normal)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    v = ref - np.dot(ref, n) * n
    return v / np.linalg.norm(v)


def _in_plane_axis_batch(normals: np.ndarray) -> np.ndarray:
    """Return (N, 3) unit vectors perpendicular to each row of ``normals``."""
    n = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    n_pts = n.shape[0]
    ref_z = np.tile(np.array([0.0, 0.0, 1.0]), (n_pts, 1))
    ref_x = np.tile(np.array([1.0, 0.0, 0.0]), (n_pts, 1))
    use_x = np.abs(n[:, 2:3]) > 0.9
    ref = np.where(use_x, ref_x, ref_z)
    proj = ref - np.sum(ref * n, axis=-1, keepdims=True) * n
    norms = np.linalg.norm(proj, axis=-1, keepdims=True)
    return proj / np.where(norms < 1e-12, 1.0, norms)


def _limited_one_axis_angles(
    ideal_wrapped_deg: np.ndarray,
    times_s: np.ndarray,
    minimum_deg: float,
    maximum_deg: float,
    slew_rate_deg_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Follow an ideal 1-axis command within cable-twist limits.

    The ideal command is first unwrapped into a continuous trajectory. Its
    current 360-degree-equivalent branch is retained until that branch leaves
    the legal travel interval. The controller then selects an equivalent
    branch inside the interval and rate-limits the actual angle while the
    gimbal unwinds toward it.
    """
    ideal_wrapped_deg = np.asarray(ideal_wrapped_deg, dtype=float)
    times_s = np.asarray(times_s, dtype=float)
    if ideal_wrapped_deg.shape != times_s.shape:
        raise ValueError("times_s must have one value per panel-geometry sample")
    if np.any(np.diff(times_s) <= 0.0):
        raise ValueError("times_s must be strictly increasing")

    ideal_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(ideal_wrapped_deg)))
    target = np.empty_like(ideal_unwrapped)
    actual = np.empty_like(ideal_unwrapped)
    rewinding = np.zeros(ideal_unwrapped.shape, dtype=bool)

    def _legal_branch(reference_deg: float, ideal_deg: float) -> int:
        k_min = int(np.ceil((minimum_deg - ideal_deg) / 360.0 - 1e-12))
        k_max = int(np.floor((maximum_deg - ideal_deg) / 360.0 + 1e-12))
        if k_min > k_max:  # Defensive: validation requires a >=360 deg interval.
            raise ValueError("no equivalent Sun-tracking angle lies within gimbal limits")
        return min(range(k_min, k_max + 1), key=lambda k: abs(ideal_deg + 360.0 * k - reference_deg))

    branch = _legal_branch(0.0, float(ideal_unwrapped[0]))
    target[0] = ideal_unwrapped[0] + 360.0 * branch
    actual[0] = 0.0
    rewind_active = False

    for i in range(1, len(ideal_unwrapped)):
        ideal = float(ideal_unwrapped[i])
        branch_target = ideal + 360.0 * branch
        if branch_target < minimum_deg - 1e-9 or branch_target > maximum_deg + 1e-9:
            branch = _legal_branch(float(actual[i - 1]), ideal)
            branch_target = ideal + 360.0 * branch
            rewind_active = True
        target[i] = branch_target

        max_step = slew_rate_deg_s * float(times_s[i] - times_s[i - 1])
        error = branch_target - float(actual[i - 1])
        actual[i] = np.clip(
            actual[i - 1] + np.clip(error, -max_step, max_step),
            minimum_deg,
            maximum_deg,
        )
        rewinding[i] = rewind_active
        if abs(branch_target - actual[i]) <= 1e-9:
            rewind_active = False

    return actual, target, rewinding


def panel_frames_eci(
    panel: PanelConfig,
    body_to_eci: np.ndarray,  # (N, 3, 3)
    u_sun_eci: np.ndarray,  # (N, 3)
    times_s: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return per-time-step panel pointing geometry in the ECI frame.

    Keys:

    - ``normal_eci`` (N, 3): the actual pointing normal of the panel.
        - FIXED: ``R @ n_body``.
        - ONE_AXIS: the Sun is projected onto the plane perpendicular to the
          rotation axis (in body coords); that direction, normalized, is the
          tracked normal; fallback to the rest normal when the Sun is parallel
          to the axis. This is the same vector the power model implicitly
          picks, so the drawing and the computed ``cos(incidence)`` cannot
          disagree.
        - TWO_AXIS: ``u_sun_eci`` directly.
    - ``edge_eci`` (N, 3): a long in-plane axis for drawing a plate.
    - ``angle_deg`` (N,): gimbal/slew angle. For 1-axis it is the signed angle
      about ``rotation_axis_body`` from the rest normal to the actual normal.
      For 2-axis it is the total slew angle between the rest normal and the
      Sun. For fixed panels it is zero.
    - ``target_angle_deg`` (N,): ideal legal gimbal command. This differs from
      ``angle_deg`` while a twist-limited gimbal is slewing.
    - ``rewinding`` (N,): true while a twist-limited gimbal is unwinding after
      switching to a 360-degree-equivalent tracking branch.
    - ``cos_incidence`` (N,): ``clip(normal_eci . u_sun_eci, 0, 1)``.
    """
    body_to_eci = np.asarray(body_to_eci, dtype=float)
    u_sun_eci = np.atleast_2d(np.asarray(u_sun_eci, dtype=float))
    n_pts = body_to_eci.shape[0]

    normal_body = np.array(panel.normal_body, dtype=float)
    normal_body = normal_body / np.linalg.norm(normal_body)

    axis_body = np.array(panel.rotation_axis_body, dtype=float)
    axis_norm = np.linalg.norm(axis_body)
    axis_body = axis_body / axis_norm if axis_norm > 0.0 else np.array([0.0, 1.0, 0.0])

    # Sun direction in the body frame: u_sun_body = R^T @ u_sun_eci.
    u_sun_body = np.einsum("nji,nj->ni", body_to_eci, u_sun_eci)

    if panel.mounting is PanelMounting.FIXED:
        n_body = np.tile(normal_body, (n_pts, 1))
        normal_eci = np.einsum("nij,nj->ni", body_to_eci, n_body)
        edge_body = np.tile(_in_plane_axis_body(normal_body), (n_pts, 1))
        edge_eci = np.einsum("nij,nj->ni", body_to_eci, edge_body)
        angle_deg = np.zeros(n_pts)
        target_angle_deg = angle_deg.copy()
        rewinding = np.zeros(n_pts, dtype=bool)

    elif panel.mounting is PanelMounting.ONE_AXIS:
        s_dot_a = u_sun_body @ axis_body
        s_perp = u_sun_body - np.outer(s_dot_a, axis_body)
        s_perp_norm = np.linalg.norm(s_perp, axis=-1, keepdims=True)
        rest_tile = np.tile(normal_body, (n_pts, 1))
        safe_norm = np.where(s_perp_norm < 1e-9, 1.0, s_perp_norm)
        n_best_body = np.where(s_perp_norm < 1e-9, rest_tile, s_perp / safe_norm)
        normal_eci = np.einsum("nij,nj->ni", body_to_eci, n_best_body)
        axis_tile = np.tile(axis_body, (n_pts, 1))
        edge_eci = np.einsum("nij,nj->ni", body_to_eci, axis_tile)
        # Signed angle about axis_body from rest normal to tracked normal.
        cross_n = np.cross(rest_tile, n_best_body)
        sin_part = cross_n @ axis_body
        cos_part = np.sum(rest_tile * n_best_body, axis=-1)
        ideal_angle_deg = np.rad2deg(np.arctan2(sin_part, cos_part))

        if panel.gimbal_min_deg is not None and panel.gimbal_max_deg is not None:
            if times_s is None:
                raise ValueError("times_s is required for a twist-limited 1-axis panel")
            angle_deg, target_angle_deg, rewinding = _limited_one_axis_angles(
                ideal_angle_deg,
                np.asarray(times_s, dtype=float),
                panel.gimbal_min_deg,
                panel.gimbal_max_deg,
                panel.max_slew_rate_deg_s,
            )
            angle_rad = np.deg2rad(angle_deg)
            n_parallel = np.outer(np.full(n_pts, np.dot(normal_body, axis_body)), axis_body)
            n_perp = rest_tile - n_parallel
            axis_cross_normal = np.tile(np.cross(axis_body, normal_body), (n_pts, 1))
            actual_body = (
                n_parallel
                + n_perp * np.cos(angle_rad)[:, None]
                + axis_cross_normal * np.sin(angle_rad)[:, None]
            )
            normal_eci = np.einsum("nij,nj->ni", body_to_eci, actual_body)
        else:
            angle_deg = ideal_angle_deg
            target_angle_deg = angle_deg.copy()
            rewinding = np.zeros(n_pts, dtype=bool)

    else:  # TWO_AXIS
        normal_eci = u_sun_eci.copy()
        # Edge: project boom axis (rotation_axis_body) perpendicular to normal.
        axis_eci = np.einsum("nij,nj->ni", body_to_eci, np.tile(axis_body, (n_pts, 1)))
        axis_dot_n = np.sum(axis_eci * normal_eci, axis=-1, keepdims=True)
        edge_raw = axis_eci - axis_dot_n * normal_eci
        edge_norm = np.linalg.norm(edge_raw, axis=-1, keepdims=True)
        fallback = _in_plane_axis_batch(normal_eci)
        safe_norm = np.where(edge_norm < 1e-9, 1.0, edge_norm)
        edge_eci = np.where(edge_norm < 1e-9, fallback, edge_raw / safe_norm)
        # 2-axis slew angle: angle between rest normal and the Sun, both in body.
        cos_slew = np.clip(u_sun_body @ normal_body, -1.0, 1.0)
        angle_deg = np.rad2deg(np.arccos(cos_slew))
        target_angle_deg = angle_deg.copy()
        rewinding = np.zeros(n_pts, dtype=bool)

    cos_inc = np.clip(np.sum(normal_eci * u_sun_eci, axis=-1), 0.0, 1.0)
    return {
        "normal_eci": normal_eci,
        "edge_eci": edge_eci,
        "angle_deg": angle_deg,
        "target_angle_deg": target_angle_deg,
        "rewinding": rewinding,
        "cos_incidence": cos_inc,
    }


def _cos_incidence_for_panel(
    panel: PanelConfig,
    body_to_eci: np.ndarray,  # (N, 3, 3)
    u_sun_eci: np.ndarray,  # (N, 3)
) -> np.ndarray:
    """Return cos(incidence) per time step for a single panel, in 0..1."""
    return panel_frames_eci(panel, body_to_eci, u_sun_eci)["cos_incidence"]


def array_power_watts(
    array: SolarArrayConfig,
    body_to_eci: np.ndarray,
    u_sun_eci: np.ndarray,
    distance_au: np.ndarray,
    illumination: np.ndarray,
    times_s: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return per-panel and total generated power (W) at each time step.

    Keys:
    - ``total``: shape (N,) in W
    - ``per_panel``: shape (num_panels, N) in W
    - ``panel_names``: list[str]
    """
    flux = array.solar_constant_w_m2 / (distance_au**2)  # 1 AU scaling
    n_pts = flux.shape[0]
    per_panel = np.zeros((len(array.panels), n_pts))
    normal_eci = np.zeros((len(array.panels), n_pts, 3))
    edge_eci = np.zeros((len(array.panels), n_pts, 3))
    cos_incidence = np.zeros((len(array.panels), n_pts))
    angle_deg = np.zeros((len(array.panels), n_pts))
    target_angle_deg = np.zeros((len(array.panels), n_pts))
    rewinding = np.zeros((len(array.panels), n_pts), dtype=bool)
    for idx, panel in enumerate(array.panels):
        frames = panel_frames_eci(panel, body_to_eci, u_sun_eci, times_s)
        cos_inc = frames["cos_incidence"]
        normal_eci[idx] = frames["normal_eci"]
        edge_eci[idx] = frames["edge_eci"]
        cos_incidence[idx] = cos_inc
        angle_deg[idx] = frames["angle_deg"]
        target_angle_deg[idx] = frames["target_angle_deg"]
        rewinding[idx] = frames["rewinding"]
        per_panel[idx] = (
            flux
            * panel.area_m2
            * _panel_effective_flux_factor(panel)
            * cos_inc
            * illumination
        )
    total = np.sum(per_panel, axis=0)
    names = [p.name for p in array.panels]
    return {
        "total": total,
        "per_panel": per_panel,
        "panel_names": names,
        "normal_eci": normal_eci,
        "edge_eci": edge_eci,
        "cos_incidence": cos_incidence,
        "angle_deg": angle_deg,
        "target_angle_deg": target_angle_deg,
        "rewinding": rewinding,
    }
