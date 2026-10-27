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


def _rewind_branch_angles(
    ideal_unwrapped: np.ndarray,
    times_s: np.ndarray,
    minimum_deg: float,
    maximum_deg: float,
    slew_rate_deg_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Follow an ideal unwrapped command within a >=360-deg cable-wrap interval.

    The current 360-degree-equivalent branch is retained until that branch
    leaves the legal travel interval. The controller then selects an
    equivalent branch inside the interval and rate-limits the actual angle
    while the gimbal unwinds toward it.

    The gimbal starts already on its Sun-tracking command rather than at the
    0-degree rest position, so epoch is a spacecraft that has been tracking
    all along rather than one that has just deployed. Starting at rest would
    open the run with a slew of up to a full revolution that is an artifact of
    the initial condition, not of the orbit. The starting branch is still the
    legal one nearest rest, so the drive begins unwound and the first rewind
    is driven by the orbit.
    """
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
    # Already on command at epoch; _legal_branch keeps this inside the limits.
    actual[0] = target[0]
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


def _hard_stop_angles(
    ideal_unwrapped: np.ndarray,
    times_s: np.ndarray,
    minimum_deg: float,
    maximum_deg: float,
    slew_rate_deg_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Follow an ideal unwrapped command within a hard mechanical stop.

    Unlike the cable-wrap case there is no equivalent branch to rewind to:
    the command is simply clamped into ``[minimum_deg, maximum_deg]`` and the
    gimbal rate-limits its way toward that clamped target. The Sun can go out
    of reach, in which case the panel stays pinned at the nearer limit and
    incidence degrades.

    As in the cable-wrap case the gimbal starts already on its (clamped)
    command rather than at the 0-degree rest position, so the run does not
    open with a slew that is an artifact of the initial condition.
    """
    target = np.clip(ideal_unwrapped, minimum_deg, maximum_deg)
    actual = np.empty_like(target)
    # Already on command at epoch; target is clipped, so this is within limits.
    actual[0] = target[0]
    for i in range(1, len(target)):
        max_step = slew_rate_deg_s * float(times_s[i] - times_s[i - 1])
        error = target[i] - float(actual[i - 1])
        actual[i] = np.clip(
            actual[i - 1] + np.clip(error, -max_step, max_step),
            minimum_deg,
            maximum_deg,
        )
    rewinding = np.zeros(target.shape, dtype=bool)
    return actual, target, rewinding


def _limited_axis_angles(
    ideal_wrapped_deg: np.ndarray,
    times_s: np.ndarray,
    minimum_deg: float,
    maximum_deg: float,
    slew_rate_deg_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Follow an ideal single-axis command within gimbal limits.

    The ideal command is first unwrapped into a continuous trajectory. If the
    interval spans >=360 deg the gimbal can always rewind to a
    360-degree-equivalent angle and keep tracking the Sun exactly (cable-wrap
    mode, see :func:`_rewind_branch_angles`). Otherwise it is a hard
    mechanical stop (see :func:`_hard_stop_angles`): no rewind, the command is
    directly clamped, and incidence degrades once the Sun goes out of reach.
    """
    ideal_wrapped_deg = np.asarray(ideal_wrapped_deg, dtype=float)
    times_s = np.asarray(times_s, dtype=float)
    if ideal_wrapped_deg.shape != times_s.shape:
        raise ValueError("times_s must have one value per panel-geometry sample")
    if np.any(np.diff(times_s) <= 0.0):
        raise ValueError("times_s must be strictly increasing")

    ideal_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(ideal_wrapped_deg)))

    if maximum_deg - minimum_deg >= 360.0:
        return _rewind_branch_angles(ideal_unwrapped, times_s, minimum_deg, maximum_deg, slew_rate_deg_s)
    return _hard_stop_angles(ideal_unwrapped, times_s, minimum_deg, maximum_deg, slew_rate_deg_s)


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
        - TWO_AXIS: ``u_sun_eci`` directly when unlimited; when a gimbal limit
          is configured, the two physical gimbal angles are solved for and
          clamped, and the resulting (possibly off-Sun) normal is used
          instead.
    - ``edge_eci`` (N, 3): a long in-plane axis for drawing a plate.
    - ``angle_deg`` (N,): gimbal/slew angle. For 1-axis it is the signed angle
      about ``rotation_axis_body`` from the rest normal to the actual normal.
      For 2-axis it is the total slew angle between the rest normal and the
      *actual* normal (this can be less than the Sun offset while a limited
      gimbal is clamped). For fixed panels it is zero.
    - ``target_angle_deg`` (N,): ideal legal gimbal command (1-axis), or the
      total slew angle to the Sun (2-axis). This differs from ``angle_deg``
      while a limited gimbal is slewing or clamped at a stop.
    - ``angle1_deg`` / ``angle2_deg`` (N,): the actual primary/secondary-axis
      gimbal angles. For fixed panels both are zero. For 1-axis panels
      ``angle1_deg`` equals ``angle_deg`` and ``angle2_deg`` is zero (no
      secondary axis exists). For 2-axis panels these are the two physical
      gimbal angles computed from the ``(normal_body, a2, rotation_axis_body)``
      triad.
    - ``target_angle1_deg`` / ``target_angle2_deg`` (N,): the ideal
      (unconstrained) primary/secondary-axis commands, mirroring
      ``angle1_deg``/``angle2_deg``.
    - ``rewinding`` (N,): true while a gimbal is unwinding after switching to
      a 360-degree-equivalent cable-wrap tracking branch. Always false for
      hard-stop limits (which never rewind).
    - ``cos_incidence`` (N,): ``clip(normal_eci . u_sun_eci, 0, 1)``.

    A gimbal limit (``gimbal_min_deg``/``gimbal_max_deg`` for the primary
    axis, ``gimbal2_min_deg``/``gimbal2_max_deg`` for the 2-axis secondary
    axis) spanning >=360 deg is a cable-wrap limit: the gimbal always finds a
    360-degree-equivalent angle and keeps tracking the Sun exactly. A span
    <360 deg is a hard mechanical stop: the Sun can go out of reach and
    incidence degrades smoothly, with no rewind.
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
        angle1_deg = angle_deg.copy()
        angle2_deg = angle_deg.copy()
        target_angle1_deg = angle_deg.copy()
        target_angle2_deg = angle_deg.copy()
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
                raise ValueError("times_s is required for a gimbal-limited 1-axis panel")
            angle_deg, target_angle_deg, rewinding = _limited_axis_angles(
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

        angle1_deg = angle_deg
        angle2_deg = np.zeros(n_pts)
        target_angle1_deg = target_angle_deg
        target_angle2_deg = np.zeros(n_pts)

    else:  # TWO_AXIS
        # Secondary gimbal axis: perpendicular to both the primary axis and
        # the rest normal (which is itself perpendicular to the primary axis
        # by convention), so (normal_body, a2_body, axis_body) is an
        # orthonormal right-handed triad.
        a2_body = np.cross(axis_body, normal_body)
        a2_body = a2_body / np.linalg.norm(a2_body)

        p = u_sun_body @ normal_body
        q = u_sun_body @ a2_body
        r = np.clip(u_sun_body @ axis_body, -1.0, 1.0)
        angle1_ideal_deg = np.rad2deg(np.arctan2(q, p))
        angle2_ideal_deg = np.rad2deg(-np.arcsin(r))

        has_limit1 = panel.gimbal_min_deg is not None and panel.gimbal_max_deg is not None
        has_limit2 = panel.gimbal2_min_deg is not None and panel.gimbal2_max_deg is not None

        if not has_limit1 and not has_limit2:
            # Fast exact path: preserves today's bit-for-bit behavior (always
            # points exactly at the Sun) when no limit is configured.
            normal_eci = u_sun_eci.copy()
            angle1_deg = angle1_ideal_deg
            angle2_deg = angle2_ideal_deg
            target_angle1_deg = angle1_ideal_deg.copy()
            target_angle2_deg = angle2_ideal_deg.copy()
            rewinding = np.zeros(n_pts, dtype=bool)
        else:
            if times_s is None:
                raise ValueError("times_s is required for a gimbal-limited 2-axis panel")
            times_s = np.asarray(times_s, dtype=float)

            if has_limit1:
                angle1_deg, target_angle1_deg, rewinding = _limited_axis_angles(
                    angle1_ideal_deg,
                    times_s,
                    panel.gimbal_min_deg,
                    panel.gimbal_max_deg,
                    panel.max_slew_rate_deg_s,
                )
            else:
                angle1_deg = angle1_ideal_deg
                target_angle1_deg = angle1_ideal_deg.copy()
                rewinding = np.zeros(n_pts, dtype=bool)

            if has_limit2:
                # Axis 2 is always a hard stop (its range is physically
                # bounded to +/-90 deg), never a cable-wrap rewind.
                angle2_deg, target_angle2_deg, _ = _limited_axis_angles(
                    angle2_ideal_deg,
                    times_s,
                    panel.gimbal2_min_deg,
                    panel.gimbal2_max_deg,
                    panel.max_slew_rate_deg_s,
                )
            else:
                angle2_deg = angle2_ideal_deg
                target_angle2_deg = angle2_ideal_deg.copy()

            # Each axis is clamped independently (not a joint-workspace
            # solve); recombine via the same sequential-rotation kinematics
            # used to derive the ideal angles.
            angle1_rad = np.deg2rad(angle1_deg)
            angle2_rad = np.deg2rad(angle2_deg)
            n0_tile = np.tile(normal_body, (n_pts, 1))
            a2_tile = np.tile(a2_body, (n_pts, 1))
            a1_tile = np.tile(axis_body, (n_pts, 1))
            actual_body = (
                (np.cos(angle2_rad) * np.cos(angle1_rad))[:, None] * n0_tile
                + (np.cos(angle2_rad) * np.sin(angle1_rad))[:, None] * a2_tile
                - np.sin(angle2_rad)[:, None] * a1_tile
            )
            normal_eci = np.einsum("nij,nj->ni", body_to_eci, actual_body)

        # Edge: project boom axis (rotation_axis_body) perpendicular to normal.
        axis_eci = np.einsum("nij,nj->ni", body_to_eci, np.tile(axis_body, (n_pts, 1)))
        axis_dot_n = np.sum(axis_eci * normal_eci, axis=-1, keepdims=True)
        edge_raw = axis_eci - axis_dot_n * normal_eci
        edge_norm = np.linalg.norm(edge_raw, axis=-1, keepdims=True)
        fallback = _in_plane_axis_batch(normal_eci)
        safe_norm = np.where(edge_norm < 1e-9, 1.0, edge_norm)
        edge_eci = np.where(edge_norm < 1e-9, fallback, edge_raw / safe_norm)

        # Total slew angle: rest normal to the actual (possibly clamped)
        # normal, vs. rest normal to the Sun (the ideal/target slew).
        actual_normal_body = np.einsum("nji,nj->ni", body_to_eci, normal_eci)
        cos_slew_actual = np.clip(np.sum(actual_normal_body * normal_body, axis=-1), -1.0, 1.0)
        angle_deg = np.rad2deg(np.arccos(cos_slew_actual))
        cos_slew_ideal = np.clip(u_sun_body @ normal_body, -1.0, 1.0)
        target_angle_deg = np.rad2deg(np.arccos(cos_slew_ideal))

    cos_inc = np.clip(np.sum(normal_eci * u_sun_eci, axis=-1), 0.0, 1.0)
    return {
        "normal_eci": normal_eci,
        "edge_eci": edge_eci,
        "angle_deg": angle_deg,
        "target_angle_deg": target_angle_deg,
        "angle1_deg": angle1_deg,
        "angle2_deg": angle2_deg,
        "target_angle1_deg": target_angle1_deg,
        "target_angle2_deg": target_angle2_deg,
        "rewinding": rewinding,
        "cos_incidence": cos_inc,
    }


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
    angle1_deg = np.zeros((len(array.panels), n_pts))
    angle2_deg = np.zeros((len(array.panels), n_pts))
    target_angle1_deg = np.zeros((len(array.panels), n_pts))
    target_angle2_deg = np.zeros((len(array.panels), n_pts))
    rewinding = np.zeros((len(array.panels), n_pts), dtype=bool)
    for idx, panel in enumerate(array.panels):
        frames = panel_frames_eci(panel, body_to_eci, u_sun_eci, times_s)
        cos_inc = frames["cos_incidence"]
        normal_eci[idx] = frames["normal_eci"]
        edge_eci[idx] = frames["edge_eci"]
        cos_incidence[idx] = cos_inc
        angle_deg[idx] = frames["angle_deg"]
        target_angle_deg[idx] = frames["target_angle_deg"]
        angle1_deg[idx] = frames["angle1_deg"]
        angle2_deg[idx] = frames["angle2_deg"]
        target_angle1_deg[idx] = frames["target_angle1_deg"]
        target_angle2_deg[idx] = frames["target_angle2_deg"]
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
        "angle1_deg": angle1_deg,
        "angle2_deg": angle2_deg,
        "target_angle1_deg": target_angle1_deg,
        "target_angle2_deg": target_angle2_deg,
        "rewinding": rewinding,
    }
