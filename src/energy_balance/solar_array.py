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


def _cos_incidence_for_panel(
    panel: PanelConfig,
    body_to_eci: np.ndarray,  # (N, 3, 3)
    u_sun_eci: np.ndarray,  # (N, 3)
) -> np.ndarray:
    """Return cos(incidence) per time step for a single panel, in 0..1."""
    normal_body = np.array(panel.normal_body, dtype=float)
    normal_body = normal_body / np.linalg.norm(normal_body)
    axis_body = np.array(panel.rotation_axis_body, dtype=float)
    nrm = np.linalg.norm(axis_body)
    axis_body = axis_body / nrm if nrm > 0.0 else np.array([0.0, 1.0, 0.0])

    if panel.mounting is PanelMounting.TWO_AXIS:
        # Panel can always point directly at the Sun.
        return np.ones(u_sun_eci.shape[0])

    # Sun direction expressed in the body frame: u_sun_body = R^T @ u_sun_eci.
    u_sun_body = np.einsum("nji,nj->ni", body_to_eci, u_sun_eci)

    if panel.mounting is PanelMounting.FIXED:
        cos_inc = u_sun_body @ normal_body
        return np.clip(cos_inc, 0.0, 1.0)

    # 1-axis tracking: panel normal lies in the plane perpendicular to axis_body
    # and starts at normal_body. The best cos(incidence) is the magnitude of the
    # Sun's projection onto that plane.
    s_dot_a = u_sun_body @ axis_body
    s_perp = u_sun_body - np.outer(s_dot_a, axis_body)
    cos_inc = np.linalg.norm(s_perp, axis=-1)
    return np.clip(cos_inc, 0.0, 1.0)


def array_power_watts(
    array: SolarArrayConfig,
    body_to_eci: np.ndarray,
    u_sun_eci: np.ndarray,
    distance_au: np.ndarray,
    illumination: np.ndarray,
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
    for idx, panel in enumerate(array.panels):
        cos_inc = _cos_incidence_for_panel(panel, body_to_eci, u_sun_eci)
        per_panel[idx] = (
            flux
            * panel.area_m2
            * _panel_effective_flux_factor(panel)
            * cos_inc
            * illumination
        )
    total = np.sum(per_panel, axis=0)
    names = [p.name for p in array.panels]
    return {"total": total, "per_panel": per_panel, "panel_names": names}
