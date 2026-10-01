"""Umbra / penumbra illumination fraction from the satellite's point of view.

The conical model computes the apparent angular radii of the Sun and Earth and
their angular separation; the fraction of the Sun's disk visible from the
satellite is used as the illumination factor (0 in umbra, 1 in full sunlight).

A cylindrical fallback is also provided for comparison and tests.
"""

from __future__ import annotations

import numpy as np

R_EARTH_KM = 6378.137
R_SUN_KM = 695_700.0


def _lens_area_fraction(d: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Fraction of disk 1 (apparent Sun) hidden by disk 2 (apparent Earth)."""
    full = r1 + r2
    none = np.abs(r1 - r2)

    # Full occultation when d <= |r_sun - r_earth| AND earth bigger than sun.
    fully_blocked = (d <= none) & (r2 >= r1)
    # Earth smaller than sun, fully inside: ring eclipse, blocked = (r2/r1)^2
    annular = (d <= none) & (r2 < r1)
    partial = (d < full) & (d > none)

    # Compute lens overlap area.
    with np.errstate(divide="ignore", invalid="ignore"):
        a1 = np.where(
            partial,
            r1 * r1 * np.arccos(np.clip((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1), -1.0, 1.0)),
            0.0,
        )
        a2 = np.where(
            partial,
            r2 * r2 * np.arccos(np.clip((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2), -1.0, 1.0)),
            0.0,
        )
        s = 0.5 * (d + r1 + r2)
        tri = np.where(
            partial,
            np.sqrt(np.maximum(s * (s - d) * (s - r1) * (s - r2), 0.0)),
            0.0,
        )
    lens_area = a1 + a2 - 2.0 * tri
    sun_area = np.pi * r1 * r1

    blocked_frac = np.zeros_like(d)
    blocked_frac = np.where(partial, lens_area / sun_area, blocked_frac)
    blocked_frac = np.where(annular, (r2 * r2) / (r1 * r1), blocked_frac)
    blocked_frac = np.where(fully_blocked, 1.0, blocked_frac)
    return np.clip(blocked_frac, 0.0, 1.0)


def illumination_fraction_conical(
    r_sat_eci_km: np.ndarray, r_sun_eci_km: np.ndarray
) -> np.ndarray:
    """Return the illumination fraction (0..1) using a conical umbra/penumbra model.

    Shapes: either both inputs are (3,) or both are (N, 3).
    """
    r_sat = np.atleast_2d(r_sat_eci_km).astype(float)
    r_sun = np.atleast_2d(r_sun_eci_km).astype(float)

    sat_to_sun = r_sun - r_sat
    d_sun = np.linalg.norm(sat_to_sun, axis=-1)
    d_sat = np.linalg.norm(r_sat, axis=-1)

    # Apparent angular radii (rad).
    rho_sun = np.arcsin(np.clip(R_SUN_KM / d_sun, -1.0, 1.0))
    rho_earth = np.arcsin(np.clip(R_EARTH_KM / d_sat, -1.0, 1.0))

    # Angular separation between the Sun and Earth centers as seen from the sat.
    u_sun = sat_to_sun / d_sun[:, None]
    u_earth = -r_sat / d_sat[:, None]
    cos_theta = np.sum(u_sun * u_earth, axis=-1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    blocked = _lens_area_fraction(theta, rho_sun, rho_earth)
    illum = 1.0 - blocked

    # If Earth is behind the Sun (|theta| > pi/2 + rho_earth), ignore occultation.
    illum = np.where(theta > (np.pi / 2.0 + rho_earth), 1.0, illum)
    return np.squeeze(illum)


def illumination_fraction_cylindrical(
    r_sat_eci_km: np.ndarray, r_sun_eci_km: np.ndarray
) -> np.ndarray:
    """Simple cylindrical shadow (returns 0 in umbra, 1 outside)."""
    r_sat = np.atleast_2d(r_sat_eci_km).astype(float)
    r_sun = np.atleast_2d(r_sun_eci_km).astype(float)

    u_sun = r_sun / np.linalg.norm(r_sun, axis=-1, keepdims=True)
    # Component of sat position along sun direction.
    s_along = np.sum(r_sat * u_sun, axis=-1)
    # Perpendicular component magnitude.
    perp = r_sat - s_along[:, None] * u_sun
    r_perp = np.linalg.norm(perp, axis=-1)
    in_shadow = (s_along < 0.0) & (r_perp < R_EARTH_KM)
    illum = np.where(in_shadow, 0.0, 1.0)
    return np.squeeze(illum)


def analytic_eclipse_fraction(altitude_km: float, beta_deg: float) -> float:
    """Analytic circular-orbit eclipse fraction (0..1) for a given beta angle.

    The formula is: f_E = (1/pi) * arccos(sqrt(h^2 + 2 R h) / (r cos beta))
    valid when |beta| < beta* = arcsin(R / r), else no eclipse (f_E = 0).
    """
    r = R_EARTH_KM + altitude_km
    beta = np.deg2rad(beta_deg)
    beta_star = np.arcsin(R_EARTH_KM / r)
    if np.abs(beta) >= beta_star:
        return 0.0
    numerator = np.sqrt(altitude_km**2 + 2.0 * R_EARTH_KM * altitude_km)
    denominator = r * np.cos(beta)
    if denominator <= 0.0:
        return 0.0
    arg = np.clip(numerator / denominator, -1.0, 1.0)
    return float(np.arccos(arg) / np.pi)
