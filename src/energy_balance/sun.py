"""Low-precision solar ephemeris in the Earth-centered inertial (ECI) frame.

The algorithm follows the Astronomical Almanac / Vallado "low-precision" formulas,
which are accurate to roughly 0.01 degree on the Sun's position for dates within
a few decades of J2000. That is far more than enough for an eclipse and energy
balance analysis.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from .orbit import julian_date

AU_KM = 149_597_870.7


def _sun_vector_from_jd(jd_ut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sun position in ECI (km) and distance (AU) for scalar or vector JD."""
    t_ut = (jd_ut - 2_451_545.0) / 36_525.0

    lambda_m_sun_deg = (280.460 + 36_000.771 * t_ut) % 360.0
    m_sun_deg = (357.5277233 + 35_999.05034 * t_ut) % 360.0
    m_sun = np.deg2rad(m_sun_deg)

    lambda_ecl_deg = (
        lambda_m_sun_deg
        + 1.914666471 * np.sin(m_sun)
        + 0.019994643 * np.sin(2.0 * m_sun)
    )
    lambda_ecl = np.deg2rad(lambda_ecl_deg)

    r_au = (
        1.000140612
        - 0.016708617 * np.cos(m_sun)
        - 0.000139589 * np.cos(2.0 * m_sun)
    )
    eps = np.deg2rad(23.439291 - 0.0130042 * t_ut)

    x = r_au * np.cos(lambda_ecl)
    y = r_au * np.cos(eps) * np.sin(lambda_ecl)
    z = r_au * np.sin(eps) * np.sin(lambda_ecl)
    r_eci_km = np.stack([x, y, z], axis=-1) * AU_KM
    return r_eci_km, r_au


def sun_vector_eci(epoch: datetime, times_s: np.ndarray) -> dict[str, np.ndarray]:
    """Return Sun ECI positions along a time grid.

    - ``r_sun_eci_km``: shape (N, 3), Sun position in km in ECI.
    - ``u_sun_eci``: shape (N, 3), unit vector from Earth to Sun.
    - ``distance_au``: shape (N,), Earth-Sun distance in AU.
    """
    jd0 = julian_date(epoch)
    jd = jd0 + np.asarray(times_s, dtype=float) / 86_400.0
    r_eci_km, r_au = _sun_vector_from_jd(jd)
    u = r_eci_km / np.linalg.norm(r_eci_km, axis=-1, keepdims=True)
    return {"r_sun_eci_km": r_eci_km, "u_sun_eci": u, "distance_au": r_au}


def sun_vector_at(epoch: datetime) -> np.ndarray:
    """Convenience: Sun ECI position (km) at a single datetime."""
    jd = julian_date(epoch)
    r, _ = _sun_vector_from_jd(np.array([jd]))
    return r[0]


def beta_angle_deg(h_eci: np.ndarray, u_sun: np.ndarray) -> np.ndarray:
    """Angle between the Sun vector and the orbital plane (deg).

    ``h_eci`` is the orbital angular momentum vector (direction only matters).
    """
    h_hat = h_eci / np.linalg.norm(h_eci, axis=-1, keepdims=True)
    cos_angle = np.sum(h_hat * u_sun, axis=-1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # beta = 90 deg - angle(h, u_sun)
    return 90.0 - np.rad2deg(np.arccos(cos_angle))


def seasonal_sweep(
    epoch: datetime, step_days: float = 5.0, duration_days: float = 365.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times_s, u_sun)`` sampled across the span."""
    n = int(duration_days / step_days) + 1
    times_s = np.arange(n) * step_days * 86400.0
    jd0 = julian_date(epoch)
    jd = jd0 + times_s / 86400.0
    r, _ = _sun_vector_from_jd(jd)
    u = r / np.linalg.norm(r, axis=-1, keepdims=True)
    return times_s, u


__all__ = [
    "AU_KM",
    "beta_angle_deg",
    "seasonal_sweep",
    "sun_vector_at",
    "sun_vector_eci",
]
