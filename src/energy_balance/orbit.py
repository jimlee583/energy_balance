"""Keplerian orbit elements and secular-J2 propagation.

The propagator used here is intentionally simple: it holds the semi-major axis,
eccentricity, and inclination constant, lets RAAN and argument of perigee drift
with the secular J2 rates, and advances the mean anomaly with the Kozai-style
corrected mean motion. That is sufficient for energy-balance sizing over days
to weeks; for higher fidelity, an SGP4 or numerical propagator can be added
later and expose the same ``propagate_orbit`` interface.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from .config import OrbitConfig, OrbitSpec

# --- Constants -------------------------------------------------------------------------

MU_EARTH_KM3_S2 = 398_600.4418
R_EARTH_KM = 6378.137
J2 = 1.08262668e-3
EARTH_SIDEREAL_DAY_S = 86_164.0905
EARTH_ORBITAL_RATE_RAD_S = 2.0 * np.pi / (365.2421897 * 86_400.0)


# --- Julian date and sidereal time -----------------------------------------------------


def julian_date(dt: datetime) -> float:
    """Julian date (UT) for a timezone-aware datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt_utc = dt.astimezone(UTC)
    year = dt_utc.year
    month = dt_utc.month
    day = dt_utc.day
    hour = dt_utc.hour + dt_utc.minute / 60.0 + (dt_utc.second + dt_utc.microsecond / 1e6) / 3600.0
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    jd0 = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    return jd0 + hour / 24.0


def gmst_rad(dt: datetime) -> float:
    """Greenwich mean sidereal time (radians)."""
    jd = julian_date(dt)
    t = (jd - 2_451_545.0) / 36_525.0
    gmst_sec = (
        67310.54841
        + (876_600.0 * 3600.0 + 8_640_184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    gmst_deg = (gmst_sec % 86400.0) / 240.0  # 86400 s / 360 deg = 240
    return np.deg2rad(gmst_deg % 360.0)


# --- Resolve orbit elements ------------------------------------------------------------


def _resolve_sma_ecc(orbit: OrbitConfig) -> tuple[float, float]:
    """Return (semi-major axis in km, eccentricity) from the OrbitConfig."""
    if orbit.spec is OrbitSpec.ALTITUDES:
        rp = R_EARTH_KM + orbit.perigee_altitude_km
        ra = R_EARTH_KM + orbit.apogee_altitude_km
        a = 0.5 * (rp + ra)
        e = (ra - rp) / (ra + rp) if (ra + rp) > 0 else 0.0
        return a, e
    return orbit.semi_major_axis_km, orbit.eccentricity


def sun_sync_inclination_deg(semi_major_axis_km: float, eccentricity: float) -> float:
    """Inclination (deg) for a sun-synchronous orbit at the given mean elements.

    Solves the J2 RAAN-drift equation for the inclination that makes the RAAN
    rate equal to Earth's mean orbital rate around the Sun.
    """
    a = semi_major_axis_km
    e = eccentricity
    p = a * (1.0 - e * e)
    n = np.sqrt(MU_EARTH_KM3_S2 / a**3)  # rad/s
    cos_i = -2.0 * EARTH_ORBITAL_RATE_RAD_S / (3.0 * n * J2 * (R_EARTH_KM / p) ** 2)
    cos_i = float(np.clip(cos_i, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos_i)))


def raan_from_ltan(ltan_hours: float, epoch: datetime) -> float:
    """RAAN (deg) that places the ascending node at the given local mean solar time."""
    # Right ascension of the mean Sun at epoch.
    jd = julian_date(epoch)
    t_ut = (jd - 2_451_545.0) / 36_525.0
    lambda_m_sun = np.deg2rad((280.460 + 36_000.771 * t_ut) % 360.0)
    m_sun = np.deg2rad((357.5277233 + 35_999.05034 * t_ut) % 360.0)
    lam = lambda_m_sun + np.deg2rad(1.914666471) * np.sin(m_sun) + np.deg2rad(0.019994643) * np.sin(
        2.0 * m_sun
    )
    eps = np.deg2rad(23.439291 - 0.0130042 * t_ut)
    alpha_sun = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    raan = alpha_sun + np.deg2rad((ltan_hours - 12.0) * 15.0)
    return float(np.rad2deg(raan) % 360.0)


# --- Kepler solver ---------------------------------------------------------------------


def kepler_solve(mean_anomaly: np.ndarray, eccentricity: float, tol: float = 1e-12) -> np.ndarray:
    """Solve Kepler's equation M = E - e sin E for the eccentric anomaly."""
    m = np.mod(mean_anomaly, 2.0 * np.pi)
    e_an = np.where(eccentricity < 0.8, m, np.pi * np.ones_like(m))
    for _ in range(60):
        f = e_an - eccentricity * np.sin(e_an) - m
        fp = 1.0 - eccentricity * np.cos(e_an)
        delta = f / fp
        e_an = e_an - delta
        if np.max(np.abs(delta)) < tol:
            break
    return e_an


def true_anomaly_from_eccentric(e_an: np.ndarray, eccentricity: float) -> np.ndarray:
    s = np.sqrt(1.0 - eccentricity * eccentricity) * np.sin(e_an)
    c = np.cos(e_an) - eccentricity
    return np.arctan2(s, c)


# --- Rotations -------------------------------------------------------------------------


def _rotz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def perifocal_to_eci(raan: float, incl: float, argp: float) -> np.ndarray:
    """Rotation matrix from the perifocal frame to ECI."""
    return _rotz(raan) @ _rotx(incl) @ _rotz(argp)


# --- Secular J2 rates ------------------------------------------------------------------


def secular_j2_rates(
    semi_major_axis_km: float, eccentricity: float, inclination_rad: float
) -> tuple[float, float, float]:
    """Return (raan_dot, argp_dot, mean_motion_corrected) in rad/s."""
    a = semi_major_axis_km
    e = eccentricity
    i = inclination_rad
    n = np.sqrt(MU_EARTH_KM3_S2 / a**3)
    p = a * (1.0 - e * e)
    factor = n * J2 * (R_EARTH_KM / p) ** 2
    raan_dot = -1.5 * factor * np.cos(i)
    argp_dot = 0.75 * factor * (5.0 * np.cos(i) ** 2 - 1.0)
    m_dot_pert = 0.75 * factor * np.sqrt(1.0 - e * e) * (3.0 * np.cos(i) ** 2 - 1.0)
    return float(raan_dot), float(argp_dot), float(n + m_dot_pert)


# --- Public propagation API ------------------------------------------------------------


class OrbitState:
    """Resolved orbital elements and derived propagation rates (SI-ish units).

    Lengths are in km, angles in radians, times in seconds, rates in rad/s.
    """

    def __init__(self, orbit: OrbitConfig, epoch: datetime) -> None:
        a, e = _resolve_sma_ecc(orbit)
        if orbit.sun_synchronous:
            inclination_deg = sun_sync_inclination_deg(a, e)
            raan_deg = raan_from_ltan(orbit.ltan_hours, epoch)
        else:
            inclination_deg = orbit.inclination_deg
            raan_deg = orbit.raan_deg

        self.epoch = epoch
        self.a = a
        self.e = e
        self.i = float(np.deg2rad(inclination_deg))
        self.raan0 = float(np.deg2rad(raan_deg))
        self.argp0 = float(np.deg2rad(orbit.arg_perigee_deg))
        self.nu0 = float(np.deg2rad(orbit.true_anomaly_deg))

        # Initial mean anomaly from the initial true anomaly.
        e_an0 = 2.0 * np.arctan2(
            np.sqrt(1.0 - e) * np.sin(self.nu0 / 2.0),
            np.sqrt(1.0 + e) * np.cos(self.nu0 / 2.0),
        )
        self.m0 = float(e_an0 - e * np.sin(e_an0))

        self.raan_dot, self.argp_dot, self.n_corrected = secular_j2_rates(a, e, self.i)
        self.period_s = float(2.0 * np.pi / self.n_corrected)
        self.inclination_deg = inclination_deg
        self.raan_deg = raan_deg


def propagate_orbit(state: OrbitState, times_s: np.ndarray) -> dict[str, np.ndarray]:
    """Propagate the orbit; return a dict with ECI position/velocity arrays.

    Keys: ``r_eci_km`` (N, 3), ``v_eci_km_s`` (N, 3), ``raan`` (N,), ``argp`` (N,).
    """
    t = np.asarray(times_s, dtype=float)
    a, e = state.a, state.e
    i = state.i
    raan = state.raan0 + state.raan_dot * t
    argp = state.argp0 + state.argp_dot * t
    mean = state.m0 + state.n_corrected * t
    e_an = kepler_solve(mean, e)
    nu = true_anomaly_from_eccentric(e_an, e)

    p = a * (1.0 - e * e)
    r_mag = p / (1.0 + e * np.cos(nu))
    r_pqw = np.stack([r_mag * np.cos(nu), r_mag * np.sin(nu), np.zeros_like(nu)], axis=-1)
    mu = MU_EARTH_KM3_S2
    v_pqw = np.stack(
        [
            -np.sqrt(mu / p) * np.sin(nu),
            np.sqrt(mu / p) * (e + np.cos(nu)),
            np.zeros_like(nu),
        ],
        axis=-1,
    )

    r_eci = np.empty_like(r_pqw)
    v_eci = np.empty_like(v_pqw)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_argp = np.cos(argp)
    sin_argp = np.sin(argp)

    # Combined perifocal -> ECI rotation per step, applied explicitly.
    r11 = cos_raan * cos_argp - sin_raan * sin_argp * cos_i
    r12 = -cos_raan * sin_argp - sin_raan * cos_argp * cos_i
    r21 = sin_raan * cos_argp + cos_raan * sin_argp * cos_i
    r22 = -sin_raan * sin_argp + cos_raan * cos_argp * cos_i
    r31 = sin_argp * sin_i
    r32 = cos_argp * sin_i

    r_eci[:, 0] = r11 * r_pqw[:, 0] + r12 * r_pqw[:, 1]
    r_eci[:, 1] = r21 * r_pqw[:, 0] + r22 * r_pqw[:, 1]
    r_eci[:, 2] = r31 * r_pqw[:, 0] + r32 * r_pqw[:, 1]

    v_eci[:, 0] = r11 * v_pqw[:, 0] + r12 * v_pqw[:, 1]
    v_eci[:, 1] = r21 * v_pqw[:, 0] + r22 * v_pqw[:, 1]
    v_eci[:, 2] = r31 * v_pqw[:, 0] + r32 * v_pqw[:, 1]

    return {
        "r_eci_km": r_eci,
        "v_eci_km_s": v_eci,
        "raan_rad": raan,
        "argp_rad": argp,
        "true_anomaly_rad": nu,
    }
