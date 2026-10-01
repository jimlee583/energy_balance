"""Validation tests against classical textbook values."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from energy_balance.config import OrbitConfig
from energy_balance.orbit import (
    EARTH_ORBITAL_RATE_RAD_S,
    MU_EARTH_KM3_S2,
    R_EARTH_KM,
    OrbitState,
    kepler_solve,
    propagate_orbit,
    raan_from_ltan,
    secular_j2_rates,
    sun_sync_inclination_deg,
    true_anomaly_from_eccentric,
)


def test_iss_period_matches_textbook():
    cfg = OrbitConfig(perigee_altitude_km=420, apogee_altitude_km=420, inclination_deg=51.6)
    epoch = datetime(2026, 3, 20, 12, tzinfo=UTC)
    state = OrbitState(cfg, epoch)
    # ~92.7 min textbook; secular-J2 correction tightens slightly. Allow 1% tolerance.
    assert state.period_s / 60.0 == pytest.approx(92.7, rel=0.01)


def test_sun_sync_inclination_known_values():
    assert sun_sync_inclination_deg(R_EARTH_KM + 550.0, 0.0) == pytest.approx(97.6, abs=0.1)
    assert sun_sync_inclination_deg(R_EARTH_KM + 800.0, 0.0) == pytest.approx(98.6, abs=0.1)


def test_sun_sync_raan_drift_matches_earth_rate():
    inc = sun_sync_inclination_deg(R_EARTH_KM + 600.0, 0.0)
    raan_dot, _, _ = secular_j2_rates(R_EARTH_KM + 600.0, 0.0, np.deg2rad(inc))
    deg_per_day = np.rad2deg(raan_dot) * 86400.0
    expected = np.rad2deg(EARTH_ORBITAL_RATE_RAD_S) * 86400.0
    assert deg_per_day == pytest.approx(expected, rel=1e-3)
    assert deg_per_day == pytest.approx(0.9856, abs=0.01)


def test_kepler_solver_identity():
    e = 0.3
    e_an = np.linspace(0.0, 2 * np.pi, 100)
    mean = e_an - e * np.sin(e_an)
    solved = kepler_solve(mean, e)
    assert np.allclose(np.mod(solved, 2 * np.pi), np.mod(e_an, 2 * np.pi), atol=1e-10)


def test_true_anomaly_round_trip():
    e = 0.2
    nu = np.linspace(0.1, 2 * np.pi - 0.1, 50)
    e_an = 2.0 * np.arctan2(
        np.sqrt(1 - e) * np.sin(nu / 2.0), np.sqrt(1 + e) * np.cos(nu / 2.0)
    )
    nu_round = true_anomaly_from_eccentric(e_an, e)
    assert np.allclose(np.mod(nu_round, 2 * np.pi), np.mod(nu, 2 * np.pi), atol=1e-10)


def test_circular_orbit_radius_constant():
    cfg = OrbitConfig(perigee_altitude_km=500, apogee_altitude_km=500, inclination_deg=45.0)
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    state = OrbitState(cfg, epoch)
    times = np.linspace(0, 2 * state.period_s, 200)
    out = propagate_orbit(state, times)
    r_mag = np.linalg.norm(out["r_eci_km"], axis=1)
    assert np.allclose(r_mag, R_EARTH_KM + 500.0, atol=1e-3)


def test_eccentric_orbit_perigee_and_apogee():
    perigee = 400.0
    apogee = 1200.0
    cfg = OrbitConfig(
        perigee_altitude_km=perigee, apogee_altitude_km=apogee, inclination_deg=30.0
    )
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    state = OrbitState(cfg, epoch)
    times = np.linspace(0, state.period_s, 2000)
    r_mag = np.linalg.norm(propagate_orbit(state, times)["r_eci_km"], axis=1)
    assert np.min(r_mag) == pytest.approx(R_EARTH_KM + perigee, abs=1.0)
    assert np.max(r_mag) == pytest.approx(R_EARTH_KM + apogee, abs=1.0)


def test_two_body_speed_matches_vis_viva():
    cfg = OrbitConfig(perigee_altitude_km=500, apogee_altitude_km=500, inclination_deg=0.0)
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    state = OrbitState(cfg, epoch)
    out = propagate_orbit(state, np.array([0.0]))
    v = out["v_eci_km_s"][0]
    r = out["r_eci_km"][0]
    expected_speed = np.sqrt(MU_EARTH_KM3_S2 / np.linalg.norm(r))
    assert np.linalg.norm(v) == pytest.approx(expected_speed, rel=1e-3)


def test_raan_from_ltan_is_in_range():
    epoch = datetime(2026, 3, 20, 12, tzinfo=UTC)
    r0 = raan_from_ltan(0.0, epoch)
    r12 = raan_from_ltan(12.0, epoch)
    assert 0.0 <= r0 < 360.0
    assert 0.0 <= r12 < 360.0
    # Separation of 12 hours in LTAN should move RAAN by ~180 degrees.
    diff = (r12 - r0) % 360.0
    assert diff == pytest.approx(180.0, abs=0.1)
