"""Seasonal beta-angle / eclipse-fraction tests under J2 nodal regression.

Regression coverage for a bug where the "Seasonal" tab computed the Sun's
position over a year but froze the orbit's angular-momentum direction at
epoch, instead of letting it precess with the orbit's own secular J2 nodal
drift. That made beta angle track only the Sun's ~365-day annual motion
against a fixed orbit plane (amplitude ~57 deg for a 1000 km/80 deg orbit),
producing eclipses on every orbit all year. With the orbit plane correctly
precessing, beta should reach a much larger amplitude and cycle on the
relative period between the orbit node's J2 drift and the Sun's apparent
motion (~178 days for this orbit), with real eclipse-free seasons.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from energy_balance.config import OrbitConfig
from energy_balance.orbit import EARTH_ORBITAL_RATE_RAD_S, OrbitState
from energy_balance.sun import beta_angle_deg, seasonal_beta_eclipse


def _leo80_state(epoch: datetime) -> OrbitState:
    cfg = OrbitConfig(perigee_altitude_km=1000.0, apogee_altitude_km=1000.0, inclination_deg=80.0)
    return OrbitState(cfg, epoch)


def test_raan_drift_matches_expected_j2_rate_for_1000km_80deg():
    """Textbook secular J2 nodal regression for this orbit is ~ -1.04 deg/day."""
    epoch = datetime(2026, 3, 20, 12, tzinfo=UTC)
    state = _leo80_state(epoch)
    deg_per_day = np.rad2deg(state.raan_dot) * 86400.0
    assert deg_per_day == pytest.approx(-1.04, abs=0.02)


def test_seasonal_beta_amplitude_exceeds_frozen_orbit_plane_result():
    """The orbit-plane-precessing result must differ substantially from (and
    reach a larger amplitude than) the buggy frozen-orbit-normal result."""
    epoch = datetime(2026, 3, 20, 12, tzinfo=UTC)
    state = _leo80_state(epoch)

    result = seasonal_beta_eclipse(state, epoch, step_days=2.0, duration_days=365.0)
    beta = result["beta_deg"]

    # Reconstruct the old (buggy) computation: orbit normal frozen at epoch,
    # Sun swept over the year.
    from energy_balance.orbit import propagate_orbit
    from energy_balance.sun import seasonal_sweep

    out0 = propagate_orbit(state, np.array([0.0]))
    h0 = np.cross(out0["r_eci_km"][0], out0["v_eci_km_s"][0])
    h0 = h0 / np.linalg.norm(h0)
    _times_s, u_sun = seasonal_sweep(epoch, step_days=2.0, duration_days=365.0)
    beta_frozen = beta_angle_deg(np.tile(h0, (u_sun.shape[0], 1)), u_sun)

    # The physically correct (precessing) amplitude is well beyond the
    # ~57 deg a frozen orbit plane would show for this orbit/epoch.
    assert np.max(np.abs(beta_frozen)) < 60.0
    assert np.max(np.abs(beta)) > 75.0
    assert np.max(np.abs(beta)) <= 90.0 + 1e-6


def test_seasonal_beta_period_matches_node_sun_relative_rate():
    """Beta's cycle period should match the orbit node vs. Sun relative rate
    (~178 days here), not the Sun's bare ~365-day annual period."""
    epoch = datetime(2026, 3, 20, 12, tzinfo=UTC)
    state = _leo80_state(epoch)
    result = seasonal_beta_eclipse(state, epoch, step_days=1.0, duration_days=730.0)
    beta = result["beta_deg"]
    days = result["times_s"] / 86400.0

    sign = np.sign(beta)
    crossing_idx = np.flatnonzero(np.diff(sign) != 0)
    assert len(crossing_idx) >= 6, "expected several beta zero-crossings over two years"
    crossing_days = days[crossing_idx]
    half_periods = np.diff(crossing_days)
    measured_period_days = 2.0 * float(np.median(half_periods))

    sun_rate_deg_day = np.rad2deg(EARTH_ORBITAL_RATE_RAD_S) * 86400.0
    node_rate_deg_day = np.rad2deg(state.raan_dot) * 86400.0
    expected_period_days = 360.0 / abs(sun_rate_deg_day - node_rate_deg_day)

    assert expected_period_days == pytest.approx(178.0, abs=5.0)
    assert measured_period_days == pytest.approx(expected_period_days, rel=0.1)


def test_seasonal_eclipse_has_eclipse_free_seasons():
    """With beta correctly swinging past the eclipse threshold, some seasons
    must be eclipse-free -- not every orbit, all year, as the frozen-orbit-
    plane bug produced."""
    epoch = datetime(2026, 3, 20, 12, tzinfo=UTC)
    state = _leo80_state(epoch)
    result = seasonal_beta_eclipse(state, epoch, step_days=2.0, duration_days=365.0)
    eclipse_frac = result["eclipse_fraction"]

    assert np.any(eclipse_frac == 0.0), "expected at least one eclipse-free sample"
    assert np.any(eclipse_frac > 0.0), "expected at least one sample with eclipses"


def test_seasonal_beta_eclipse_shapes_and_bounds():
    epoch = datetime(2026, 6, 1, 0, tzinfo=UTC)
    state = _leo80_state(epoch)
    result = seasonal_beta_eclipse(state, epoch, step_days=5.0, duration_days=365.0)

    n = len(result["times_s"])
    assert result["beta_deg"].shape == (n,)
    assert result["eclipse_fraction"].shape == (n,)
    assert np.all(np.abs(result["beta_deg"]) <= 90.0 + 1e-9)
    assert np.all((result["eclipse_fraction"] >= 0.0) & (result["eclipse_fraction"] <= 1.0))
