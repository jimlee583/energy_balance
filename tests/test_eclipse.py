"""Eclipse model tests against analytic formulas and known orbits."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from energy_balance.config import OrbitConfig
from energy_balance.eclipse import (
    R_EARTH_KM,
    R_SUN_KM,
    analytic_eclipse_fraction,
    illumination_fraction_conical,
    illumination_fraction_cylindrical,
)
from energy_balance.orbit import OrbitState, propagate_orbit
from energy_balance.sun import sun_vector_eci


def test_cylindrical_shadow_simple_geometry():
    r_sun = np.array([1.496e8, 0.0, 0.0])
    # Satellite directly behind Earth (negative X), close to Earth.
    r_sat = np.array([-R_EARTH_KM - 400.0, 0.0, 0.0])
    assert illumination_fraction_cylindrical(r_sat, r_sun) == pytest.approx(0.0)
    # Satellite in front of Earth: fully illuminated.
    assert illumination_fraction_cylindrical(np.array([R_EARTH_KM + 400.0, 0, 0]), r_sun) == pytest.approx(1.0)
    # Satellite well off to the side: fully illuminated.
    assert illumination_fraction_cylindrical(np.array([0.0, 2 * R_EARTH_KM, 0.0]), r_sun) == pytest.approx(1.0)


def test_conical_matches_cylindrical_for_bulk():
    rng = np.random.default_rng(1)
    r_sun = np.array([1.496e8, 0.0, 0.0])
    r_sat = rng.normal(scale=7000.0, size=(500, 3))
    cyl = illumination_fraction_cylindrical(r_sat, np.broadcast_to(r_sun, r_sat.shape))
    con = illumination_fraction_conical(r_sat, np.broadcast_to(r_sun, r_sat.shape))
    # Conical gives penumbra fractions near the edges; most points should match.
    diffs = np.abs(cyl - con)
    assert np.mean(diffs < 0.01) > 0.9


def test_conical_fully_illuminated_when_far_from_shadow():
    r_sun = np.array([1.496e8, 0.0, 0.0])
    r_sat = np.array([7000.0, 0.0, 0.0])  # Satellite on the sunlit side.
    assert illumination_fraction_conical(r_sat, r_sun) == pytest.approx(1.0)


def test_conical_umbra_center_dark():
    r_sun = np.array([1.496e8, 0.0, 0.0])
    r_sat = np.array([-7000.0, 0.0, 0.0])  # Satellite centered in Earth's shadow.
    assert illumination_fraction_conical(r_sat, r_sun) == pytest.approx(0.0, abs=1e-6)


def test_analytic_eclipse_fraction_zero_beta():
    # At beta = 0, 400 km altitude -> ~36 minutes out of ~92.6 min -> ~0.388 fraction.
    frac = analytic_eclipse_fraction(400.0, 0.0)
    assert frac == pytest.approx(0.388, abs=0.01)


def test_analytic_eclipse_fraction_zero_above_beta_star():
    frac = analytic_eclipse_fraction(500.0, 85.0)
    assert frac == 0.0


def test_iss_eclipse_duration_at_beta_zero():
    """A fresh ISS-like orbit near beta=0 should spend ~36 min in eclipse per revolution."""
    cfg = OrbitConfig(perigee_altitude_km=400.0, apogee_altitude_km=400.0, inclination_deg=51.6)
    epoch = datetime(2026, 3, 20, 12, tzinfo=UTC)
    state = OrbitState(cfg, epoch)
    times = np.linspace(0.0, state.period_s, 2000)
    orbit = propagate_orbit(state, times)
    sun = sun_vector_eci(epoch, times)
    illum = illumination_fraction_conical(orbit["r_eci_km"], sun["r_sun_eci_km"])
    eclipse_frac = 1.0 - np.mean(illum)
    duration_min = eclipse_frac * state.period_s / 60.0
    assert duration_min == pytest.approx(36.0, abs=1.5)


def test_eclipse_model_constants_sanity():
    assert R_SUN_KM > R_EARTH_KM
