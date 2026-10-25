"""Tests for the Streamlit app's symmetric solar-wing configuration."""

from __future__ import annotations

import numpy as np
import pytest

from app import (
    _default_wing_side_axis,
    _orbit_display_rotation,
    _solar_array_summary,
    _symmetric_wing_panels,
    _velocity_fixed_display_rotation,
)
from energy_balance import PanelConfig, PanelMounting, SolarArrayConfig


@pytest.mark.parametrize("beta_deg", [-35.0, 0.0, 42.0])
def test_orbit_display_rotation_makes_beta_visible(beta_deg):
    beta = np.deg2rad(beta_deg)
    orbit_normal = np.array([0.2, -0.4, 0.7])
    orbit_normal /= np.linalg.norm(orbit_normal)
    in_plane = np.array([1.0, 0.5, 0.0])
    in_plane -= np.dot(in_plane, orbit_normal) * orbit_normal
    in_plane /= np.linalg.norm(in_plane)
    sun = np.cos(beta) * in_plane + np.sin(beta) * orbit_normal

    rotation = _orbit_display_rotation(orbit_normal, sun)

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotation @ orbit_normal, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        rotation @ sun,
        [np.cos(beta), 0.0, np.sin(beta)],
        atol=1e-12,
    )


def test_orbit_display_rotation_is_stable_at_polar_sun():
    orbit_normal = np.array([0.0, 0.0, 1.0])

    rotation = _orbit_display_rotation(orbit_normal, orbit_normal)

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotation @ orbit_normal, [0.0, 0.0, 1.0], atol=1e-12)


def test_velocity_fixed_display_rotation_holds_velocity_along_positive_x():
    velocity = np.array([0.4, -0.8, 0.3])
    orbit_normal = np.array([0.2, 0.4, 0.8])
    velocity -= np.dot(velocity, orbit_normal) / np.dot(
        orbit_normal, orbit_normal,
    ) * orbit_normal

    rotation = _velocity_fixed_display_rotation(velocity, orbit_normal)

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        rotation @ (velocity / np.linalg.norm(velocity)),
        [1.0, 0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rotation @ (orbit_normal / np.linalg.norm(orbit_normal)),
        [0.0, 0.0, 1.0],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("side_axis", "axis", "names"),
    [
        ("±Y", (0.0, 1.0, 0.0), ["Wing (+Y)", "Wing (-Y)"]),
        ("±X", (1.0, 0.0, 0.0), ["Wing (+X)", "Wing (-X)"]),
    ],
)
def test_symmetric_wings_share_per_wing_parameters(side_axis, axis, names):
    shared = PanelConfig(
        name="ignored",
        area_m2=3.25,
        normal_face="+Z",
        mounting=PanelMounting.TWO_AXIS,
        cell_efficiency=0.28,
        packing_factor=0.9,
        inherent_degradation=0.8,
        degradation_per_year=0.03,
        mission_year=4.0,
        temperature_loss=0.12,
    )

    wings = _symmetric_wing_panels(shared, side_axis)

    assert [wing.name for wing in wings] == names
    assert [wing.area_m2 for wing in wings] == [3.25, 3.25]
    assert [wing.rotation_axis_body for wing in wings] == [axis, axis]

    first = wings[0].model_dump(exclude={"name"})
    second = wings[1].model_dump(exclude={"name"})
    assert first == second


def test_symmetric_wings_copy_twist_cap_parameters():
    shared = PanelConfig(
        mounting=PanelMounting.ONE_AXIS,
        gimbal_min_deg=-270.0,
        gimbal_max_deg=270.0,
        max_slew_rate_deg_s=0.75,
    )

    wings = _symmetric_wing_panels(shared, "±Y")

    assert [wing.gimbal_min_deg for wing in wings] == [-270.0, -270.0]
    assert [wing.gimbal_max_deg for wing in wings] == [270.0, 270.0]
    assert [wing.max_slew_rate_deg_s for wing in wings] == [0.75, 0.75]


def test_symmetric_wings_copy_secondary_gimbal_limit_parameters():
    shared = PanelConfig(
        mounting=PanelMounting.TWO_AXIS,
        gimbal_min_deg=-80.0,
        gimbal_max_deg=80.0,
        gimbal2_min_deg=-70.0,
        gimbal2_max_deg=70.0,
        max_slew_rate_deg_s=0.75,
    )

    wings = _symmetric_wing_panels(shared, "±Y")

    assert [wing.gimbal_min_deg for wing in wings] == [-80.0, -80.0]
    assert [wing.gimbal_max_deg for wing in wings] == [80.0, 80.0]
    assert [wing.gimbal2_min_deg for wing in wings] == [-70.0, -70.0]
    assert [wing.gimbal2_max_deg for wing in wings] == [70.0, 70.0]


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        ((1.0, 0.0, 0.0), "±X"),
        ((-0.8, 0.2, 0.0), "±X"),
        ((0.0, -1.0, 0.0), "±Y"),
        ((0.0, 0.0, 1.0), "±Y"),
    ],
)
def test_default_wing_side_axis_uses_stored_panel_axis(axis, expected):
    panel = PanelConfig(rotation_axis_body=axis)

    assert _default_wing_side_axis(panel) == expected


def test_solar_array_summary_describes_paired_wings_and_total_area():
    array = SolarArrayConfig(
        panels=[
            PanelConfig(area_m2=4.0, rotation_axis_body=(0.0, 1.0, 0.0)),
            PanelConfig(area_m2=4.0, rotation_axis_body=(0.0, 1.0, 0.0)),
        ]
    )

    assert _solar_array_summary(array) == "2 panels (±Y), 8.0 m²"


def test_solar_array_summary_describes_single_panel():
    array = SolarArrayConfig(panels=[PanelConfig(area_m2=2.5)])

    assert _solar_array_summary(array) == "1 panel, 2.5 m²"


def test_solar_array_summary_marks_gimbal_limited_arrays():
    array = SolarArrayConfig(
        panels=[
            PanelConfig(
                area_m2=2.5,
                mounting=PanelMounting.ONE_AXIS,
                gimbal_min_deg=-270.0,
                gimbal_max_deg=270.0,
            )
        ]
    )

    assert _solar_array_summary(array) == "1 panel, 2.5 m², gimbal-limited"


def test_solar_array_summary_marks_secondary_axis_limited_arrays():
    array = SolarArrayConfig(
        panels=[
            PanelConfig(
                area_m2=2.5,
                mounting=PanelMounting.TWO_AXIS,
                gimbal2_min_deg=-70.0,
                gimbal2_max_deg=70.0,
            )
        ]
    )

    assert _solar_array_summary(array) == "1 panel, 2.5 m², gimbal-limited"
