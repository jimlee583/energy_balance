"""Geometry tests for panel_frames_eci and the refactored cos(incidence)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from energy_balance import PanelConfig, PanelMounting, panel_frames_eci


def _identity_rotations(n: int) -> np.ndarray:
    return np.broadcast_to(np.eye(3), (n, 3, 3)).copy()


def _rot_about_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_fixed_panel_normal_is_body_normal_in_eci():
    # Identity rotation: ECI normal == body normal.
    panel = PanelConfig(
        name="fixed",
        area_m2=1.0,
        normal_face="+Z",
        mounting=PanelMounting.FIXED,
    )
    rot = _identity_rotations(3)
    u_sun = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    out = panel_frames_eci(panel, rot, u_sun)
    assert np.allclose(out["normal_eci"], np.tile([0.0, 0.0, 1.0], (3, 1)))
    # Fixed panels never gimbal.
    assert np.allclose(out["angle_deg"], 0.0)
    assert np.allclose(out["angle1_deg"], 0.0)
    assert np.allclose(out["angle2_deg"], 0.0)
    # cos(incidence) equals normal . sun, clipped at 0.
    assert np.allclose(out["cos_incidence"], [1.0, 0.0, 0.0])


def test_two_axis_panel_normal_follows_sun():
    # Unlimited 2-axis panel: exercises only the fast exact-tracking path
    # (no gimbal_min/max_deg or gimbal2_min/max_deg set).
    panel = PanelConfig(
        name="2ax",
        area_m2=1.0,
        mounting=PanelMounting.TWO_AXIS,
    )
    rot = _identity_rotations(4)
    u_sun = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    u_sun = u_sun / np.linalg.norm(u_sun, axis=-1, keepdims=True)
    out = panel_frames_eci(panel, rot, u_sun)
    assert np.allclose(out["normal_eci"], u_sun)
    # cos(incidence) is identically 1.
    assert np.allclose(out["cos_incidence"], 1.0)
    # Edge must be unit length and perpendicular to the normal.
    edge_norms = np.linalg.norm(out["edge_eci"], axis=-1)
    assert np.allclose(edge_norms, 1.0)
    assert np.allclose(np.sum(out["edge_eci"] * out["normal_eci"], axis=-1), 0.0, atol=1e-9)


def test_one_axis_normal_perpendicular_to_axis_and_maximizes_cos():
    # Axis along body +Y, rest normal along body +Z. Rotate about Y so the
    # normal is pulled toward the Sun's component in the XZ plane.
    panel = PanelConfig(
        name="1ax",
        area_m2=1.0,
        normal_face="+Z",
        mounting=PanelMounting.ONE_AXIS,
        rotation_axis_body=(0.0, 1.0, 0.0),
    )
    rot = _identity_rotations(5)
    # Build Sun vectors with varying azimuth in the XZ plane and some Y bias.
    thetas = np.array([0.0, np.pi / 6, np.pi / 3, -np.pi / 4, np.pi / 2])
    y_bias = 0.3
    sun = np.stack(
        [np.sin(thetas), np.full_like(thetas, y_bias), np.cos(thetas)], axis=-1
    )
    sun = sun / np.linalg.norm(sun, axis=-1, keepdims=True)
    out = panel_frames_eci(panel, rot, sun)
    normals = out["normal_eci"]
    # Normals must be perpendicular to the rotation axis.
    assert np.allclose(normals[:, 1], 0.0, atol=1e-9)
    # cos(incidence) equals the magnitude of the Sun's projection perpendicular
    # to the axis (the classical 1-axis maximum).
    expected = np.sqrt(sun[:, 0] ** 2 + sun[:, 2] ** 2)
    assert np.allclose(out["cos_incidence"], expected, atol=1e-9)
    # Signed gimbal angle: with rest normal +Z and axis +Y, +X-direction Sun
    # yields a positive rotation.
    assert out["angle_deg"][0] == 0.0
    assert out["angle_deg"][1] > 0.0  # Sun tilted toward +X
    assert out["angle_deg"][3] < 0.0  # Sun tilted toward -X


def test_two_axis_slew_angle_matches_sun_offset_from_rest():
    panel = PanelConfig(
        name="2ax",
        area_m2=1.0,
        normal_face="+Z",
        mounting=PanelMounting.TWO_AXIS,
    )
    rot = _identity_rotations(3)
    offsets_deg = np.array([0.0, 30.0, 90.0])
    sun = np.stack(
        [
            np.sin(np.deg2rad(offsets_deg)),
            np.zeros_like(offsets_deg),
            np.cos(np.deg2rad(offsets_deg)),
        ],
        axis=-1,
    )
    out = panel_frames_eci(panel, rot, sun)
    assert np.allclose(out["angle_deg"], offsets_deg, atol=1e-9)


def test_rotation_matrix_applies_correctly():
    # Non-identity attitude: with this rotation body +Z maps to ECI -Y.
    rot = _rot_about_x(np.pi / 2)[None, :, :]
    panel = PanelConfig(name="fixed", normal_face="+Z", mounting=PanelMounting.FIXED)
    sun = np.array([[0.0, -1.0, 0.0]])
    out = panel_frames_eci(panel, rot, sun)
    assert np.allclose(out["normal_eci"][0], [0.0, -1.0, 0.0], atol=1e-9)
    assert np.allclose(out["cos_incidence"][0], 1.0)


def test_twist_limited_panel_rewinds_at_rate_and_loses_incidence():
    panel = PanelConfig(
        name="capped",
        area_m2=1.0,
        normal_face="+Z",
        mounting=PanelMounting.ONE_AXIS,
        rotation_axis_body=(0.0, 1.0, 0.0),
        gimbal_min_deg=-180.0,
        gimbal_max_deg=180.0,
        max_slew_rate_deg_s=60.0,
    )
    ideal_deg = np.array([0.0, 170.0, 190.0])
    sun = np.stack(
        [
            np.sin(np.deg2rad(ideal_deg)),
            np.zeros_like(ideal_deg),
            np.cos(np.deg2rad(ideal_deg)),
        ],
        axis=-1,
    )

    out = panel_frames_eci(panel, _identity_rotations(3), sun, np.array([0.0, 1.0, 2.0]))

    np.testing.assert_allclose(out["target_angle_deg"], [0.0, 170.0, -170.0])
    np.testing.assert_allclose(out["angle_deg"], [0.0, 60.0, 0.0])
    assert out["rewinding"].tolist() == [False, False, True]
    assert np.all(out["angle_deg"] >= -180.0)
    assert np.all(out["angle_deg"] <= 180.0)
    assert out["cos_incidence"][-1] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        (-180.0, None),
        (None, 180.0),
        (10.0, 370.0),
        (-370.0, -10.0),
        (10.0, 10.0),
    ],
)
def test_invalid_primary_gimbal_limits_are_rejected(minimum, maximum):
    with pytest.raises(ValidationError):
        PanelConfig(
            mounting=PanelMounting.ONE_AXIS,
            gimbal_min_deg=minimum,
            gimbal_max_deg=maximum,
        )


def test_primary_gimbal_limit_rejected_for_fixed_panel():
    with pytest.raises(ValidationError):
        PanelConfig(
            mounting=PanelMounting.FIXED,
            gimbal_min_deg=-90.0,
            gimbal_max_deg=90.0,
        )


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        (-180.0, None),
        (None, 90.0),
        (-100.0, 80.0),
        (-80.0, 100.0),
        (10.0, 10.0),
    ],
)
def test_invalid_secondary_gimbal_limits_are_rejected(minimum, maximum):
    with pytest.raises(ValidationError):
        PanelConfig(
            mounting=PanelMounting.TWO_AXIS,
            gimbal2_min_deg=minimum,
            gimbal2_max_deg=maximum,
        )


def test_secondary_gimbal_limit_rejected_for_one_axis_panel():
    with pytest.raises(ValidationError):
        PanelConfig(
            mounting=PanelMounting.ONE_AXIS,
            rotation_axis_body=(0.0, 1.0, 0.0),
            gimbal2_min_deg=-70.0,
            gimbal2_max_deg=70.0,
        )


def test_twist_limited_panel_requires_sample_times():
    panel = PanelConfig(
        mounting=PanelMounting.ONE_AXIS,
        gimbal_min_deg=-180.0,
        gimbal_max_deg=180.0,
    )

    with pytest.raises(ValueError, match="times_s is required"):
        panel_frames_eci(panel, _identity_rotations(1), np.array([[0.0, 0.0, 1.0]]))


def test_hard_stop_one_axis_panel_clamps_without_rewind():
    # Span < 360 deg: hard mechanical stop, no rewind, incidence degrades
    # once the Sun passes the limit.
    panel = PanelConfig(
        name="hard-stopped",
        area_m2=1.0,
        normal_face="+Z",
        mounting=PanelMounting.ONE_AXIS,
        rotation_axis_body=(0.0, 1.0, 0.0),
        gimbal_min_deg=-90.0,
        gimbal_max_deg=90.0,
        max_slew_rate_deg_s=60.0,
    )
    ideal_deg = np.array([0.0, 60.0, 120.0])
    sun = np.stack(
        [
            np.sin(np.deg2rad(ideal_deg)),
            np.zeros_like(ideal_deg),
            np.cos(np.deg2rad(ideal_deg)),
        ],
        axis=-1,
    )

    out = panel_frames_eci(panel, _identity_rotations(3), sun, np.array([0.0, 1.0, 2.0]))

    np.testing.assert_allclose(out["target_angle_deg"], [0.0, 60.0, 90.0])
    np.testing.assert_allclose(out["angle_deg"], [0.0, 60.0, 90.0])
    assert out["rewinding"].tolist() == [False, False, False]
    assert out["cos_incidence"][-1] < 1.0


def test_two_axis_primary_axis_limit_clamps_and_degrades():
    panel = PanelConfig(
        name="2ax-primary",
        area_m2=1.0,
        normal_face="+Z",
        mounting=PanelMounting.TWO_AXIS,
        rotation_axis_body=(0.0, 1.0, 0.0),
        gimbal_min_deg=-80.0,
        gimbal_max_deg=80.0,
        max_slew_rate_deg_s=180.0,
    )
    # Sun sweeps in the XZ plane (which is exactly the primary-axis' plane of
    # travel for this geometry) out to 90 deg, beyond the 80 deg limit.
    offsets_deg = np.array([0.0, 60.0, 90.0])
    sun = np.stack(
        [
            np.sin(np.deg2rad(offsets_deg)),
            np.zeros_like(offsets_deg),
            np.cos(np.deg2rad(offsets_deg)),
        ],
        axis=-1,
    )
    times_s = np.array([0.0, 1.0, 2.0])

    out = panel_frames_eci(panel, _identity_rotations(3), sun, times_s)

    np.testing.assert_allclose(out["angle1_deg"], [0.0, 60.0, 80.0], atol=1e-6)
    assert np.allclose(out["angle2_deg"], 0.0, atol=1e-9)
    assert out["cos_incidence"][-1] < 1.0
    assert out["rewinding"].tolist() == [False, False, False]


def test_two_axis_secondary_axis_limit_clamps_and_degrades():
    panel = PanelConfig(
        name="2ax-secondary",
        area_m2=1.0,
        normal_face="+Z",
        mounting=PanelMounting.TWO_AXIS,
        rotation_axis_body=(0.0, 1.0, 0.0),
        gimbal2_min_deg=-70.0,
        gimbal2_max_deg=70.0,
        max_slew_rate_deg_s=180.0,
    )
    # Sun sweeps toward the primary axis itself (body +Y), which is purely a
    # secondary-axis excursion, out to 80 deg, beyond the 70 deg limit.
    offsets_deg = np.array([0.0, 50.0, 80.0])
    sun = np.stack(
        [
            np.zeros_like(offsets_deg),
            np.sin(np.deg2rad(offsets_deg)),
            np.cos(np.deg2rad(offsets_deg)),
        ],
        axis=-1,
    )
    times_s = np.array([0.0, 1.0, 2.0])

    out = panel_frames_eci(panel, _identity_rotations(3), sun, times_s)

    np.testing.assert_allclose(out["angle2_deg"], [0.0, -50.0, -70.0], atol=1e-6)
    assert out["cos_incidence"][-1] < 1.0
    assert out["rewinding"].tolist() == [False, False, False]


def test_two_axis_both_axes_limited_independent_clamp():
    panel = PanelConfig(
        name="2ax-both",
        area_m2=1.0,
        normal_face="+Z",
        mounting=PanelMounting.TWO_AXIS,
        rotation_axis_body=(0.0, 1.0, 0.0),
        gimbal_min_deg=-80.0,
        gimbal_max_deg=80.0,
        gimbal2_min_deg=-70.0,
        gimbal2_max_deg=70.0,
        max_slew_rate_deg_s=180.0,
    )
    # Construct a Sun direction whose ideal primary angle is 90 deg (exceeds
    # the 80 deg limit) and ideal secondary angle is 80 deg (exceeds the
    # 70 deg limit), via the forward kinematics: with n0=(0,0,1), a2=(1,0,0),
    # a1=(0,1,0), n_final = cos(a2)*cos(a1)*n0 + cos(a2)*sin(a1)*a2 - sin(a2)*a1.
    angle1_deg, angle2_deg = 90.0, 80.0
    c1, s1 = np.cos(np.deg2rad(angle1_deg)), np.sin(np.deg2rad(angle1_deg))
    c2, s2 = np.cos(np.deg2rad(angle2_deg)), np.sin(np.deg2rad(angle2_deg))
    sun = np.array([[c2 * s1, -s2, c2 * c1]])
    times_s = np.array([0.0])

    out = panel_frames_eci(panel, _identity_rotations(1), sun, times_s)

    assert abs(out["angle1_deg"][0]) <= 80.0 + 1e-6
    assert abs(out["angle2_deg"][0]) <= 70.0 + 1e-6
    assert out["cos_incidence"][0] < 1.0


def test_two_axis_gimbal_limit_requires_sample_times():
    panel = PanelConfig(
        mounting=PanelMounting.TWO_AXIS,
        gimbal_min_deg=-80.0,
        gimbal_max_deg=80.0,
    )

    with pytest.raises(ValueError, match="times_s is required"):
        panel_frames_eci(panel, _identity_rotations(1), np.array([[0.0, 0.0, 1.0]]))
