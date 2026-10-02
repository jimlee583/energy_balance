"""Geometry tests for panel_frames_eci and the refactored cos(incidence)."""

from __future__ import annotations

import numpy as np

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
    # cos(incidence) equals normal . sun, clipped at 0.
    assert np.allclose(out["cos_incidence"], [1.0, 0.0, 0.0])


def test_two_axis_panel_normal_follows_sun():
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
