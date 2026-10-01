"""Attitude: rotation from the body frame to ECI for a few common modes.

Each builder returns a stack of 3x3 matrices with shape (N, 3, 3) such that
``r_eci = R @ r_body``.

Modes implemented:
- ``sun_pointing``: a chosen body axis points at the Sun. The remaining axes
  are chosen to be orthonormal but their exact phase doesn't matter for a
  2-axis-tracking array that already has the Sun on its normal.
- ``nadir_pointing``: the +Z body axis points toward Earth (nadir), the chosen
  yaw axis is placed along the velocity direction, and the third axis completes
  the right-handed frame.
- ``inertial``: identity rotation (body axes aligned with ECI).
"""

from __future__ import annotations

import numpy as np

from .config import BODY_FACES, AttitudeConfig, AttitudeMode


def _orthonormal_basis(primary: np.ndarray, hint: np.ndarray) -> np.ndarray:
    """Return a 3x3 column-stacked basis whose first column is ``primary``.

    ``primary`` is (N, 3), ``hint`` is (N, 3) and must not be parallel to primary.
    """
    primary = primary / np.linalg.norm(primary, axis=-1, keepdims=True)
    # Second axis orthogonal to primary, derived from hint.
    tmp = hint - np.sum(hint * primary, axis=-1, keepdims=True) * primary
    norms = np.linalg.norm(tmp, axis=-1, keepdims=True)
    # If hint is parallel to primary, choose an arbitrary perpendicular vector.
    fallback = np.where(
        np.abs(primary[..., 2:3]) < 0.9,
        np.tile(np.array([0.0, 0.0, 1.0]), (primary.shape[0], 1)),
        np.tile(np.array([1.0, 0.0, 0.0]), (primary.shape[0], 1)),
    )
    tmp = np.where(norms < 1e-9, fallback, tmp)
    second = tmp / np.linalg.norm(tmp, axis=-1, keepdims=True)
    third = np.cross(primary, second)
    return np.stack([primary, second, third], axis=-1)  # (N, 3, 3) with columns = axes


def body_to_eci(
    attitude: AttitudeConfig,
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    u_sun_eci: np.ndarray,
) -> np.ndarray:
    """Return rotation matrices body->ECI, shape (N, 3, 3)."""
    r_eci_km = np.atleast_2d(r_eci_km)
    v_eci_km_s = np.atleast_2d(v_eci_km_s)
    u_sun_eci = np.atleast_2d(u_sun_eci)
    n_pts = r_eci_km.shape[0]

    if attitude.mode is AttitudeMode.INERTIAL:
        return np.broadcast_to(np.eye(3), (n_pts, 3, 3)).copy()

    if attitude.mode is AttitudeMode.SUN_POINTING:
        # The requested body axis points at the Sun. Build an orthonormal basis
        # whose first column is the Sun direction in ECI.
        axis_body = np.array(BODY_FACES[attitude.sun_pointing_axis])
        # We need body-to-ECI such that R @ axis_body = u_sun.
        # Build B = [axis_body, e2, e3] and E = [u_sun, f2, f3] (orthonormal).
        # Then R = E @ B^T.
        hint_eci = np.tile(np.array([0.0, 0.0, 1.0]), (n_pts, 1))
        e_basis = _orthonormal_basis(u_sun_eci, hint_eci)
        hint_body = np.tile(np.array([0.0, 0.0, 1.0]), (n_pts, 1))
        if np.abs(axis_body[2]) > 0.9:
            hint_body = np.tile(np.array([1.0, 0.0, 0.0]), (n_pts, 1))
        b_basis = _orthonormal_basis(np.tile(axis_body, (n_pts, 1)), hint_body)
        return np.einsum("nij,nkj->nik", e_basis, b_basis)

    # Nadir pointing.
    nadir = -r_eci_km / np.linalg.norm(r_eci_km, axis=-1, keepdims=True)
    v_hat = v_eci_km_s / np.linalg.norm(v_eci_km_s, axis=-1, keepdims=True)
    # +Z body axis -> nadir, yaw axis -> velocity direction.
    yaw_axis_body = np.array(BODY_FACES[attitude.nadir_yaw_axis])
    z_body_target_eci = nadir
    yaw_target_eci = v_hat
    # Build ECI basis with first column = z_body_target_eci, second from yaw hint.
    eci_basis = _orthonormal_basis(z_body_target_eci, yaw_target_eci)
    body_axis_cols = _orthonormal_basis(
        np.tile(np.array(BODY_FACES["+Z"]), (n_pts, 1)),
        np.tile(yaw_axis_body, (n_pts, 1)),
    )
    return np.einsum("nij,nkj->nik", eci_basis, body_axis_cols)


__all__ = ["body_to_eci"]
