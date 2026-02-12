"""Geometry Metrics - encounter geometry computation.

Computes B-plane (Alfriend convention) geometry, miss distance,
and time to TCA from relative motion data.

B-plane convention:
  xi   = unit vector along relative velocity at TCA
  zeta = h_hat x xi (in the encounter plane)
  eta  = xi x zeta  (completes right-handed frame)
"""

from __future__ import annotations

import numpy as np

from stresslab.types import GeometryResult


def compute_geometry(
    rel_position: np.ndarray,
    rel_velocity: np.ndarray,
    estimated_tca: float,
) -> GeometryResult:
    """Compute encounter geometry at current timestep.

    Args:
        rel_position: (3,) relative position vector obj2 - obj1 [km]
        rel_velocity: (3,) relative velocity vector obj2 - obj1 [km/s]
        estimated_tca: estimated time to closest approach [seconds]

    Returns:
        GeometryResult with B-plane basis vectors and metrics.
    """
    r_rel = np.linalg.norm(rel_position)
    v_rel = np.linalg.norm(rel_velocity)

    # B-plane basis vectors (Alfriend convention)
    if v_rel > 1e-12:
        xi = rel_velocity / v_rel  # along relative velocity
    else:
        xi = np.array([1.0, 0.0, 0.0])

    # Use angular momentum direction as reference
    h = np.cross(rel_position, rel_velocity)
    h_norm = np.linalg.norm(h)
    if h_norm > 1e-12:
        h_hat = h / h_norm
        zeta = np.cross(h_hat, xi)
        zeta_norm = np.linalg.norm(zeta)
        if zeta_norm > 1e-12:
            zeta = zeta / zeta_norm
        else:
            # Fallback: pick arbitrary perpendicular
            zeta = _arbitrary_perp(xi)
    else:
        zeta = _arbitrary_perp(xi)

    eta = np.cross(xi, zeta)

    # Miss distance: component of rel_position in the B-plane
    # (perpendicular to relative velocity)
    b_eta = np.dot(rel_position, eta)
    b_zeta = np.dot(rel_position, zeta)
    miss_distance = np.sqrt(b_eta**2 + b_zeta**2)

    return GeometryResult(
        miss_distance=miss_distance,
        rel_velocity_at_tca=v_rel,
        time_to_tca=max(estimated_tca, 0.0),
        b_plane_xi=xi,
        b_plane_eta=eta,
        b_plane_zeta=zeta,
    )


def _arbitrary_perp(v: np.ndarray) -> np.ndarray:
    """Return a unit vector perpendicular to v."""
    if abs(v[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    perp = np.cross(v, ref)
    return perp / np.linalg.norm(perp)


def project_covariance_to_bplane(
    P_combined: np.ndarray,
    eta: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:
    """Project combined position covariance onto the 2D B-plane.

    Args:
        P_combined: (3,3) combined position covariance in ECI
        eta: (3,) B-plane basis vector 1
        zeta: (3,) B-plane basis vector 2

    Returns:
        (2,2) covariance in B-plane coordinates [eta, zeta].
    """
    # Projection matrix: 2x3
    T = np.vstack([eta, zeta])  # (2, 3)
    return T @ P_combined @ T.T
