"""Risk Model - collision probability computation.

Implements the Foster short-encounter 2D Gaussian Pc formulation.
Projects combined covariance onto the encounter (B) plane and
integrates the 2D Gaussian over a circular hard-body region.

Reference: Foster (1992), "Short-encounter Gaussian collision probability."
"""

from __future__ import annotations

import numpy as np
from scipy.special import expi

from stresslab.types import RiskResult


def _pc_foster_2d(
    miss_eta: float,
    miss_zeta: float,
    C_2d: np.ndarray,
    hard_body_radius: float,
) -> float:
    """Compute 2D collision probability using Foster formulation.

    The Pc is the integral of a 2D Gaussian over a circular disk
    of radius = combined hard-body radius, centered at the
    miss-distance vector in the B-plane.

    For the 2D Gaussian N(mu, C) integrated over disk of radius R:
        Pc = (R^2 / (2 * det(C)^0.5)) * exp(-0.5 * mu^T C^{-1} mu)
             * (approximation for small R relative to sigmas)

    More precisely, we use the series expansion:
        Pc = 1 - exp(-R^2 / (2 * sigma_max^2))  (upper bound, Alfriend)

    We implement the exact 2D integral via eigendecomposition.
    """
    # Ensure C_2d is positive definite
    eigvals, eigvecs = np.linalg.eigh(C_2d)
    eigvals = np.maximum(eigvals, 1e-30)

    # Rotate miss vector into principal axes
    mu = np.array([miss_eta, miss_zeta])
    mu_rot = eigvecs.T @ mu

    sigma1_sq = eigvals[0]
    sigma2_sq = eigvals[1]

    # Mahalanobis distance squared
    d_sq = mu_rot[0]**2 / sigma1_sq + mu_rot[1]**2 / sigma2_sq
    R = hard_body_radius
    R_sq = R * R

    # Foster approximation: valid when R << sigma
    # Pc ~ (R^2 / (2 * sqrt(sigma1_sq * sigma2_sq))) * exp(-d_sq / 2)
    det_C = sigma1_sq * sigma2_sq
    if det_C < 1e-60:
        return 0.0

    pc = (R_sq / (2.0 * np.sqrt(det_C))) * np.exp(-0.5 * d_sq)

    return float(np.clip(pc, 0.0, 1.0))


def compute_collision_probability(
    rel_position: np.ndarray,
    P_obj1: np.ndarray,
    P_obj2: np.ndarray,
    eta: np.ndarray,
    zeta: np.ndarray,
    combined_hard_body_radius: float,
) -> float:
    """Compute collision probability.

    Args:
        rel_position: (3,) relative position [km]
        P_obj1: (6,6) covariance of object 1
        P_obj2: (6,6) covariance of object 2
        eta: (3,) B-plane basis vector 1
        zeta: (3,) B-plane basis vector 2
        combined_hard_body_radius: combined hard-body radius [km]

    Returns:
        Collision probability.
    """
    # Combined position covariance (3x3 upper-left block)
    P_combined_pos = P_obj1[:3, :3] + P_obj2[:3, :3]

    # Project to B-plane
    T = np.vstack([eta, zeta])
    C_2d = T @ P_combined_pos @ T.T

    # Miss vector in B-plane
    miss_eta = np.dot(rel_position, eta)
    miss_zeta = np.dot(rel_position, zeta)

    return _pc_foster_2d(miss_eta, miss_zeta, C_2d, combined_hard_body_radius)


def compute_risk(
    rel_position: np.ndarray,
    P_baseline_obj1: np.ndarray,
    P_baseline_obj2: np.ndarray,
    P_degraded_obj1: np.ndarray,
    P_degraded_obj2: np.ndarray,
    eta: np.ndarray,
    zeta: np.ndarray,
    combined_hard_body_radius: float,
) -> RiskResult:
    """Compute baseline and degraded collision probabilities.

    Baseline: Pc with nominal covariance (no outage effects).
    Degraded: Pc with actual covariance (includes outages).

    In practice during the simulation, we track both a "nominal"
    covariance (always updated) and the actual covariance. For
    simplicity, if only one covariance stream is available, the
    baseline can use the same covariance and the degraded uses
    the one with outage effects applied.
    """
    pc_baseline = compute_collision_probability(
        rel_position, P_baseline_obj1, P_baseline_obj2,
        eta, zeta, combined_hard_body_radius,
    )
    pc_degraded = compute_collision_probability(
        rel_position, P_degraded_obj1, P_degraded_obj2,
        eta, zeta, combined_hard_body_radius,
    )

    ratio = pc_degraded / max(pc_baseline, 1e-30)

    return RiskResult(
        pc_baseline=pc_baseline,
        pc_degraded=pc_degraded,
        risk_ratio=ratio,
    )
