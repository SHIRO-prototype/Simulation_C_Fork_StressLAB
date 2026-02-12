"""Covariance Engine - propagates uncertainty and applies Kalman updates.

Propagates covariance using the STM from the propagation engine.
Process noise is defined in RTN frame and rotated to ECI.
Measurement model is full-state position observation: H = [I_3x3 | 0_3x3].
"""

from __future__ import annotations

import numpy as np

from stresslab.types import (
    CovarianceResult,
    ProcessNoiseConfig,
    MU_EARTH_KM3S2,
)


# ---------------------------------------------------------------------------
# Frame rotation helpers
# ---------------------------------------------------------------------------

def _eci_to_rtn_rotation(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Compute 3x3 rotation matrix from ECI to RTN frame.

    R (radial)     = r_hat
    T (tangential) = cross(r, v) x r / |...| ~ along-track
    N (normal)     = cross(r, v) / |cross(r, v)|
    """
    r_hat = pos / np.linalg.norm(pos)
    h = np.cross(pos, vel)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)

    # Rows of rotation matrix: ECI -> RTN
    return np.vstack([r_hat, t_hat, n_hat])


def _process_noise_eci(
    pos: np.ndarray,
    vel: np.ndarray,
    pn_config: ProcessNoiseConfig,
    dt: float,
) -> np.ndarray:
    """Compute 6x6 process noise in ECI from RTN acceleration noise.

    Q_ECI maps acceleration noise in RTN through a velocity-only
    noise coupling over the timestep dt.

    Model: Piecewise-constant acceleration noise.
    Q_pos  = Q_accel * dt^4 / 4
    Q_pv   = Q_accel * dt^3 / 2
    Q_vel  = Q_accel * dt^2
    """
    R_rtn_to_eci = _eci_to_rtn_rotation(pos, vel).T  # 3x3, RTN->ECI

    Q_rtn_3 = pn_config.Q_rtn()  # 3x3 acceleration noise in RTN
    Q_eci_3 = R_rtn_to_eci @ Q_rtn_3 @ R_rtn_to_eci.T

    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt

    Q = np.zeros((6, 6))
    Q[:3, :3] = Q_eci_3 * dt4 / 4.0
    Q[:3, 3:6] = Q_eci_3 * dt3 / 2.0
    Q[3:6, :3] = Q_eci_3 * dt3 / 2.0
    Q[3:6, 3:6] = Q_eci_3 * dt2
    return Q


# ---------------------------------------------------------------------------
# Kalman measurement update
# ---------------------------------------------------------------------------

def _measurement_update(
    P: np.ndarray,
    noise_sigma_pos: float,
) -> np.ndarray:
    """Apply Kalman measurement update with position-only observation.

    H = [I_3x3 | 0_3x3]
    R = sigma^2 * I_3x3

    Returns updated covariance.
    """
    H = np.zeros((3, 6))
    H[:3, :3] = np.eye(3)
    R = noise_sigma_pos**2 * np.eye(3)

    S = H @ P @ H.T + R  # 3x3 innovation covariance
    K = P @ H.T @ np.linalg.inv(S)  # 6x3 Kalman gain

    # Joseph form for numerical stability
    I6 = np.eye(6)
    IKH = I6 - K @ H
    P_updated = IKH @ P @ IKH.T + K @ R @ K.T
    return P_updated


# ---------------------------------------------------------------------------
# Maneuver uncertainty injection
# ---------------------------------------------------------------------------

def _inject_maneuver_uncertainty(
    P: np.ndarray,
    delta_v_sigma: float,
) -> np.ndarray:
    """Add maneuver execution uncertainty to covariance.

    Adds isotropic velocity uncertainty.
    """
    Q_man = np.zeros((6, 6))
    Q_man[3:6, 3:6] = delta_v_sigma**2 * np.eye(3)
    return P + Q_man


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def propagate_covariance(
    P: np.ndarray,
    stm: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    pn_config: ProcessNoiseConfig,
    dt: float,
) -> np.ndarray:
    """Propagate covariance one timestep.

    P_new = STM @ P @ STM^T + Q
    """
    Q = _process_noise_eci(pos, vel, pn_config, dt)
    P_new = stm @ P @ stm.T + Q
    # Enforce symmetry
    P_new = 0.5 * (P_new + P_new.T)
    return P_new


def measurement_update(
    P: np.ndarray,
    noise_sigma_pos: float,
) -> np.ndarray:
    """Apply Kalman measurement update (position-only)."""
    return _measurement_update(P, noise_sigma_pos)


def inject_maneuver(
    P: np.ndarray,
    delta_v_sigma: float,
) -> np.ndarray:
    """Inject maneuver execution uncertainty."""
    return _inject_maneuver_uncertainty(P, delta_v_sigma)


def covariance_metrics(
    P: np.ndarray,
    P_prev_trace: float,
    dt: float,
) -> CovarianceResult:
    """Compute covariance summary metrics."""
    trace = np.trace(P)
    frob = np.linalg.norm(P, "fro")
    eigvals = np.linalg.eigvalsh(P)
    max_eig = float(np.max(eigvals))

    growth_rate = (trace - P_prev_trace) / dt if dt > 0 else 0.0

    return CovarianceResult(
        covariance=P,
        trace=trace,
        frobenius=frob,
        max_eigenvalue=max_eig,
        growth_rate=growth_rate,
    )
