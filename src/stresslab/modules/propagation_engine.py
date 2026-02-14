"""Propagation Engine - orbital propagation with STM computation.

Supports two-body and two-body + J2 dynamics. Computes state transition
matrices by integrating the variational equations alongside the state.
All computations in ECI frame, units km and km/s.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from stresslab.stresslab_types import (
    DynamicsModel,
    MU_EARTH_KM3S2,
    RE_EARTH_KM,
    J2,
    StateVector,
    PropagationResult,
)


# ---------------------------------------------------------------------------
# Dynamics functions
# ---------------------------------------------------------------------------

def _two_body_accel(pos: np.ndarray) -> np.ndarray:
    """Two-body gravitational acceleration."""
    r = np.linalg.norm(pos)
    return -MU_EARTH_KM3S2 / r**3 * pos


def _two_body_jacobian(pos: np.ndarray) -> np.ndarray:
    """Jacobian of two-body acceleration w.r.t. position (3x3)."""
    r = np.linalg.norm(pos)
    r2 = r * r
    r5 = r**5
    I3 = np.eye(3)
    return MU_EARTH_KM3S2 * (3.0 * np.outer(pos, pos) / r5 - I3 / r**3)


def _j2_accel(pos: np.ndarray) -> np.ndarray:
    """J2 perturbation acceleration in ECI."""
    x, y, z = pos
    r = np.linalg.norm(pos)
    r2 = r * r
    factor = -1.5 * J2 * MU_EARTH_KM3S2 * RE_EARTH_KM**2 / r**5
    z2_r2 = (z / r) ** 2

    ax = factor * x * (1.0 - 5.0 * z2_r2)
    ay = factor * y * (1.0 - 5.0 * z2_r2)
    az = factor * z * (3.0 - 5.0 * z2_r2)
    return np.array([ax, ay, az])


def _j2_jacobian(pos: np.ndarray) -> np.ndarray:
    """Jacobian of J2 acceleration w.r.t. position (3x3).

    Computed via finite differences for robustness.
    """
    eps = 1e-6  # km
    J = np.zeros((3, 3))
    a0 = _j2_accel(pos)
    for j in range(3):
        pos_p = pos.copy()
        pos_p[j] += eps
        J[:, j] = (_j2_accel(pos_p) - a0) / eps
    return J


def _total_accel(pos: np.ndarray, model: DynamicsModel) -> np.ndarray:
    """Total acceleration for given dynamics model."""
    a = _two_body_accel(pos)
    if model in (DynamicsModel.TWO_BODY_J2,):
        a += _j2_accel(pos)
    return a


def _total_jacobian(pos: np.ndarray, model: DynamicsModel) -> np.ndarray:
    """Total Jacobian da/dr for given dynamics model."""
    G = _two_body_jacobian(pos)
    if model in (DynamicsModel.TWO_BODY_J2,):
        G += _j2_jacobian(pos)
    return G


# ---------------------------------------------------------------------------
# ODE for state + STM (42 elements)
# ---------------------------------------------------------------------------

def _state_stm_ode(t: float, y: np.ndarray, model: DynamicsModel) -> np.ndarray:
    """Combined ODE for state (6) + STM (36) = 42 elements."""
    pos = y[:3]
    vel = y[3:6]
    Phi = y[6:42].reshape(6, 6)

    accel = _total_accel(pos, model)
    G = _total_jacobian(pos, model)

    # State derivative
    dstate = np.concatenate([vel, accel])

    # STM derivative: dPhi/dt = A @ Phi
    A = np.zeros((6, 6))
    A[:3, 3:6] = np.eye(3)
    A[3:6, :3] = G
    dPhi = (A @ Phi).flatten()

    return np.concatenate([dstate, dPhi])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def propagate_one_step(
    state: np.ndarray,
    t0: float,
    dt: float,
    model: DynamicsModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a single object one timestep.

    Returns:
        (new_state (6,), STM (6,6)) for the interval [t0, t0+dt].
    """
    # Initial condition: state + identity STM
    y0 = np.concatenate([state, np.eye(6).flatten()])

    sol = solve_ivp(
        _state_stm_ode,
        (t0, t0 + dt),
        y0,
        args=(model,),
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"Propagation failed: {sol.message}")

    yf = sol.y[:, -1]
    new_state = yf[:6]
    stm = yf[6:42].reshape(6, 6)
    return new_state, stm


def estimate_tca(
    state1: np.ndarray,
    state2: np.ndarray,
    t0: float,
    dt_search: float,
    model: DynamicsModel,
    n_steps: int = 100,
) -> float:
    """Estimate time to closest approach via brute-force search.

    Propagates both objects forward in small steps and finds
    the time of minimum relative distance.

    Returns:
        Estimated TCA in seconds from t0.
    """
    dt_sub = dt_search / n_steps
    s1 = state1.copy()
    s2 = state2.copy()
    min_dist = np.linalg.norm(s1[:3] - s2[:3])
    tca = 0.0

    for i in range(1, n_steps + 1):
        # Simple two-body propagation for TCA search (fast)
        s1_new, _ = propagate_one_step(s1, t0 + (i - 1) * dt_sub, dt_sub, model)
        s2_new, _ = propagate_one_step(s2, t0 + (i - 1) * dt_sub, dt_sub, model)
        dist = np.linalg.norm(s1_new[:3] - s2_new[:3])
        if dist < min_dist:
            min_dist = dist
            tca = i * dt_sub
        s1 = s1_new
        s2 = s2_new

    return tca


def propagate_step(
    state_obj1: np.ndarray,
    state_obj2: np.ndarray,
    t: float,
    dt: float,
    model: DynamicsModel,
    tca_search_window: float = 86400.0,
) -> PropagationResult:
    """Propagate both objects one timestep and compute encounter info.

    Args:
        state_obj1: (6,) state of object 1
        state_obj2: (6,) state of object 2
        t: current epoch (seconds)
        dt: timestep (seconds)
        model: dynamics model
        tca_search_window: forward window for TCA search (seconds)

    Returns:
        PropagationResult with new states, STMs, and relative motion.
    """
    new_s1, stm1 = propagate_one_step(state_obj1, t, dt, model)
    new_s2, stm2 = propagate_one_step(state_obj2, t, dt, model)

    rel_pos = new_s2[:3] - new_s1[:3]
    rel_vel = new_s2[3:6] - new_s1[3:6]

    # Rough TCA estimate from current relative motion
    r_rel = np.linalg.norm(rel_pos)
    v_rel = np.linalg.norm(rel_vel)
    # Linear approximation: t_tca ~ -dot(r,v)/dot(v,v)
    rdotv = np.dot(rel_pos, rel_vel)
    vdotv = np.dot(rel_vel, rel_vel)
    if vdotv > 0:
        t_tca_linear = max(-rdotv / vdotv, 0.0)
    else:
        t_tca_linear = tca_search_window

    return PropagationResult(
        state_obj1=StateVector.from_array(new_s1),
        state_obj2=StateVector.from_array(new_s2),
        rel_position=rel_pos,
        rel_velocity=rel_vel,
        stm_obj1=stm1,
        stm_obj2=stm2,
        estimated_tca=t_tca_linear,
    )
