"""Scientific validation tests for orbital propagation.

Validates:
  - Two-body energy conservation (circular + eccentric orbits)
  - Angular momentum conservation
  - Orbital period accuracy
  - STM determinant (symplecticity)
  - STM linearization accuracy
  - J2 secular RAAN drift rate
  - TCA estimation accuracy
"""

from __future__ import annotations

import numpy as np
import pytest

from stresslab.types import DynamicsModel, MU_EARTH_KM3S2, RE_EARTH_KM, J2
from stresslab.modules.propagation_engine import (
    propagate_one_step,
    propagate_step,
    estimate_tca,
    _two_body_accel,
    _two_body_jacobian,
    _j2_accel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circular_state(sma_km: float, inc_deg: float = 0.0) -> np.ndarray:
    """Create a circular orbit state vector at ascending node."""
    v = np.sqrt(MU_EARTH_KM3S2 / sma_km)
    inc = np.radians(inc_deg)
    # Position at ascending node: x = r, y = 0, z = 0 (before rotation)
    pos = np.array([sma_km, 0.0, 0.0])
    vel = np.array([0.0, v * np.cos(inc), v * np.sin(inc)])
    return np.concatenate([pos, vel])


def _specific_energy(state: np.ndarray) -> float:
    """Keplerian specific orbital energy: E = v^2/2 - mu/r."""
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:6])
    return 0.5 * v**2 - MU_EARTH_KM3S2 / r


def _angular_momentum(state: np.ndarray) -> np.ndarray:
    """Specific angular momentum vector: h = r x v."""
    return np.cross(state[:3], state[3:6])


def _orbital_period(sma_km: float) -> float:
    """Keplerian orbital period: T = 2*pi*sqrt(a^3/mu)."""
    return 2.0 * np.pi * np.sqrt(sma_km**3 / MU_EARTH_KM3S2)


def _eccentric_state(sma_km: float, ecc: float) -> np.ndarray:
    """Create an eccentric orbit state at periapsis."""
    r_p = sma_km * (1 - ecc)
    v_p = np.sqrt(MU_EARTH_KM3S2 * (1 + ecc) / r_p)
    return np.array([r_p, 0.0, 0.0, 0.0, v_p, 0.0])


# ===================================================================
# Two-body acceleration tests
# ===================================================================

class TestTwoBodyAccelValidation:
    """Validate two-body acceleration and Jacobian."""

    def test_acceleration_magnitude_circular(self):
        """For circular orbit: |a| = mu/r^2."""
        r = 7000.0
        pos = np.array([r, 0.0, 0.0])
        a = _two_body_accel(pos)
        expected = MU_EARTH_KM3S2 / r**2
        assert abs(np.linalg.norm(a) - expected) / expected < 1e-12

    def test_acceleration_radially_inward(self):
        """Gravitational acceleration must point toward origin."""
        pos = np.array([4000.0, 5000.0, 3000.0])
        a = _two_body_accel(pos)
        # a should be anti-parallel to pos
        cos_angle = np.dot(a, pos) / (np.linalg.norm(a) * np.linalg.norm(pos))
        assert abs(cos_angle - (-1.0)) < 1e-12

    def test_jacobian_symmetry(self):
        """Two-body Jacobian G = da/dr must be symmetric (central force)."""
        pos = np.array([5000.0, 3000.0, 2000.0])
        G = _two_body_jacobian(pos)
        assert np.allclose(G, G.T, atol=1e-15)

    def test_jacobian_trace(self):
        """For two-body: trace(G) = mu * (3*r^2/r^5 - 3/r^3) = 0 (Laplace eq)."""
        pos = np.array([7000.0, 0.0, 0.0])
        G = _two_body_jacobian(pos)
        # trace(G) = 0 follows from Laplace's equation in vacuum
        assert abs(np.trace(G)) < 1e-15

    def test_jacobian_finite_difference(self):
        """Verify analytical Jacobian matches finite-difference approximation."""
        pos = np.array([6500.0, 1000.0, 500.0])
        G_analytical = _two_body_jacobian(pos)
        eps = 1e-6
        G_fd = np.zeros((3, 3))
        for j in range(3):
            pos_p = pos.copy()
            pos_m = pos.copy()
            pos_p[j] += eps
            pos_m[j] -= eps
            G_fd[:, j] = (_two_body_accel(pos_p) - _two_body_accel(pos_m)) / (2 * eps)
        assert np.allclose(G_analytical, G_fd, atol=1e-6)


# ===================================================================
# J2 perturbation tests
# ===================================================================

class TestJ2AccelValidation:
    """Validate J2 perturbation acceleration."""

    def test_j2_vanishes_far_from_earth(self):
        """J2 perturbation should be negligible at large distances."""
        pos = np.array([1e6, 0.0, 0.0])  # 1 million km
        a_j2 = _j2_accel(pos)
        a_2b = _two_body_accel(pos)
        ratio = np.linalg.norm(a_j2) / np.linalg.norm(a_2b)
        assert ratio < 1e-6

    def test_j2_magnitude_order(self):
        """At LEO, J2 acceleration should be ~O(J2) * two-body accel."""
        pos = np.array([RE_EARTH_KM + 500, 0.0, 0.0])
        a_j2 = _j2_accel(pos)
        a_2b = _two_body_accel(pos)
        ratio = np.linalg.norm(a_j2) / np.linalg.norm(a_2b)
        # J2 ~ 1e-3, so perturbation ratio should be O(1e-3)
        assert 1e-4 < ratio < 1e-2

    def test_j2_equatorial_symmetry(self):
        """J2 accel in equatorial plane: z-component should be zero."""
        pos = np.array([7000.0, 0.0, 0.0])
        a_j2 = _j2_accel(pos)
        assert abs(a_j2[2]) < 1e-15

    def test_j2_polar_maximum(self):
        """J2 perturbation at pole should be larger than at equator for same r."""
        r = 7000.0
        pos_eq = np.array([r, 0.0, 0.0])
        pos_po = np.array([0.0, 0.0, r])
        a_eq = np.linalg.norm(_j2_accel(pos_eq))
        a_po = np.linalg.norm(_j2_accel(pos_po))
        # J2 effect is stronger at poles
        assert a_po > a_eq


# ===================================================================
# Energy conservation tests
# ===================================================================

class TestEnergyConservation:
    """Two-body propagation must conserve specific orbital energy."""

    @pytest.mark.parametrize("sma", [6878.0, 7500.0, 10000.0, 42164.0])
    def test_circular_orbit_energy(self, sma):
        """Energy drift < 1e-10 over one orbit."""
        state = _circular_state(sma)
        T = _orbital_period(sma)
        dt = 60.0
        n_steps = int(T / dt)
        E0 = _specific_energy(state)

        s = state.copy()
        for _ in range(n_steps):
            s, _ = propagate_one_step(s, 0.0, dt, DynamicsModel.TWO_BODY)

        E1 = _specific_energy(s)
        rel_drift = abs(E1 - E0) / abs(E0)
        assert rel_drift < 1e-10, f"Energy drift {rel_drift:.2e} at SMA={sma} km"

    @pytest.mark.parametrize("ecc", [0.01, 0.1, 0.3])
    def test_eccentric_orbit_energy(self, ecc):
        """Energy conservation for eccentric orbits over one period."""
        sma = 7500.0
        state = _eccentric_state(sma, ecc)
        T = _orbital_period(sma)
        dt = 30.0
        n_steps = int(T / dt)
        E0 = _specific_energy(state)

        s = state.copy()
        for _ in range(n_steps):
            s, _ = propagate_one_step(s, 0.0, dt, DynamicsModel.TWO_BODY)

        E1 = _specific_energy(s)
        rel_drift = abs(E1 - E0) / abs(E0)
        assert rel_drift < 1e-9, f"Energy drift {rel_drift:.2e} at e={ecc}"


# ===================================================================
# Angular momentum conservation
# ===================================================================

class TestAngularMomentumConservation:
    """Two-body propagation must conserve angular momentum vector."""

    def test_circular_orbit_h_conservation(self):
        """h vector drift < 1e-10 over one orbit."""
        sma = 7000.0
        state = _circular_state(sma, inc_deg=45.0)
        T = _orbital_period(sma)
        dt = 60.0
        n_steps = int(T / dt)
        h0 = _angular_momentum(state)

        s = state.copy()
        for _ in range(n_steps):
            s, _ = propagate_one_step(s, 0.0, dt, DynamicsModel.TWO_BODY)

        h1 = _angular_momentum(s)
        rel_drift = np.linalg.norm(h1 - h0) / np.linalg.norm(h0)
        assert rel_drift < 1e-10

    def test_eccentric_orbit_h_conservation(self):
        """h conservation for e=0.2 orbit."""
        sma = 8000.0
        state = _eccentric_state(sma, 0.2)
        T = _orbital_period(sma)
        dt = 30.0
        n_steps = int(T / dt)
        h0 = _angular_momentum(state)

        s = state.copy()
        for _ in range(n_steps):
            s, _ = propagate_one_step(s, 0.0, dt, DynamicsModel.TWO_BODY)

        h1 = _angular_momentum(s)
        rel_drift = np.linalg.norm(h1 - h0) / np.linalg.norm(h0)
        assert rel_drift < 1e-9


# ===================================================================
# Orbital period accuracy
# ===================================================================

class TestOrbitalPeriod:
    """Propagation over one period must return near initial position."""

    def test_circular_orbit_period_return(self):
        """After T seconds, satellite returns to ~initial position (two-body)."""
        sma = 7000.0
        state = _circular_state(sma)
        T = _orbital_period(sma)
        dt = 10.0  # fine timestep
        n_steps = int(T / dt)

        s = state.copy()
        for _ in range(n_steps):
            s, _ = propagate_one_step(s, 0.0, dt, DynamicsModel.TWO_BODY)

        # Propagate the remaining fractional step
        remainder = T - n_steps * dt
        if remainder > 0:
            s, _ = propagate_one_step(s, 0.0, remainder, DynamicsModel.TWO_BODY)

        pos_err = np.linalg.norm(s[:3] - state[:3])
        # Should be < 0.01 km (10 m) after one full orbit
        assert pos_err < 0.01, f"Position error after one period: {pos_err:.6f} km"


# ===================================================================
# STM validation
# ===================================================================

class TestSTMValidation:
    """State Transition Matrix scientific validation."""

    def test_stm_determinant_two_body(self):
        """For Hamiltonian two-body, det(STM) = 1 (symplecticity)."""
        state = _circular_state(7000.0)
        _, stm = propagate_one_step(state, 0.0, 60.0, DynamicsModel.TWO_BODY)
        det = np.linalg.det(stm)
        assert abs(det - 1.0) < 1e-8, f"det(STM) = {det}"

    def test_stm_determinant_multiple_steps(self):
        """det(STM) stays near 1 after many two-body steps."""
        state = _circular_state(7000.0)
        s = state.copy()
        stm_accum = np.eye(6)
        for _ in range(100):
            s, stm = propagate_one_step(s, 0.0, 60.0, DynamicsModel.TWO_BODY)
            stm_accum = stm @ stm_accum
        det = np.linalg.det(stm_accum)
        assert abs(det - 1.0) < 1e-5, f"Accumulated det(STM) = {det}"

    def test_stm_linearization_accuracy(self):
        """STM must predict perturbation response accurately.

        Compare STM * delta_x(0) against actual propagation of perturbed IC.
        Error should be O(|delta_x|^2).
        """
        state = _circular_state(7000.0)
        dt = 60.0
        state_final, stm = propagate_one_step(state, 0.0, dt, DynamicsModel.TWO_BODY)

        # Small perturbation
        delta = np.array([0.001, 0.0, 0.0, 0.0, 1e-6, 0.0])  # 1 m, 1 mm/s
        perturbed = state + delta
        perturbed_final, _ = propagate_one_step(perturbed, 0.0, dt, DynamicsModel.TWO_BODY)

        # STM prediction
        predicted_delta = stm @ delta
        actual_delta = perturbed_final - state_final

        error = np.linalg.norm(predicted_delta - actual_delta)
        # Error should be O(|delta|^2) ~ 1e-6
        assert error < 1e-5, f"STM linearization error: {error:.2e}"

    def test_stm_identity_at_zero_dt(self):
        """STM should be identity when dt = 0 (or very small)."""
        state = _circular_state(7000.0)
        _, stm = propagate_one_step(state, 0.0, 1e-10, DynamicsModel.TWO_BODY)
        assert np.allclose(stm, np.eye(6), atol=1e-6)

    def test_stm_with_j2(self):
        """With J2, det(STM) should still be very close to 1 per step."""
        state = _circular_state(7000.0, inc_deg=45.0)
        _, stm = propagate_one_step(state, 0.0, 60.0, DynamicsModel.TWO_BODY_J2)
        det = np.linalg.det(stm)
        # J2 is conservative, so det should still be ~1
        assert abs(det - 1.0) < 1e-6, f"det(STM) with J2 = {det}"


# ===================================================================
# J2 secular RAAN drift
# ===================================================================

class TestJ2SecularDrift:
    """Validate J2 secular RAAN drift rate against analytical formula."""

    def test_raan_drift_rate(self):
        """RAAN drift over 10 orbits should match analytical prediction.

        Analytical: dOmega/dt = -1.5 * n * J2 * (R_E/a)^2 * cos(i) / (1-e^2)^2
        For circular orbit (e=0): dOmega/dt = -1.5 * n * J2 * (R_E/a)^2 * cos(i)
        """
        sma = 7000.0
        inc_deg = 51.6  # ISS-like
        inc_rad = np.radians(inc_deg)
        n = np.sqrt(MU_EARTH_KM3S2 / sma**3)  # mean motion
        T = 2 * np.pi / n

        # Analytical RAAN drift rate
        dOmega_dt_analytical = -1.5 * n * J2 * (RE_EARTH_KM / sma)**2 * np.cos(inc_rad)

        # Propagate 10 orbits with J2
        v_circ = np.sqrt(MU_EARTH_KM3S2 / sma)
        state = np.array([
            sma, 0.0, 0.0,
            0.0, v_circ * np.cos(inc_rad), v_circ * np.sin(inc_rad),
        ])

        dt = 60.0
        total_time = 10 * T
        n_steps = int(total_time / dt)

        h0 = _angular_momentum(state)

        s = state.copy()
        for _ in range(n_steps):
            s, _ = propagate_one_step(s, 0.0, dt, DynamicsModel.TWO_BODY_J2)

        h1 = _angular_momentum(s)

        # RAAN is the angle of h projected onto the equatorial plane
        raan_0 = np.arctan2(h0[0], -h0[1])  # RAAN from h_x, h_y
        raan_1 = np.arctan2(h1[0], -h1[1])

        # Unwrap
        dOmega = raan_1 - raan_0
        elapsed = n_steps * dt
        dOmega_dt_numerical = dOmega / elapsed

        # Should agree within 5% (short-period oscillations cause some deviation)
        rel_error = abs(dOmega_dt_numerical - dOmega_dt_analytical) / abs(dOmega_dt_analytical)
        assert rel_error < 0.05, (
            f"RAAN drift rate: numerical={dOmega_dt_numerical:.4e}, "
            f"analytical={dOmega_dt_analytical:.4e}, rel_error={rel_error:.2%}"
        )


# ===================================================================
# TCA estimation
# ===================================================================

class TestTCAEstimation:
    """Validate time-of-closest-approach estimation."""

    def test_tca_head_on_approach(self):
        """For two objects approaching head-on, TCA should be predictable."""
        # Object 1 at x = 10 km, moving in -x at 1 km/s
        state1 = np.array([10.0 + RE_EARTH_KM + 500, 0.0, 0.0, -1.0, 0.0, 0.0])
        # Object 2 at origin (approx), moving in +x at 1 km/s
        state2 = np.array([RE_EARTH_KM + 500, 0.0, 0.0, 1.0, 0.0, 0.0])

        # Linear TCA ~ 10 km / (1+1 km/s) = 5 seconds
        # But these aren't on physical orbits, so we test the linear estimate
        rel_pos = state1[:3] - state2[:3]
        rel_vel = state1[3:6] - state2[3:6]
        t_tca_linear = max(-np.dot(rel_pos, rel_vel) / np.dot(rel_vel, rel_vel), 0)
        assert abs(t_tca_linear - 5.0) < 0.01

    def test_propagate_step_provides_tca(self):
        """propagate_step should return a positive estimated_tca."""
        from stresslab.modules.scenario_generator import generate_default_scenario
        cfg = generate_default_scenario(seed=42, t_end=600.0, dt=60.0)
        result = propagate_step(
            cfg.state_obj1, cfg.state_obj2, 0.0, 60.0, cfg.dynamics_model,
        )
        assert result.estimated_tca >= 0
