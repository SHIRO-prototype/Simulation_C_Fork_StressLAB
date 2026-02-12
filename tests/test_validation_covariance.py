"""Scientific validation tests for covariance propagation.

Validates:
  - RTN rotation matrix orthonormality and determinant for diverse geometries
  - Process noise Q-matrix structure, symmetry, positive semi-definiteness
  - Process noise time-scaling (dt^4 position, dt^2 velocity)
  - Covariance positive-definiteness preservation over long propagation
  - Joseph-form measurement update stability
  - Measurement update information reduction
  - Maneuver injection additivity
  - Steady-state Kalman filter convergence
"""

from __future__ import annotations

import numpy as np
import pytest

from stresslab.types import (
    MU_EARTH_KM3S2, RE_EARTH_KM,
    ProcessNoiseConfig, SimulationConfig, DynamicsModel,
)
from stresslab.modules.covariance_engine import (
    _eci_to_rtn_rotation,
    _process_noise_eci,
    measurement_update,
    inject_maneuver,
    propagate_covariance,
    covariance_metrics,
)
from stresslab.modules.propagation_engine import propagate_one_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circular_state(sma_km: float, inc_deg: float = 0.0) -> np.ndarray:
    v = np.sqrt(MU_EARTH_KM3S2 / sma_km)
    inc = np.radians(inc_deg)
    return np.array([sma_km, 0.0, 0.0, 0.0, v * np.cos(inc), v * np.sin(inc)])


def _random_orbit_state(rng: np.random.Generator) -> np.ndarray:
    """Generate a physically valid random orbit state."""
    sma = rng.uniform(6600, 42000)
    inc = rng.uniform(0, 180)
    arg_lat = rng.uniform(0, 360)
    v = np.sqrt(MU_EARTH_KM3S2 / sma)
    lat = np.radians(arg_lat)
    inc_r = np.radians(inc)
    pos = sma * np.array([np.cos(lat), np.sin(lat) * np.cos(inc_r), np.sin(lat) * np.sin(inc_r)])
    vel = v * np.array([-np.sin(lat), np.cos(lat) * np.cos(inc_r), np.cos(lat) * np.sin(inc_r)])
    return np.concatenate([pos, vel])


# ===================================================================
# RTN rotation matrix validation
# ===================================================================

class TestRTNRotationValidation:
    """Exhaustive validation of ECI-to-RTN rotation matrix."""

    @pytest.mark.parametrize("inc_deg", [0.0, 30.0, 51.6, 90.0, 135.0])
    def test_orthonormality_various_inclinations(self, inc_deg):
        """R @ R^T = I_3 for various orbit inclinations."""
        state = _circular_state(7000.0, inc_deg)
        R = _eci_to_rtn_rotation(state[:3], state[3:6])
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-14)

    @pytest.mark.parametrize("inc_deg", [0.0, 30.0, 51.6, 90.0, 135.0])
    def test_determinant_plus_one(self, inc_deg):
        """det(R) = +1 (proper rotation, right-handed)."""
        state = _circular_state(7000.0, inc_deg)
        R = _eci_to_rtn_rotation(state[:3], state[3:6])
        assert abs(np.linalg.det(R) - 1.0) < 1e-14

    def test_fuzz_100_random_orbits(self):
        """RTN rotation is orthonormal for 100 random orbit geometries."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            state = _random_orbit_state(rng)
            R = _eci_to_rtn_rotation(state[:3], state[3:6])
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
            assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_radial_points_outward(self):
        """R-hat (first row) should be parallel to position vector."""
        state = _circular_state(7000.0, 45.0)
        R = _eci_to_rtn_rotation(state[:3], state[3:6])
        r_hat = R[0, :]
        pos_hat = state[:3] / np.linalg.norm(state[:3])
        assert np.allclose(r_hat, pos_hat, atol=1e-14)


# ===================================================================
# Process noise Q-matrix validation
# ===================================================================

class TestProcessNoiseValidation:
    """Validate process noise matrix structure and properties."""

    def test_q_symmetric(self):
        """Q_ECI must be symmetric."""
        state = _circular_state(7000.0, 51.6)
        pn = ProcessNoiseConfig()
        Q = _process_noise_eci(state[:3], state[3:6], pn, 60.0)
        assert np.allclose(Q, Q.T, atol=1e-18)

    def test_q_positive_semidefinite(self):
        """Q_ECI eigenvalues must be >= 0."""
        state = _circular_state(7000.0, 51.6)
        pn = ProcessNoiseConfig()
        Q = _process_noise_eci(state[:3], state[3:6], pn, 60.0)
        eigvals = np.linalg.eigvalsh(Q)
        assert np.all(eigvals >= -1e-18)

    def test_q_shape(self):
        """Q must be 6x6."""
        state = _circular_state(7000.0)
        pn = ProcessNoiseConfig()
        Q = _process_noise_eci(state[:3], state[3:6], pn, 60.0)
        assert Q.shape == (6, 6)

    def test_q_position_block_scales_dt4(self):
        """Position block of Q should scale as dt^4/4.

        Q_pos = Q_3 * dt^4 / 4
        Doubling dt should increase Q_pos by 16x.
        """
        state = _circular_state(7000.0, 51.6)
        pn = ProcessNoiseConfig(sigma_radial=1e-6, sigma_tangential=1e-6,
                                 sigma_normal=1e-6)
        Q1 = _process_noise_eci(state[:3], state[3:6], pn, 60.0)
        Q2 = _process_noise_eci(state[:3], state[3:6], pn, 120.0)

        # Position block is top-left 3x3
        ratio = np.linalg.norm(Q2[:3, :3]) / np.linalg.norm(Q1[:3, :3])
        expected_ratio = (120.0 / 60.0)**4
        assert abs(ratio - expected_ratio) / expected_ratio < 1e-10

    def test_q_velocity_block_scales_dt2(self):
        """Velocity block of Q should scale as dt^2.

        Q_vel = Q_3 * dt^2
        Doubling dt should increase Q_vel by 4x.
        """
        state = _circular_state(7000.0, 51.6)
        pn = ProcessNoiseConfig(sigma_radial=1e-6, sigma_tangential=1e-6,
                                 sigma_normal=1e-6)
        Q1 = _process_noise_eci(state[:3], state[3:6], pn, 60.0)
        Q2 = _process_noise_eci(state[:3], state[3:6], pn, 120.0)

        # Velocity block is bottom-right 3x3
        ratio = np.linalg.norm(Q2[3:6, 3:6]) / np.linalg.norm(Q1[3:6, 3:6])
        expected_ratio = (120.0 / 60.0)**2
        assert abs(ratio - expected_ratio) / expected_ratio < 1e-10

    def test_q_scales_with_process_noise_scale(self):
        """Q should scale linearly with process_noise.scale."""
        state = _circular_state(7000.0, 51.6)
        pn1 = ProcessNoiseConfig(scale=1.0)
        pn2 = ProcessNoiseConfig(scale=3.0)
        Q1 = _process_noise_eci(state[:3], state[3:6], pn1, 60.0)
        Q2 = _process_noise_eci(state[:3], state[3:6], pn2, 60.0)
        assert np.allclose(Q2, 3.0 * Q1, atol=1e-20)

    def test_q_zero_noise_is_zero(self):
        """Zero process noise sigmas should give zero Q."""
        state = _circular_state(7000.0)
        pn = ProcessNoiseConfig(sigma_radial=0.0, sigma_tangential=0.0, sigma_normal=0.0)
        Q = _process_noise_eci(state[:3], state[3:6], pn, 60.0)
        assert np.allclose(Q, 0.0, atol=1e-30)


# ===================================================================
# Covariance positive-definiteness preservation
# ===================================================================

class TestCovariancePDPreservation:
    """Covariance must remain positive-definite through propagation."""

    def test_pd_through_1000_steps(self):
        """Propagate covariance for 1000 steps; eigenvalues must stay positive."""
        state = _circular_state(7000.0, 51.6)
        P = np.eye(6) * 1e-4
        pn = ProcessNoiseConfig()
        dt = 60.0

        s = state.copy()
        for _ in range(1000):
            s_new, stm = propagate_one_step(s, 0.0, dt, DynamicsModel.TWO_BODY_J2)
            P = propagate_covariance(P, stm, s_new[:3], s_new[3:6], pn, dt)
            eigvals = np.linalg.eigvalsh(P)
            assert np.all(eigvals > 0), f"Negative eigenvalue: {eigvals.min()}"
            s = s_new

    def test_pd_with_near_singular_initial(self):
        """Start with near-singular covariance; must stay PD after propagation."""
        state = _circular_state(7000.0)
        # Near-singular: very small eigenvalues
        P = np.eye(6) * 1e-12
        pn = ProcessNoiseConfig()
        dt = 60.0

        s = state.copy()
        for _ in range(100):
            s_new, stm = propagate_one_step(s, 0.0, dt, DynamicsModel.TWO_BODY)
            P = propagate_covariance(P, stm, s_new[:3], s_new[3:6], pn, dt)
            eigvals = np.linalg.eigvalsh(P)
            assert np.all(eigvals > 0)
            s = s_new

    def test_symmetry_preserved(self):
        """Covariance must remain symmetric throughout propagation."""
        state = _circular_state(7000.0, 51.6)
        P = np.eye(6) * 1e-4
        pn = ProcessNoiseConfig()
        dt = 60.0

        s = state.copy()
        for _ in range(100):
            s_new, stm = propagate_one_step(s, 0.0, dt, DynamicsModel.TWO_BODY_J2)
            P = propagate_covariance(P, stm, s_new[:3], s_new[3:6], pn, dt)
            assert np.allclose(P, P.T, atol=1e-18)
            s = s_new


# ===================================================================
# Joseph-form measurement update stability
# ===================================================================

class TestJosephFormStability:
    """Validate Joseph-form Kalman measurement update."""

    def test_update_reduces_position_uncertainty(self):
        """Measurement update must reduce position block trace."""
        P = np.eye(6) * 1.0
        sigma_pos = 0.01
        P_up = measurement_update(P, sigma_pos)
        # Position block trace should decrease
        assert np.trace(P_up[:3, :3]) < np.trace(P[:3, :3])

    def test_update_preserves_pd(self):
        """Updated covariance must be positive-definite."""
        P = np.eye(6) * 1.0
        P_up = measurement_update(P, 0.01)
        eigvals = np.linalg.eigvalsh(P_up)
        assert np.all(eigvals > 0)

    def test_update_preserves_symmetry(self):
        """Updated covariance must be symmetric."""
        P = np.eye(6) * 0.5
        P_up = measurement_update(P, 0.01)
        assert np.allclose(P_up, P_up.T, atol=1e-16)

    def test_joseph_form_vs_naive_form(self):
        """Joseph form should give same result as naive form for well-conditioned P.

        Naive: P_up = (I - KH) @ P
        Joseph: P_up = (I - KH) @ P @ (I - KH)^T + K @ R @ K^T
        """
        P = np.eye(6) * 0.5
        sigma = 0.01
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        R = np.eye(3) * sigma**2
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        IKH = np.eye(6) - K @ H

        naive = IKH @ P
        joseph = IKH @ P @ IKH.T + K @ R @ K.T

        # For well-conditioned case, they should agree closely
        assert np.allclose(naive, joseph, atol=1e-12)

    def test_joseph_form_with_ill_conditioned_P(self):
        """Joseph form should still give PD result with ill-conditioned P."""
        # Ill-conditioned: large spread in eigenvalues
        P = np.diag([100.0, 100.0, 100.0, 1e-6, 1e-6, 1e-6])
        sigma = 0.01
        P_up = measurement_update(P, sigma)
        eigvals = np.linalg.eigvalsh(P_up)
        assert np.all(eigvals > 0), f"Negative eigenvalue: {eigvals.min()}"

    def test_repeated_updates_converge(self):
        """Repeated measurement updates should converge to a lower bound."""
        P = np.eye(6) * 100.0
        sigma = 0.01
        traces = []
        for _ in range(50):
            P = measurement_update(P, sigma)
            traces.append(np.trace(P))

        # Trace should be monotonically decreasing
        for i in range(1, len(traces)):
            assert traces[i] <= traces[i - 1] + 1e-12

        # Should converge (last few traces should be very similar)
        assert abs(traces[-1] - traces[-2]) / traces[-2] < 1e-6


# ===================================================================
# Maneuver injection
# ===================================================================

class TestManeuverInjection:
    """Validate maneuver uncertainty injection."""

    def test_additivity(self):
        """P_after = P_before + Q_maneuver exactly."""
        P = np.eye(6) * 0.1
        dv_sigma = 0.001
        P_after = inject_maneuver(P, dv_sigma)
        Q_man = np.zeros((6, 6))
        Q_man[3, 3] = Q_man[4, 4] = Q_man[5, 5] = dv_sigma**2
        assert np.allclose(P_after, P + Q_man, atol=1e-18)

    def test_position_block_unchanged(self):
        """Maneuver injection should not change position covariance."""
        P = np.eye(6) * 0.1
        P_after = inject_maneuver(P, 0.001)
        assert np.allclose(P_after[:3, :3], P[:3, :3], atol=1e-18)

    def test_velocity_block_increased(self):
        """Velocity diagonal should increase by dv_sigma^2."""
        P = np.eye(6) * 0.1
        dv_sigma = 0.005
        P_after = inject_maneuver(P, dv_sigma)
        for i in range(3, 6):
            assert abs(P_after[i, i] - (P[i, i] + dv_sigma**2)) < 1e-18


# ===================================================================
# Steady-state Kalman filter convergence
# ===================================================================

class TestSteadyStateConvergence:
    """With periodic updates, covariance should converge to steady-state."""

    def test_convergence_with_periodic_updates(self):
        """Propagate + update cycle should converge covariance trace.

        The trace follows a sawtooth pattern (grows during propagation,
        drops at measurement update), so we check convergence of the
        *post-update* traces rather than every step.
        """
        state = _circular_state(7000.0, 51.6)
        P = np.eye(6) * 10.0  # Start with large uncertainty
        pn = ProcessNoiseConfig()
        dt = 60.0
        update_interval = 600.0  # Update every 10 minutes

        post_update_traces = []
        s = state.copy()
        t = 0.0
        last_update = 0.0
        for _ in range(500):
            s_new, stm = propagate_one_step(s, t, dt, DynamicsModel.TWO_BODY_J2)
            P = propagate_covariance(P, stm, s_new[:3], s_new[3:6], pn, dt)
            t += dt

            if t - last_update >= update_interval:
                P = measurement_update(P, 0.01)
                last_update = t
                post_update_traces.append(np.trace(P))

            s = s_new

        # After 500 steps (~8.3 hours) with 10-min updates, we have ~50
        # post-update traces. The last 10 should be within 5% of each other,
        # showing convergence of the steady-state Kalman gain.
        assert len(post_update_traces) >= 10, (
            f"Expected >=10 post-update traces, got {len(post_update_traces)}"
        )
        last_10 = post_update_traces[-10:]
        mean_trace = np.mean(last_10)
        trace_range = max(last_10) - min(last_10)
        assert trace_range < 0.05 * mean_trace, (
            f"Post-update trace not converged: range={trace_range:.4e}, "
            f"mean={mean_trace:.4e}, ratio={trace_range/mean_trace:.4f}"
        )


# ===================================================================
# Covariance metrics
# ===================================================================

class TestCovarianceMetricsValidation:
    """Validate covariance metric computations."""

    def test_trace_correctness(self):
        """Trace metric must equal numpy trace."""
        P = np.diag([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        result = covariance_metrics(P, 0.0, 60.0)
        assert abs(result.trace - np.trace(P)) < 1e-15

    def test_frobenius_correctness(self):
        """Frobenius norm must equal numpy Frobenius norm."""
        P = np.diag([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        result = covariance_metrics(P, 0.0, 60.0)
        assert abs(result.frobenius - np.linalg.norm(P, 'fro')) < 1e-15

    def test_max_eigenvalue(self):
        """Max eigenvalue must equal largest eigenvalue."""
        P = np.diag([1.0, 5.0, 3.0, 0.1, 0.2, 0.3])
        result = covariance_metrics(P, 0.0, 60.0)
        assert abs(result.max_eigenvalue - 5.0) < 1e-12

    def test_growth_rate(self):
        """Growth rate must equal (trace - prev_trace) / dt."""
        P = np.diag([2.0, 2.0, 2.0, 0.1, 0.1, 0.1])
        prev_trace = 5.0
        dt = 60.0
        result = covariance_metrics(P, prev_trace, dt)
        expected = (np.trace(P) - prev_trace) / dt
        assert abs(result.growth_rate - expected) < 1e-15
