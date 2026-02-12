"""Tests for the covariance engine."""

import numpy as np
import pytest

from stresslab.types import ProcessNoiseConfig
from stresslab.modules.covariance_engine import (
    propagate_covariance,
    measurement_update,
    inject_maneuver,
    covariance_metrics,
    _eci_to_rtn_rotation,
)


class TestRTNRotation:
    def test_orthonormality(self):
        """RTN rotation matrix should be orthonormal."""
        pos = np.array([7000.0, 0.0, 0.0])
        vel = np.array([0.0, 7.5, 0.0])
        R = _eci_to_rtn_rotation(pos, vel)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_radial_direction(self):
        """First row should be radial (along position)."""
        pos = np.array([7000.0, 0.0, 0.0])
        vel = np.array([0.0, 7.5, 0.0])
        R = _eci_to_rtn_rotation(pos, vel)
        r_hat = pos / np.linalg.norm(pos)
        np.testing.assert_allclose(R[0], r_hat, atol=1e-12)


class TestCovariancePropagation:
    def test_covariance_grows(self):
        """Covariance trace should grow with process noise."""
        P = np.eye(6) * 1e-4
        stm = np.eye(6)
        stm[:3, 3:6] = np.eye(3) * 60.0  # dt coupling
        pos = np.array([7000.0, 0.0, 0.0])
        vel = np.array([0.0, 7.5, 0.0])
        pn = ProcessNoiseConfig(scale=1.0)

        P_new = propagate_covariance(P, stm, pos, vel, pn, 60.0)

        assert np.trace(P_new) > np.trace(P)

    def test_symmetry(self):
        """Propagated covariance should be symmetric."""
        P = np.eye(6) * 1e-4
        stm = np.eye(6)
        pos = np.array([7000.0, 0.0, 0.0])
        vel = np.array([0.0, 7.5, 0.0])
        pn = ProcessNoiseConfig()

        P_new = propagate_covariance(P, stm, pos, vel, pn, 60.0)
        np.testing.assert_allclose(P_new, P_new.T, atol=1e-15)

    def test_positive_definite(self):
        """Propagated covariance should remain positive definite."""
        P = np.eye(6) * 1e-4
        stm = np.eye(6)
        pos = np.array([7000.0, 0.0, 0.0])
        vel = np.array([0.0, 7.5, 0.0])
        pn = ProcessNoiseConfig()

        P_new = propagate_covariance(P, stm, pos, vel, pn, 60.0)
        eigvals = np.linalg.eigvalsh(P_new)
        assert np.all(eigvals > 0)


class TestMeasurementUpdate:
    def test_reduces_uncertainty(self):
        """Measurement update should reduce covariance trace."""
        P = np.eye(6) * 1.0  # large uncertainty
        P_updated = measurement_update(P, noise_sigma_pos=0.01)
        assert np.trace(P_updated) < np.trace(P)

    def test_still_positive_definite(self):
        """Updated covariance should remain positive definite."""
        P = np.eye(6) * 1.0
        P_updated = measurement_update(P, noise_sigma_pos=0.01)
        eigvals = np.linalg.eigvalsh(P_updated)
        assert np.all(eigvals > 0)


class TestManeuverInjection:
    def test_increases_velocity_uncertainty(self):
        """Maneuver injection should increase velocity covariance."""
        P = np.eye(6) * 1e-4
        P_man = inject_maneuver(P, delta_v_sigma=0.001)
        # Velocity block should be larger
        assert np.trace(P_man[3:6, 3:6]) > np.trace(P[3:6, 3:6])
        # Position block unchanged
        np.testing.assert_allclose(P_man[:3, :3], P[:3, :3])


class TestCovarianceMetrics:
    def test_metrics(self):
        """Metrics should be computed correctly."""
        P = np.diag([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        result = covariance_metrics(P, 5.0, 60.0)
        assert abs(result.trace - 6.6) < 1e-10
        assert result.max_eigenvalue == 3.0
        assert abs(result.growth_rate - (6.6 - 5.0) / 60.0) < 1e-10
