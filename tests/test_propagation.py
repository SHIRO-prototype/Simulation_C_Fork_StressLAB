"""Tests for the propagation engine."""

import numpy as np
import pytest

from stresslab.types import DynamicsModel, MU_EARTH_KM3S2, RE_EARTH_KM
from stresslab.modules.propagation_engine import (
    propagate_one_step,
    propagate_step,
    _two_body_accel,
)


class TestTwoBodyAccel:
    def test_magnitude(self):
        """Acceleration magnitude should equal mu/r^2."""
        r = RE_EARTH_KM + 500.0
        pos = np.array([r, 0.0, 0.0])
        accel = _two_body_accel(pos)
        expected_mag = MU_EARTH_KM3S2 / r**2
        assert abs(np.linalg.norm(accel) - expected_mag) < 1e-10

    def test_direction(self):
        """Acceleration should point toward Earth center."""
        pos = np.array([7000.0, 0.0, 0.0])
        accel = _two_body_accel(pos)
        # Should be in -x direction
        assert accel[0] < 0
        assert abs(accel[1]) < 1e-15
        assert abs(accel[2]) < 1e-15


class TestPropagateOneStep:
    def test_circular_orbit_energy(self):
        """Energy should be conserved in two-body propagation."""
        r = RE_EARTH_KM + 500.0
        v = np.sqrt(MU_EARTH_KM3S2 / r)
        state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

        def energy(s):
            r_mag = np.linalg.norm(s[:3])
            v_mag = np.linalg.norm(s[3:6])
            return 0.5 * v_mag**2 - MU_EARTH_KM3S2 / r_mag

        e0 = energy(state)
        new_state, stm = propagate_one_step(state, 0.0, 60.0, DynamicsModel.TWO_BODY)
        e1 = energy(new_state)

        assert abs(e1 - e0) < 1e-10, f"Energy drift: {abs(e1 - e0)}"

    def test_stm_identity_at_zero(self):
        """STM for dt=0 should be near identity."""
        r = RE_EARTH_KM + 500.0
        v = np.sqrt(MU_EARTH_KM3S2 / r)
        state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

        # Very short propagation
        _, stm = propagate_one_step(state, 0.0, 0.001, DynamicsModel.TWO_BODY)
        assert np.allclose(stm, np.eye(6), atol=1e-2)

    def test_stm_determinant(self):
        """STM determinant should be ~1 (symplectic property)."""
        r = RE_EARTH_KM + 500.0
        v = np.sqrt(MU_EARTH_KM3S2 / r)
        state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

        _, stm = propagate_one_step(state, 0.0, 300.0, DynamicsModel.TWO_BODY)
        det = np.linalg.det(stm)
        assert abs(det - 1.0) < 1e-6, f"STM det = {det}"


class TestPropagateStep:
    def test_relative_vectors(self):
        """Relative position/velocity should be obj2 - obj1."""
        r = RE_EARTH_KM + 500.0
        v = np.sqrt(MU_EARTH_KM3S2 / r)
        s1 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])
        s2 = np.array([r + 2.0, 0.0, 0.0, 0.0, v, 0.0])

        result = propagate_step(s1, s2, 0.0, 60.0, DynamicsModel.TWO_BODY)

        expected_rel = result.state_obj2.as_array()[:3] - result.state_obj1.as_array()[:3]
        np.testing.assert_allclose(result.rel_position, expected_rel, atol=1e-12)
