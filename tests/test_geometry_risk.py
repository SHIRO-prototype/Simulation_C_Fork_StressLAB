"""Tests for geometry metrics and risk model."""

import numpy as np
import pytest

from stresslab.modules.geometry_metrics import (
    compute_geometry,
    project_covariance_to_bplane,
)
from stresslab.modules.risk_model import (
    compute_collision_probability,
    compute_risk,
    _pc_foster_2d,
)


class TestGeometry:
    def test_miss_distance(self):
        """Miss distance should match position offset in B-plane."""
        rel_pos = np.array([0.0, 1.0, 0.0])
        rel_vel = np.array([7.5, 0.0, 0.0])  # along x

        geom = compute_geometry(rel_pos, rel_vel, 100.0)

        # Miss distance should be ~1.0 km (perpendicular to velocity)
        assert abs(geom.miss_distance - 1.0) < 0.01

    def test_bplane_orthogonality(self):
        """B-plane vectors should be orthogonal."""
        rel_pos = np.array([100.0, 1.0, 0.5])
        rel_vel = np.array([7.5, 0.1, 0.05])

        geom = compute_geometry(rel_pos, rel_vel, 100.0)

        assert abs(np.dot(geom.b_plane_xi, geom.b_plane_eta)) < 1e-10
        assert abs(np.dot(geom.b_plane_xi, geom.b_plane_zeta)) < 1e-10
        assert abs(np.dot(geom.b_plane_eta, geom.b_plane_zeta)) < 1e-10

    def test_bplane_unit_vectors(self):
        """B-plane vectors should be unit vectors."""
        rel_pos = np.array([100.0, 1.0, 0.5])
        rel_vel = np.array([7.5, 0.1, 0.05])

        geom = compute_geometry(rel_pos, rel_vel, 100.0)

        assert abs(np.linalg.norm(geom.b_plane_xi) - 1.0) < 1e-10
        assert abs(np.linalg.norm(geom.b_plane_eta) - 1.0) < 1e-10
        assert abs(np.linalg.norm(geom.b_plane_zeta) - 1.0) < 1e-10


class TestBplaneProjection:
    def test_projection_reduces_dimension(self):
        """Projected covariance should be 2x2."""
        P_pos = np.eye(3) * 0.01
        eta = np.array([0.0, 1.0, 0.0])
        zeta = np.array([0.0, 0.0, 1.0])

        C_2d = project_covariance_to_bplane(P_pos, eta, zeta)
        assert C_2d.shape == (2, 2)

    def test_projection_positive_definite(self):
        """Projected covariance should be positive definite."""
        P_pos = np.eye(3) * 0.01
        eta = np.array([0.0, 1.0, 0.0])
        zeta = np.array([0.0, 0.0, 1.0])

        C_2d = project_covariance_to_bplane(P_pos, eta, zeta)
        eigvals = np.linalg.eigvalsh(C_2d)
        assert np.all(eigvals > 0)


class TestPcFoster:
    def test_head_on_collision(self):
        """Head-on with zero miss should give highest Pc."""
        C = np.eye(2) * 0.01**2
        pc_zero_miss = _pc_foster_2d(0.0, 0.0, C, 0.02)
        pc_offset = _pc_foster_2d(0.1, 0.0, C, 0.02)
        assert pc_zero_miss > pc_offset

    def test_pc_increases_with_radius(self):
        """Larger hard-body radius should give higher Pc."""
        C = np.eye(2) * 0.01**2
        pc_small = _pc_foster_2d(0.05, 0.0, C, 0.01)
        pc_large = _pc_foster_2d(0.05, 0.0, C, 0.05)
        assert pc_large > pc_small

    def test_pc_bounded(self):
        """Pc should be between 0 and 1."""
        C = np.eye(2) * 0.001**2
        pc = _pc_foster_2d(0.0, 0.0, C, 0.1)
        assert 0.0 <= pc <= 1.0

    def test_pc_decreases_with_miss(self):
        """Pc should decrease as miss distance increases."""
        C = np.eye(2) * 0.01**2
        pcs = []
        for miss in [0.0, 0.01, 0.05, 0.1, 0.5]:
            pcs.append(_pc_foster_2d(miss, 0.0, C, 0.02))
        for i in range(len(pcs) - 1):
            assert pcs[i] >= pcs[i + 1]
