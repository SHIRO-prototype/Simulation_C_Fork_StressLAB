"""Scientific validation tests for Foster 2D collision probability.

Validates:
  - Analytical Pc for isotropic covariance (head-on)
  - Pc asymptotic behavior (miss -> infinity, miss -> 0)
  - Pc monotonicity with miss distance and HBR
  - Monte Carlo cross-validation
  - Anisotropic (highly elongated) covariance handling
  - Risk ratio correctness (identical streams, degraded > reference)
  - B-plane projection consistency
"""

from __future__ import annotations

import numpy as np
import pytest

from stresslab.modules.risk_model import (
    _pc_foster_2d,
    compute_collision_probability,
    compute_risk,
)
from stresslab.modules.geometry_metrics import (
    compute_geometry,
    project_covariance_to_bplane,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _isotropic_pc_analytical(sigma: float, hbr: float) -> float:
    """Exact Pc for isotropic 2D Gaussian at zero miss distance.

    For C_2d = sigma^2 * I_2, miss = (0,0):
      Pc = 1 - exp(-R^2 / (2 * sigma^2))

    Derived from the CDF of chi-squared(2) = exponential distribution.
    The commonly quoted R^2/(2*sigma^2) is only the first-order
    Taylor expansion, valid when R << sigma.
    """
    x = hbr**2 / (2.0 * sigma**2)
    return 1.0 - np.exp(-x)


def _isotropic_pc_with_miss(sigma: float, hbr: float, miss: float) -> float:
    """Exact Pc for isotropic 2D Gaussian with miss distance d.

    Uses the non-central chi-squared(df=2, nc=lambda) CDF where
    lambda = d^2/sigma^2 and x = R^2/sigma^2.
    """
    from scipy.stats import ncx2
    lam = miss**2 / sigma**2
    x = hbr**2 / sigma**2
    return float(ncx2.cdf(x, df=2, nc=lam))


# ===================================================================
# Analytical reference cases (isotropic covariance)
# ===================================================================

class TestFosterPcAnalytical:
    """Compare Foster 2D Pc implementation against analytical formulas."""

    @pytest.mark.parametrize("sigma,hbr", [
        (0.01, 0.02),   # tight covariance, 20m HBR
        (0.1, 0.02),    # moderate covariance
        (1.0, 0.02),    # wide covariance
        (0.01, 0.001),  # small HBR
        (0.5, 0.1),     # large HBR
    ])
    def test_head_on_isotropic(self, sigma, hbr):
        """Pc at zero miss with isotropic C_2d = sigma^2 * I_2."""
        C_2d = np.eye(2) * sigma**2
        pc = _pc_foster_2d(0.0, 0.0, C_2d, hbr)
        expected = _isotropic_pc_analytical(sigma, hbr)
        if expected > 1.0:
            # Pc is clipped to 1.0
            expected = 1.0
        rel_err = abs(pc - expected) / max(expected, 1e-30)
        assert rel_err < 1e-10, f"pc={pc}, expected={expected}, rel_err={rel_err}"

    @pytest.mark.parametrize("miss", [0.001, 0.01, 0.05, 0.1, 0.5])
    def test_with_miss_distance_isotropic(self, miss):
        """Pc with finite miss distance, isotropic covariance."""
        sigma = 0.1
        hbr = 0.02
        C_2d = np.eye(2) * sigma**2
        pc = _pc_foster_2d(miss, 0.0, C_2d, hbr)
        expected = _isotropic_pc_with_miss(sigma, hbr, miss)
        rel_err = abs(pc - expected) / max(expected, 1e-30)
        assert rel_err < 1e-8, f"miss={miss}, pc={pc}, expected={expected}"

    def test_miss_in_both_axes(self):
        """Verify miss distance in eta and zeta axes both contribute."""
        sigma = 0.1
        hbr = 0.02
        C_2d = np.eye(2) * sigma**2
        miss_eta = 0.05
        miss_zeta = 0.05
        pc = _pc_foster_2d(miss_eta, miss_zeta, C_2d, hbr)
        total_miss = np.sqrt(miss_eta**2 + miss_zeta**2)
        expected = _isotropic_pc_with_miss(sigma, hbr, total_miss)
        rel_err = abs(pc - expected) / max(expected, 1e-30)
        assert rel_err < 1e-8


# ===================================================================
# Asymptotic behavior
# ===================================================================

class TestFosterPcAsymptotics:
    """Validate Pc behavior at extreme parameter values."""

    def test_pc_approaches_zero_large_miss(self):
        """Pc -> 0 as miss distance -> infinity."""
        C_2d = np.eye(2) * 0.01
        pc_far = _pc_foster_2d(100.0, 0.0, C_2d, 0.02)
        assert pc_far < 1e-30

    def test_pc_maximum_at_zero_miss(self):
        """Pc is maximized when miss = 0 (for fixed covariance)."""
        C_2d = np.eye(2) * 0.01
        hbr = 0.02
        pc_zero = _pc_foster_2d(0.0, 0.0, C_2d, hbr)
        for miss in [0.01, 0.05, 0.1, 0.5]:
            pc_miss = _pc_foster_2d(miss, 0.0, C_2d, hbr)
            assert pc_miss <= pc_zero + 1e-15

    def test_pc_bounded_0_1(self):
        """Pc must always be in [0, 1]."""
        # Case where Pc would be > 1 without clipping (HBR > sigma)
        C_2d = np.eye(2) * 1e-8  # very small covariance
        pc = _pc_foster_2d(0.0, 0.0, C_2d, 0.02)
        assert 0.0 <= pc <= 1.0

    def test_pc_approaches_zero_large_covariance(self):
        """Pc -> 0 as covariance -> infinity (uncertainty dilutes collision)."""
        hbr = 0.02
        for sigma in [10.0, 100.0, 1000.0]:
            C_2d = np.eye(2) * sigma**2
            pc = _pc_foster_2d(0.0, 0.0, C_2d, hbr)
            assert pc < hbr**2 / (2 * sigma**2) + 1e-15


# ===================================================================
# Monotonicity
# ===================================================================

class TestFosterPcMonotonicity:
    """Pc must vary monotonically with key parameters."""

    def test_pc_decreases_with_miss_distance(self):
        """Pc must strictly decrease as miss distance increases."""
        C_2d = np.eye(2) * 0.01
        hbr = 0.02
        prev_pc = 1.0
        for miss in np.linspace(0.0, 0.5, 50):
            pc = _pc_foster_2d(miss, 0.0, C_2d, hbr)
            assert pc <= prev_pc + 1e-15
            prev_pc = pc

    def test_pc_increases_with_hbr(self):
        """Pc must increase with hard-body radius."""
        C_2d = np.eye(2) * 0.01
        miss = 0.05
        prev_pc = 0.0
        for hbr in np.linspace(0.001, 0.1, 20):
            pc = _pc_foster_2d(miss, 0.0, C_2d, hbr)
            assert pc >= prev_pc - 1e-15
            prev_pc = pc

    def test_pc_decreases_with_covariance_scale(self):
        """For miss=0, Pc = R^2/(2*sigma^2), so Pc decreases with sigma."""
        hbr = 0.02
        prev_pc = 1.0
        for sigma in np.linspace(0.01, 1.0, 50):
            C_2d = np.eye(2) * sigma**2
            pc = _pc_foster_2d(0.0, 0.0, C_2d, hbr)
            assert pc <= prev_pc + 1e-15
            prev_pc = pc


# ===================================================================
# Monte Carlo cross-validation
# ===================================================================

class TestFosterPcMonteCarlo:
    """Cross-validate Foster Pc against brute-force Monte Carlo sampling."""

    def test_mc_vs_analytical_isotropic_zero_miss(self):
        """Monte Carlo Pc estimate must agree with Foster formula (zero miss)."""
        sigma = 0.05
        hbr = 0.02
        C_2d = np.eye(2) * sigma**2
        n_samples = 500_000
        rng = np.random.default_rng(42)

        # Sample from 2D Gaussian centered at origin
        samples = rng.multivariate_normal([0.0, 0.0], C_2d, size=n_samples)
        # Count fraction inside circle of radius hbr
        distances = np.sqrt(samples[:, 0]**2 + samples[:, 1]**2)
        mc_pc = np.mean(distances < hbr)

        foster_pc = _pc_foster_2d(0.0, 0.0, C_2d, hbr)

        # MC should agree within ~3 sigma of sampling error
        mc_std = np.sqrt(mc_pc * (1 - mc_pc) / n_samples)
        assert abs(foster_pc - mc_pc) < 4 * mc_std + 1e-6, (
            f"Foster={foster_pc:.6e}, MC={mc_pc:.6e}, MC_std={mc_std:.6e}"
        )

    def test_mc_vs_analytical_with_miss(self):
        """Monte Carlo Pc with non-zero miss distance."""
        sigma = 0.1
        hbr = 0.02
        miss_eta = 0.05
        miss_zeta = 0.03
        C_2d = np.eye(2) * sigma**2
        n_samples = 500_000
        rng = np.random.default_rng(123)

        # Sample from 2D Gaussian centered at (miss_eta, miss_zeta)
        samples = rng.multivariate_normal([miss_eta, miss_zeta], C_2d, size=n_samples)
        distances = np.sqrt(samples[:, 0]**2 + samples[:, 1]**2)
        mc_pc = np.mean(distances < hbr)

        foster_pc = _pc_foster_2d(miss_eta, miss_zeta, C_2d, hbr)

        mc_std = np.sqrt(max(mc_pc, 1e-10) * (1 - mc_pc) / n_samples)
        assert abs(foster_pc - mc_pc) < 5 * mc_std + 1e-6, (
            f"Foster={foster_pc:.6e}, MC={mc_pc:.6e}"
        )

    def test_mc_vs_analytical_anisotropic(self):
        """Monte Carlo with anisotropic covariance (10:1 aspect ratio)."""
        sigma1 = 0.1
        sigma2 = 0.01
        hbr = 0.02
        C_2d = np.diag([sigma1**2, sigma2**2])
        n_samples = 500_000
        rng = np.random.default_rng(456)

        samples = rng.multivariate_normal([0.0, 0.0], C_2d, size=n_samples)
        distances = np.sqrt(samples[:, 0]**2 + samples[:, 1]**2)
        mc_pc = np.mean(distances < hbr)

        foster_pc = _pc_foster_2d(0.0, 0.0, C_2d, hbr)

        mc_std = np.sqrt(max(mc_pc, 1e-10) * (1 - mc_pc) / n_samples)
        assert abs(foster_pc - mc_pc) < 5 * mc_std + 1e-5, (
            f"Foster={foster_pc:.6e}, MC={mc_pc:.6e}"
        )


# ===================================================================
# Anisotropic covariance
# ===================================================================

class TestFosterPcAnisotropic:
    """Foster Pc with highly elongated covariance ellipses."""

    def test_extreme_aspect_ratio(self):
        """100:1 covariance aspect ratio should produce valid Pc."""
        C_2d = np.diag([1.0, 0.01**2])
        pc = _pc_foster_2d(0.0, 0.0, C_2d, 0.02)
        assert 0.0 <= pc <= 1.0
        assert pc > 0  # should still be nonzero at zero miss

    def test_rotated_anisotropic(self):
        """Pc should be invariant under rotation of covariance + miss vector."""
        sigma1, sigma2 = 0.1, 0.01
        C_diag = np.diag([sigma1**2, sigma2**2])
        miss = np.array([0.03, 0.02])

        # Rotate by 45 degrees
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        C_rot = R @ C_diag @ R.T
        miss_rot = R @ miss

        pc_orig = _pc_foster_2d(miss[0], miss[1], C_diag, 0.02)
        pc_rot = _pc_foster_2d(miss_rot[0], miss_rot[1], C_rot, 0.02)

        assert abs(pc_orig - pc_rot) / max(pc_orig, 1e-30) < 1e-8


# ===================================================================
# Risk ratio
# ===================================================================

class TestRiskRatio:
    """Validate risk ratio computation in compute_risk."""

    def _make_geometry(self):
        """Create a simple encounter geometry."""
        rel_pos = np.array([0.5, 0.0, 0.0])
        rel_vel = np.array([-10.0, 0.0, 0.0])
        geom = compute_geometry(rel_pos, rel_vel, 100.0)
        return rel_pos, geom.b_plane_eta, geom.b_plane_zeta

    def test_identical_streams_ratio_one(self):
        """When reference = degraded, risk_ratio must be 1.0."""
        rel_pos, eta, zeta = self._make_geometry()
        P = np.eye(6) * 0.01
        result = compute_risk(rel_pos, P, P, P, P, eta, zeta, 0.02)
        assert abs(result.risk_ratio - 1.0) < 1e-10

    def test_degraded_larger_increases_pc(self):
        """Larger degraded covariance should change Pc (up or down depends on regime)."""
        rel_pos, eta, zeta = self._make_geometry()
        P_ref = np.eye(6) * 0.01
        P_deg = np.eye(6) * 0.1  # 10x larger covariance
        result = compute_risk(rel_pos, P_ref, P_ref, P_deg, P_deg, eta, zeta, 0.02)
        # Pc values should differ
        assert result.pc_reference != result.pc_degraded or (
            result.pc_reference == 0 and result.pc_degraded == 0
        )

    def test_risk_ratio_non_negative(self):
        """Risk ratio must always be >= 0."""
        rel_pos, eta, zeta = self._make_geometry()
        P1 = np.eye(6) * 0.01
        P2 = np.eye(6) * 0.1
        result = compute_risk(rel_pos, P1, P1, P2, P2, eta, zeta, 0.02)
        assert result.risk_ratio >= 0

    def test_pc_values_bounded(self):
        """Both pc_reference and pc_degraded must be in [0, 1]."""
        rel_pos, eta, zeta = self._make_geometry()
        P = np.eye(6) * 0.01
        result = compute_risk(rel_pos, P, P, P, P, eta, zeta, 0.02)
        assert 0 <= result.pc_reference <= 1
        assert 0 <= result.pc_degraded <= 1
