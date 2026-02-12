"""Tests for the metrics contract module."""

import numpy as np
import pytest

from stresslab.metrics_contract import (
    METRICS_CONTRACT_VERSION,
    covariance_norm,
    covariance_growth_rate,
    staleness,
    freshness_score,
    compression_window,
    is_true_danger,
    is_false_safe,
    is_false_alert,
    outage_sensitivity_score,
    MetricsSummary,
)


# ---------------------------------------------------------------------------
# Contract version
# ---------------------------------------------------------------------------

class TestContractVersion:
    def test_version_string(self):
        assert isinstance(METRICS_CONTRACT_VERSION, str)
        parts = METRICS_CONTRACT_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# Covariance metrics
# ---------------------------------------------------------------------------

class TestCovarianceNorm:
    def test_identity(self):
        P = np.eye(6)
        assert covariance_norm(P) == pytest.approx(6.0)

    def test_diagonal(self):
        P = np.diag([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        assert covariance_norm(P) == pytest.approx(6.6)

    def test_zero_matrix(self):
        P = np.zeros((6, 6))
        assert covariance_norm(P) == pytest.approx(0.0)

    def test_returns_float(self):
        P = np.eye(6)
        result = covariance_norm(P)
        assert isinstance(result, float)


class TestCovarianceGrowthRate:
    def test_positive_growth(self):
        rate = covariance_growth_rate(10.0, 5.0, 1.0)
        assert rate == pytest.approx(5.0)

    def test_negative_growth(self):
        # After a measurement update, trace may decrease
        rate = covariance_growth_rate(3.0, 5.0, 2.0)
        assert rate == pytest.approx(-1.0)

    def test_zero_dt_returns_zero(self):
        rate = covariance_growth_rate(10.0, 5.0, 0.0)
        assert rate == pytest.approx(0.0)

    def test_negative_dt_returns_zero(self):
        rate = covariance_growth_rate(10.0, 5.0, -1.0)
        assert rate == pytest.approx(0.0)

    def test_no_change(self):
        rate = covariance_growth_rate(5.0, 5.0, 10.0)
        assert rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Staleness metrics
# ---------------------------------------------------------------------------

class TestStaleness:
    def test_positive_gap(self):
        assert staleness(100.0, 90.0) == pytest.approx(10.0)

    def test_zero_gap(self):
        assert staleness(50.0, 50.0) == pytest.approx(0.0)

    def test_negative_gap_clamped(self):
        # last_update in the future should clamp to 0
        assert staleness(50.0, 60.0) == pytest.approx(0.0)


class TestFreshnessScore:
    def test_perfectly_fresh(self):
        assert freshness_score(0.0) == pytest.approx(1.0)

    def test_maximally_stale(self):
        assert freshness_score(86400.0) == pytest.approx(0.0)

    def test_beyond_reference_clamped(self):
        assert freshness_score(200000.0) == pytest.approx(0.0)

    def test_half_fresh(self):
        assert freshness_score(43200.0) == pytest.approx(0.5)

    def test_custom_reference(self):
        assert freshness_score(500.0, staleness_ref=1000.0) == pytest.approx(0.5)

    def test_bounds(self):
        # Should always be in [0, 1]
        for s in [0.0, 100.0, 86400.0, 1e9]:
            score = freshness_score(s)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Decision timing
# ---------------------------------------------------------------------------

class TestCompressionWindow:
    def test_integrity_triggers_first(self):
        # integrity at t=100, threshold at t=150 => DCW = 50 (positive)
        dcw = compression_window(150.0, 100.0)
        assert dcw == pytest.approx(50.0)

    def test_threshold_triggers_first(self):
        # threshold at t=100, integrity at t=150 => DCW = -50 (negative)
        dcw = compression_window(100.0, 150.0)
        assert dcw == pytest.approx(-50.0)

    def test_simultaneous_triggers(self):
        dcw = compression_window(100.0, 100.0)
        assert dcw == pytest.approx(0.0)

    def test_none_if_threshold_missing(self):
        assert compression_window(None, 100.0) is None

    def test_none_if_integrity_missing(self):
        assert compression_window(100.0, None) is None

    def test_none_if_both_missing(self):
        assert compression_window(None, None) is None


# ---------------------------------------------------------------------------
# Ground truth and error classification
# ---------------------------------------------------------------------------

class TestIsTrueDanger:
    def test_dangerous(self):
        assert is_true_danger(0.005, 0.01) is True

    def test_safe(self):
        assert is_true_danger(0.05, 0.01) is False

    def test_exactly_at_boundary(self):
        # miss == hbr => not strictly less than => safe
        assert is_true_danger(0.01, 0.01) is False


class TestIsFalseSafe:
    def test_predicted_safe_actually_dangerous(self):
        assert is_false_safe(True, 0.005, 0.01) is True

    def test_predicted_safe_actually_safe(self):
        assert is_false_safe(True, 0.05, 0.01) is False

    def test_predicted_alert_actually_dangerous(self):
        # Not a false-safe since predicted alert
        assert is_false_safe(False, 0.005, 0.01) is False


class TestIsFalseAlert:
    def test_predicted_alert_actually_safe(self):
        assert is_false_alert(True, 0.05, 0.01) is True

    def test_predicted_alert_actually_dangerous(self):
        assert is_false_alert(True, 0.005, 0.01) is False

    def test_predicted_safe_actually_safe(self):
        # Not a false-alert since predicted safe
        assert is_false_alert(False, 0.05, 0.01) is False


# ---------------------------------------------------------------------------
# Outage sensitivity
# ---------------------------------------------------------------------------

class TestOutageSensitivity:
    def test_identity(self):
        assert outage_sensitivity_score(3600.0) == pytest.approx(3600.0)

    def test_zero(self):
        assert outage_sensitivity_score(0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MetricsSummary dataclass
# ---------------------------------------------------------------------------

class TestMetricsSummary:
    def test_frozen(self):
        ms = MetricsSummary(
            contract_version="1.0.0",
            threshold_v1_trigger_time=100.0,
            integrity_v1_trigger_time=80.0,
            decision_compression_window=20.0,
            false_safe_rate=0.0,
            false_alert_rate=0.1,
            outage_sensitivity_score=3600.0,
            max_pc_degraded=1e-4,
            max_pc_reference=1e-5,
            max_cov_trace=50.0,
            max_staleness=3600.0,
            total_timesteps=100,
        )
        with pytest.raises(AttributeError):
            ms.contract_version = "2.0.0"  # type: ignore[misc]

    def test_all_fields_present(self):
        ms = MetricsSummary(
            contract_version="1.0.0",
            threshold_v1_trigger_time=None,
            integrity_v1_trigger_time=None,
            decision_compression_window=None,
            false_safe_rate=0.0,
            false_alert_rate=0.0,
            outage_sensitivity_score=0.0,
            max_pc_degraded=0.0,
            max_pc_reference=0.0,
            max_cov_trace=0.0,
            max_staleness=0.0,
            total_timesteps=0,
        )
        assert ms.contract_version == "1.0.0"
        assert ms.threshold_v1_trigger_time is None
        assert ms.total_timesteps == 0
