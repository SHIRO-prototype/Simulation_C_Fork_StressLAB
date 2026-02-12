"""Tests for the metrics contract module."""

import numpy as np
import pytest

from stresslab.metrics_contract import (
    METRICS_CONTRACT_VERSION,
    covariance_norm,
    covariance_growth_rate,
    staleness,
    freshness_score,
    pc_drift,
    staleness_pc_correlation,
    compression_window,
    compression_window_distribution,
    decision_transitions_per_hour,
    decision_entropy,
    decision_instability_index,
    is_true_danger,
    is_false_safe,
    is_false_alert,
    false_safe_frequency,
    outage_sensitivity_score,
    outage_sensitivity_gradient,
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

class TestPcDrift:
    def test_zero_drift(self):
        assert pc_drift(1e-5, 1e-5) == pytest.approx(0.0)

    def test_large_drift(self):
        # degraded = 1e-3, reference = 1e-5 => drift = |1e-3 - 1e-5| / 1e-5 = 99.0
        assert pc_drift(1e-3, 1e-5) == pytest.approx(99.0)

    def test_zero_reference(self):
        # reference = 0 => denominator clamps to 1e-30
        result = pc_drift(1e-5, 0.0)
        assert result > 0.0
        assert np.isfinite(result)

    def test_asymmetric_by_reference(self):
        # pc_drift divides by reference, so swapping args changes result
        drift_a = pc_drift(1e-3, 1e-5)  # |1e-3 - 1e-5| / 1e-5 = 99
        drift_b = pc_drift(1e-5, 1e-3)  # |1e-5 - 1e-3| / 1e-3 = 0.99
        assert drift_a == pytest.approx(99.0)
        assert drift_b == pytest.approx(0.99)
        assert drift_a != pytest.approx(drift_b)

    def test_returns_float(self):
        assert isinstance(pc_drift(1e-5, 1e-6), float)


class TestStalenessPcCorrelation:
    def test_perfect_positive_correlation(self):
        s = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert staleness_pc_correlation(s, d) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        s = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = [50.0, 40.0, 30.0, 20.0, 10.0]
        assert staleness_pc_correlation(s, d) == pytest.approx(-1.0)

    def test_no_correlation(self):
        # Orthogonal signals
        np.random.seed(42)
        s = list(np.random.randn(1000))
        d = list(np.random.randn(1000))
        r = staleness_pc_correlation(s, d)
        assert abs(r) < 0.1  # Should be near zero for large independent samples

    def test_insufficient_data(self):
        assert staleness_pc_correlation([1.0], [2.0]) == pytest.approx(0.0)

    def test_zero_variance(self):
        s = [5.0, 5.0, 5.0]
        d = [1.0, 2.0, 3.0]
        assert staleness_pc_correlation(s, d) == pytest.approx(0.0)

    def test_mismatched_lengths(self):
        assert staleness_pc_correlation([1.0, 2.0], [3.0]) == pytest.approx(0.0)


class TestCompressionWindowDistribution:
    def test_empty(self):
        assert compression_window_distribution([]) == {}

    def test_single_value(self):
        result = compression_window_distribution([42.0])
        assert result["mean"] == pytest.approx(42.0)
        assert result["count"] == 1
        assert result["std"] == pytest.approx(0.0)
        assert result["min"] == pytest.approx(42.0)
        assert result["max"] == pytest.approx(42.0)

    def test_known_distribution(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = compression_window_distribution(values)
        assert result["p50"] == pytest.approx(30.0)
        assert result["mean"] == pytest.approx(30.0)
        assert result["min"] == pytest.approx(10.0)
        assert result["max"] == pytest.approx(50.0)
        assert result["count"] == 5

    def test_all_keys_present(self):
        result = compression_window_distribution([1.0, 2.0, 3.0])
        expected_keys = {"p10", "p25", "p50", "p75", "p90", "mean", "std", "min", "max", "count"}
        assert expected_keys == set(result.keys())


class TestDecisionTransitionsPerHour:
    def test_no_transitions(self):
        states = ["Monitor", "Monitor", "Monitor", "Monitor"]
        assert decision_transitions_per_hour(states, 60.0) == pytest.approx(0.0)

    def test_some_transitions(self):
        # 3 timesteps, dt=3600s each => 2 intervals = 2 hours, 2 transitions
        states = ["Monitor", "Watch", "Critical"]
        assert decision_transitions_per_hour(states, 3600.0) == pytest.approx(1.0)

    def test_single_timestep(self):
        assert decision_transitions_per_hour(["Monitor"], 60.0) == pytest.approx(0.0)

    def test_empty(self):
        assert decision_transitions_per_hour([], 60.0) == pytest.approx(0.0)

    def test_zero_dt(self):
        assert decision_transitions_per_hour(["A", "B"], 0.0) == pytest.approx(0.0)

    def test_all_different(self):
        # 4 states, 3 transitions, dt=1200s => 3*1200=3600s = 1 hour
        states = ["A", "B", "C", "D"]
        assert decision_transitions_per_hour(states, 1200.0) == pytest.approx(3.0)


class TestDecisionEntropy:
    def test_single_state(self):
        # All same => entropy = 0
        assert decision_entropy(["Monitor"] * 100) == pytest.approx(0.0)

    def test_two_uniform(self):
        # Exactly half-half => entropy = ln(2)
        states = ["A"] * 50 + ["B"] * 50
        assert decision_entropy(states) == pytest.approx(np.log(2))

    def test_four_uniform(self):
        states = ["A"] * 25 + ["B"] * 25 + ["C"] * 25 + ["D"] * 25
        assert decision_entropy(states) == pytest.approx(np.log(4))

    def test_empty(self):
        assert decision_entropy([]) == pytest.approx(0.0)

    def test_non_negative(self):
        assert decision_entropy(["A", "B", "A", "B"]) >= 0.0


class TestDecisionInstabilityIndex:
    def test_all_stable(self):
        states = ["Monitor"] * 100
        idx = decision_instability_index(states, 60.0)
        # Transitions = 0, entropy = 0 => index = 0
        assert idx == pytest.approx(0.0)

    def test_unstable(self):
        # Alternating states => high transitions, high entropy
        states = ["A", "B"] * 50
        idx = decision_instability_index(states, 60.0)
        assert idx > 0.0

    def test_custom_weights(self):
        states = ["A", "B"] * 50
        idx_trans = decision_instability_index(states, 60.0, w_transitions=1.0, w_entropy=0.0)
        idx_ent = decision_instability_index(states, 60.0, w_transitions=0.0, w_entropy=1.0)
        idx_both = decision_instability_index(states, 60.0, w_transitions=0.5, w_entropy=0.5)
        assert idx_both == pytest.approx(0.5 * idx_trans + 0.5 * idx_ent)


class TestFalseSafeFrequency:
    def test_no_danger(self):
        # All safe => frequency = 0 (no dangerous timesteps)
        safe_flags = [True, True, True]
        miss = [1.0, 1.0, 1.0]
        assert false_safe_frequency(safe_flags, miss, 0.01) == pytest.approx(0.0)

    def test_all_false_safe(self):
        # All dangerous and all predicted safe => frequency = 1
        safe_flags = [True, True, True]
        miss = [0.001, 0.001, 0.001]
        assert false_safe_frequency(safe_flags, miss, 0.01) == pytest.approx(1.0)

    def test_mixed(self):
        # 2 dangerous timesteps, 1 predicted safe during danger => 0.5
        safe_flags = [True, False, True]
        miss = [0.001, 0.001, 1.0]
        assert false_safe_frequency(safe_flags, miss, 0.01) == pytest.approx(0.5)

    def test_empty(self):
        assert false_safe_frequency([], [], 0.01) == pytest.approx(0.0)


class TestOutageSensitivityGradient:
    def test_insufficient_data(self):
        assert outage_sensitivity_gradient([100.0], [60.0]) is None

    def test_no_triggers(self):
        assert outage_sensitivity_gradient([None, None], [60.0, 120.0]) is None

    def test_linear_relationship(self):
        # trigger_time = 2*outage_duration + const => gradient = 2
        trigger_times = [100.0, 200.0, 300.0, 400.0]
        outage_durations = [0.0, 50.0, 100.0, 150.0]
        grad = outage_sensitivity_gradient(trigger_times, outage_durations)
        assert grad is not None
        assert grad == pytest.approx(2.0)

    def test_constant_outage(self):
        # All same outage duration => can't compute gradient
        assert outage_sensitivity_gradient([100.0, 200.0], [60.0, 60.0]) is None

    def test_with_none_entries(self):
        # Some runs didn't trigger; should still work with remaining
        trigger_times = [None, 100.0, None, 200.0, 300.0]
        outage_durations = [10.0, 0.0, 20.0, 50.0, 100.0]
        grad = outage_sensitivity_gradient(trigger_times, outage_durations)
        assert grad is not None


class TestMetricsSummary:
    """Helper to build a valid MetricsSummary with all Phase 2 fields."""

    @staticmethod
    def _make(**overrides):
        defaults = dict(
            contract_version="2.0.0",
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
            decision_instability_index=0.5,
            decision_transitions_per_hour=2.0,
            decision_entropy=0.69,
            mean_pc_drift=1.5,
            max_pc_drift=10.0,
            staleness_pc_correlation=0.8,
            mean_freshness=0.9,
            min_freshness=0.3,
            total_timesteps=100,
        )
        defaults.update(overrides)
        return MetricsSummary(**defaults)

    def test_frozen(self):
        ms = self._make()
        with pytest.raises(AttributeError):
            ms.contract_version = "3.0.0"  # type: ignore[misc]

    def test_all_fields_present(self):
        ms = self._make(
            threshold_v1_trigger_time=None,
            integrity_v1_trigger_time=None,
            decision_compression_window=None,
            total_timesteps=0,
        )
        assert ms.contract_version == "2.0.0"
        assert ms.threshold_v1_trigger_time is None
        assert ms.total_timesteps == 0

    def test_phase2_fields(self):
        ms = self._make()
        assert ms.decision_instability_index == pytest.approx(0.5)
        assert ms.decision_transitions_per_hour == pytest.approx(2.0)
        assert ms.decision_entropy == pytest.approx(0.69)
        assert ms.mean_pc_drift == pytest.approx(1.5)
        assert ms.max_pc_drift == pytest.approx(10.0)
        assert ms.staleness_pc_correlation == pytest.approx(0.8)
        assert ms.mean_freshness == pytest.approx(0.9)
        assert ms.min_freshness == pytest.approx(0.3)
