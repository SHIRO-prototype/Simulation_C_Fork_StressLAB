"""Tests for the decision layer."""

import numpy as np
import pytest

from stresslab.types import (
    AlertState,
    IntegrityV1State,
    ThresholdV1Config,
    IntegrityV1Config,
)
from stresslab.modules.decision_layer import (
    evaluate_threshold_v1,
    evaluate_integrity_v1,
    evaluate_decision,
)


class TestThresholdV1:
    def test_safe_when_low_pc(self):
        """Should be Safe when Pc is below threshold."""
        config = ThresholdV1Config(pc_threshold=1e-4)
        state, trigger = evaluate_threshold_v1(
            pc=1e-6, miss_distance=10.0, time_to_tca=3600.0,
            config=config, current_time=1000.0, prev_trigger_time=None,
        )
        assert state == AlertState.SAFE

    def test_alert_when_high_pc(self):
        """Should be Alert when Pc exceeds threshold within TCA gate."""
        config = ThresholdV1Config(pc_threshold=1e-4, time_to_tca_gate=86400.0)
        state, trigger = evaluate_threshold_v1(
            pc=1e-3, miss_distance=0.5, time_to_tca=3600.0,
            config=config, current_time=1000.0, prev_trigger_time=None,
        )
        assert state == AlertState.ALERT
        assert trigger == 1000.0

    def test_safe_outside_tca_gate(self):
        """Should be Safe when outside TCA gate, even with high Pc."""
        config = ThresholdV1Config(pc_threshold=1e-4, time_to_tca_gate=86400.0)
        state, trigger = evaluate_threshold_v1(
            pc=1e-2, miss_distance=0.1, time_to_tca=100000.0,
            config=config, current_time=1000.0, prev_trigger_time=None,
        )
        assert state == AlertState.SAFE

    def test_trigger_time_persists(self):
        """Trigger time should persist once set."""
        config = ThresholdV1Config(pc_threshold=1e-4)
        _, trigger1 = evaluate_threshold_v1(
            pc=1e-3, miss_distance=0.5, time_to_tca=3600.0,
            config=config, current_time=1000.0, prev_trigger_time=None,
        )
        _, trigger2 = evaluate_threshold_v1(
            pc=1e-3, miss_distance=0.5, time_to_tca=3000.0,
            config=config, current_time=2000.0, prev_trigger_time=trigger1,
        )
        assert trigger2 == 1000.0  # first trigger time


class TestIntegrityV1:
    def test_monitor_at_low_risk(self):
        """Should be Monitor when all inputs are low."""
        config = IntegrityV1Config()
        state, score, trigger = evaluate_integrity_v1(
            pc=1e-8, cov_norm=0.01, growth_rate=0.001, staleness=100.0,
            config=config, current_time=1000.0,
            prev_trigger_time=None, prev_state=IntegrityV1State.MONITOR,
        )
        assert state == IntegrityV1State.MONITOR
        assert trigger is None

    def test_critical_at_high_risk(self):
        """Should be Critical when all inputs are at reference levels."""
        config = IntegrityV1Config()
        state, score, trigger = evaluate_integrity_v1(
            pc=1e-4, cov_norm=100.0, growth_rate=1.0, staleness=86400.0,
            config=config, current_time=5000.0,
            prev_trigger_time=None, prev_state=IntegrityV1State.MONITOR,
        )
        # All terms at reference = score of 1.0, should be Critical
        assert state == IntegrityV1State.CRITICAL
        assert trigger == 5000.0

    def test_escalation_sequence(self):
        """State should escalate monotonically with increasing risk."""
        config = IntegrityV1Config()
        states = []
        for scale in [0.0, 0.1, 0.3, 0.6, 0.9]:
            state, _, _ = evaluate_integrity_v1(
                pc=config.pc_ref * scale,
                cov_norm=config.cov_norm_ref * scale,
                growth_rate=config.growth_rate_ref * scale,
                staleness=config.staleness_ref * scale,
                config=config, current_time=1000.0,
                prev_trigger_time=None, prev_state=IntegrityV1State.MONITOR,
            )
            states.append(state)

        # Should see monotonic escalation
        state_order = {
            IntegrityV1State.MONITOR: 0,
            IntegrityV1State.WATCH: 1,
            IntegrityV1State.WARNING: 2,
            IntegrityV1State.CRITICAL: 3,
        }
        orders = [state_order[s] for s in states]
        for i in range(len(orders) - 1):
            assert orders[i] <= orders[i + 1]


class TestCombinedDecision:
    def test_both_models_run(self):
        """Combined evaluation should populate both model outputs."""
        result = evaluate_decision(
            pc=1e-3, miss_distance=0.5, time_to_tca=3600.0,
            cov_norm=50.0, growth_rate=0.5, staleness=7200.0,
            current_time=10000.0,
            threshold_v1_config=ThresholdV1Config(),
            integrity_v1_config=IntegrityV1Config(),
            prev_threshold_v1_trigger=None,
            prev_integrity_v1_trigger=None,
            prev_integrity_v1_state=IntegrityV1State.MONITOR,
        )
        assert result.threshold_v1_alert is not None
        assert result.integrity_v1_state is not None
        assert isinstance(result.integrity_v1_score, float)
