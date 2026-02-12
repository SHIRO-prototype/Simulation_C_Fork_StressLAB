"""Tests for the decision layer."""

import numpy as np
import pytest

from stresslab.types import (
    AlertState,
    ShiroState,
    BaselineThresholdConfig,
    ShiroConfig,
)
from stresslab.modules.decision_layer import (
    evaluate_baseline,
    evaluate_shiro,
    evaluate_decision,
)


class TestBaseline:
    def test_safe_when_low_pc(self):
        """Should be Safe when Pc is below threshold."""
        config = BaselineThresholdConfig(pc_threshold=1e-4)
        state, trigger = evaluate_baseline(
            pc=1e-6, miss_distance=10.0, time_to_tca=3600.0,
            config=config, current_time=1000.0, prev_trigger_time=None,
        )
        assert state == AlertState.SAFE

    def test_alert_when_high_pc(self):
        """Should be Alert when Pc exceeds threshold within TCA gate."""
        config = BaselineThresholdConfig(pc_threshold=1e-4, time_to_tca_gate=86400.0)
        state, trigger = evaluate_baseline(
            pc=1e-3, miss_distance=0.5, time_to_tca=3600.0,
            config=config, current_time=1000.0, prev_trigger_time=None,
        )
        assert state == AlertState.ALERT
        assert trigger == 1000.0

    def test_safe_outside_tca_gate(self):
        """Should be Safe when outside TCA gate, even with high Pc."""
        config = BaselineThresholdConfig(pc_threshold=1e-4, time_to_tca_gate=86400.0)
        state, trigger = evaluate_baseline(
            pc=1e-2, miss_distance=0.1, time_to_tca=100000.0,
            config=config, current_time=1000.0, prev_trigger_time=None,
        )
        assert state == AlertState.SAFE

    def test_trigger_time_persists(self):
        """Trigger time should persist once set."""
        config = BaselineThresholdConfig(pc_threshold=1e-4)
        _, trigger1 = evaluate_baseline(
            pc=1e-3, miss_distance=0.5, time_to_tca=3600.0,
            config=config, current_time=1000.0, prev_trigger_time=None,
        )
        _, trigger2 = evaluate_baseline(
            pc=1e-3, miss_distance=0.5, time_to_tca=3000.0,
            config=config, current_time=2000.0, prev_trigger_time=trigger1,
        )
        assert trigger2 == 1000.0  # first trigger time


class TestShiro:
    def test_monitor_at_low_risk(self):
        """Should be Monitor when all inputs are low."""
        config = ShiroConfig()
        state, score, trigger = evaluate_shiro(
            pc=1e-8, cov_norm=0.01, growth_rate=0.001, staleness=100.0,
            config=config, current_time=1000.0,
            prev_trigger_time=None, prev_state=ShiroState.MONITOR,
        )
        assert state == ShiroState.MONITOR
        assert trigger is None

    def test_critical_at_high_risk(self):
        """Should be Critical when all inputs are at reference levels."""
        config = ShiroConfig()
        state, score, trigger = evaluate_shiro(
            pc=1e-4, cov_norm=100.0, growth_rate=1.0, staleness=86400.0,
            config=config, current_time=5000.0,
            prev_trigger_time=None, prev_state=ShiroState.MONITOR,
        )
        # All terms at reference = score of 1.0, should be Critical
        assert state == ShiroState.CRITICAL
        assert trigger == 5000.0

    def test_escalation_sequence(self):
        """State should escalate monotonically with increasing risk."""
        config = ShiroConfig()
        states = []
        for scale in [0.0, 0.1, 0.3, 0.6, 0.9]:
            state, _, _ = evaluate_shiro(
                pc=config.pc_ref * scale,
                cov_norm=config.cov_norm_ref * scale,
                growth_rate=config.growth_rate_ref * scale,
                staleness=config.staleness_ref * scale,
                config=config, current_time=1000.0,
                prev_trigger_time=None, prev_state=ShiroState.MONITOR,
            )
            states.append(state)

        # Should see monotonic escalation
        state_order = {
            ShiroState.MONITOR: 0,
            ShiroState.WATCH: 1,
            ShiroState.WARNING: 2,
            ShiroState.CRITICAL: 3,
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
            baseline_config=BaselineThresholdConfig(),
            shiro_config=ShiroConfig(),
            prev_baseline_trigger=None,
            prev_shiro_trigger=None,
            prev_shiro_state=ShiroState.MONITOR,
        )
        assert result.baseline_alert is not None
        assert result.shiro_state is not None
        assert isinstance(result.shiro_score, float)
