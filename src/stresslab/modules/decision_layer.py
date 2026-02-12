"""Decision Layer - baseline threshold and SHIRO confidence-integrity models.

Evaluates two decision posture models side-by-side at each timestep:
  1. Baseline: simple threshold on Pc and optionally miss distance
  2. SHIRO: weighted score of Pc, covariance norm, growth rate, staleness
     mapped to a four-state escalation model (Monitor -> Watch -> Warning -> Critical)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from stresslab.types import (
    AlertState,
    ShiroState,
    BaselineThresholdConfig,
    ShiroConfig,
    DecisionResult,
)


# ---------------------------------------------------------------------------
# Baseline threshold model
# ---------------------------------------------------------------------------

def evaluate_baseline(
    pc: float,
    miss_distance: float,
    time_to_tca: float,
    config: BaselineThresholdConfig,
    current_time: float,
    prev_trigger_time: Optional[float],
) -> tuple[AlertState, Optional[float]]:
    """Evaluate baseline threshold model.

    Triggers Alert if:
      - Pc >= pc_threshold
      - (optional) miss_distance <= miss_distance_threshold
      - time_to_tca <= time_to_tca_gate

    Returns:
        (alert_state, trigger_time)
    """
    # Check TCA gate
    if time_to_tca > config.time_to_tca_gate:
        return AlertState.SAFE, prev_trigger_time

    # Check Pc threshold
    pc_triggered = pc >= config.pc_threshold

    # Check miss distance if configured
    md_triggered = True
    if config.miss_distance_threshold is not None:
        md_triggered = miss_distance <= config.miss_distance_threshold

    if pc_triggered and md_triggered:
        trigger_time = prev_trigger_time if prev_trigger_time is not None else current_time
        return AlertState.ALERT, trigger_time
    else:
        return AlertState.SAFE, prev_trigger_time


# ---------------------------------------------------------------------------
# SHIRO confidence-integrity model
# ---------------------------------------------------------------------------

def _compute_shiro_score(
    pc: float,
    cov_norm: float,
    growth_rate: float,
    staleness: float,
    config: ShiroConfig,
) -> float:
    """Compute weighted integrity score.

    score = w1 * (Pc / Pc_ref) + w2 * (cov_norm / cov_ref)
          + w3 * (growth_rate / growth_ref) + w4 * (staleness / staleness_ref)

    All terms are clipped to [0, 1] before weighting.
    """
    w = config.weights
    terms = np.array([
        min(pc / max(config.pc_ref, 1e-30), 1.0),
        min(cov_norm / max(config.cov_norm_ref, 1e-30), 1.0),
        min(abs(growth_rate) / max(config.growth_rate_ref, 1e-30), 1.0),
        min(staleness / max(config.staleness_ref, 1e-30), 1.0),
    ])
    return float(np.dot(w, terms))


def _score_to_state(score: float, config: ShiroConfig) -> ShiroState:
    """Map score to SHIRO state."""
    if score >= config.threshold_warning_to_critical:
        return ShiroState.CRITICAL
    elif score >= config.threshold_watch_to_warning:
        return ShiroState.WARNING
    elif score >= config.threshold_monitor_to_watch:
        return ShiroState.WATCH
    else:
        return ShiroState.MONITOR


def evaluate_shiro(
    pc: float,
    cov_norm: float,
    growth_rate: float,
    staleness: float,
    config: ShiroConfig,
    current_time: float,
    prev_trigger_time: Optional[float],
    prev_state: ShiroState,
) -> tuple[ShiroState, float, Optional[float]]:
    """Evaluate SHIRO confidence-integrity model.

    Returns:
        (new_state, score, trigger_time)
        trigger_time is set when state first reaches Warning or Critical.
    """
    score = _compute_shiro_score(pc, cov_norm, growth_rate, staleness, config)
    new_state = _score_to_state(score, config)

    trigger_time = prev_trigger_time
    if new_state in (ShiroState.WARNING, ShiroState.CRITICAL):
        if trigger_time is None:
            trigger_time = current_time

    return new_state, score, trigger_time


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_decision(
    pc: float,
    miss_distance: float,
    time_to_tca: float,
    cov_norm: float,
    growth_rate: float,
    staleness: float,
    current_time: float,
    baseline_config: BaselineThresholdConfig,
    shiro_config: ShiroConfig,
    prev_baseline_trigger: Optional[float],
    prev_shiro_trigger: Optional[float],
    prev_shiro_state: ShiroState,
) -> DecisionResult:
    """Run both decision models and return combined result."""
    baseline_alert, baseline_trigger = evaluate_baseline(
        pc, miss_distance, time_to_tca,
        baseline_config, current_time, prev_baseline_trigger,
    )

    shiro_state, shiro_score, shiro_trigger = evaluate_shiro(
        pc, cov_norm, growth_rate, staleness,
        shiro_config, current_time, prev_shiro_trigger,
        prev_shiro_state,
    )

    return DecisionResult(
        baseline_alert=baseline_alert,
        baseline_trigger_time=baseline_trigger,
        shiro_state=shiro_state,
        shiro_trigger_time=shiro_trigger,
        shiro_score=shiro_score,
    )
