"""Decision Layer - threshold-v1 and integrity-v1 decision models.

Evaluates two decision posture models side-by-side at each timestep:
  1. Threshold-v1: simple threshold on Pc and optionally miss distance
  2. Integrity-v1: weighted score of Pc, covariance norm, growth rate, staleness
     mapped to a four-state escalation model (Monitor -> Watch -> Warning -> Critical)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from stresslab.stresslab_types import (
    AlertState,
    IntegrityV1State,
    ShiroState,
    ThresholdV1Config,
    IntegrityV1Config,
    ShiroConfig,
    DecisionResult,
)


# ---------------------------------------------------------------------------
# Threshold-v1 model
# ---------------------------------------------------------------------------

def evaluate_threshold_v1(
    pc: float,
    miss_distance: float,
    time_to_tca: float,
    config: ThresholdV1Config,
    current_time: float,
    prev_trigger_time: Optional[float],
) -> tuple[AlertState, Optional[float]]:
    """Evaluate threshold-v1 decision model.

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
# Integrity-v1 model
# ---------------------------------------------------------------------------

def _compute_integrity_v1_score(
    pc: float,
    cov_norm: float,
    growth_rate: float,
    staleness: float,
    config: IntegrityV1Config,
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


def _score_to_state(score: float, config: IntegrityV1Config) -> IntegrityV1State:
    """Map score to integrity-v1 state."""
    if score >= config.threshold_warning_to_critical:
        return IntegrityV1State.CRITICAL
    elif score >= config.threshold_watch_to_warning:
        return IntegrityV1State.WARNING
    elif score >= config.threshold_monitor_to_watch:
        return IntegrityV1State.WATCH
    else:
        return IntegrityV1State.MONITOR


def evaluate_integrity_v1(
    pc: float,
    cov_norm: float,
    growth_rate: float,
    staleness: float,
    config: IntegrityV1Config,
    current_time: float,
    prev_trigger_time: Optional[float],
    prev_state: IntegrityV1State,
) -> tuple[IntegrityV1State, float, Optional[float]]:
    """Evaluate integrity-v1 decision model.

    Returns:
        (new_state, score, trigger_time)
        trigger_time is set when state first reaches Warning or Critical.
    """
    score = _compute_integrity_v1_score(pc, cov_norm, growth_rate, staleness, config)
    new_state = _score_to_state(score, config)

    trigger_time = prev_trigger_time
    if new_state in (IntegrityV1State.WARNING, IntegrityV1State.CRITICAL):
        if trigger_time is None:
            trigger_time = current_time

    return new_state, score, trigger_time


# ---------------------------------------------------------------------------
# SHIRO posture state machine (SAFE -> ELEVATED -> CRITICAL)
# ---------------------------------------------------------------------------

def evaluate_shiro(
    pc: float,
    miss_distance_km: float,
    dt_since_last_update_s: float,
    config: ShiroConfig,
) -> tuple[ShiroState, Optional[str]]:
    """Evaluate SHIRO posture: CRITICAL if inside d_act + Pc; ELEVATED if geometry relevant + stale.

    Returns:
        (shiro_state, trigger_path or None)
        trigger_path: "CRITICAL" | "FRESHNESS" | "sigma_growth_rate" | "sigma_scalar" | None
    """
    # CRITICAL: inside d_act and Pc >= threshold
    if miss_distance_km <= config.d_act_km and pc >= config.pc_critical:
        return ShiroState.CRITICAL, "CRITICAL"

    # Geometry relevant = inside watch gate (miss distance within d_watch)
    geometry_relevant = miss_distance_km <= config.d_watch_km

    # ELEVATED: geometry relevant + freshness breach
    if geometry_relevant and dt_since_last_update_s >= config.dt_max_s:
        return ShiroState.ELEVATED, "FRESHNESS"

    return ShiroState.SAFE, None


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
    threshold_v1_config: ThresholdV1Config,
    integrity_v1_config: IntegrityV1Config,
    prev_threshold_v1_trigger: Optional[float],
    prev_integrity_v1_trigger: Optional[float],
    prev_integrity_v1_state: IntegrityV1State,
    shiro_config: Optional[ShiroConfig] = None,
) -> DecisionResult:
    """Run both decision models (and optionally SHIRO) and return combined result."""
    threshold_v1_alert, threshold_v1_trigger = evaluate_threshold_v1(
        pc, miss_distance, time_to_tca,
        threshold_v1_config, current_time, prev_threshold_v1_trigger,
    )

    integrity_v1_state, integrity_v1_score, integrity_v1_trigger = evaluate_integrity_v1(
        pc, cov_norm, growth_rate, staleness,
        integrity_v1_config, current_time, prev_integrity_v1_trigger,
        prev_integrity_v1_state,
    )

    shiro_state: Optional[ShiroState] = None
    shiro_trigger_path: Optional[str] = None
    if shiro_config is not None:
        shiro_state, shiro_trigger_path = evaluate_shiro(
            pc, miss_distance, staleness, shiro_config,
        )

    return DecisionResult(
        threshold_v1_alert=threshold_v1_alert,
        threshold_v1_trigger_time=threshold_v1_trigger,
        integrity_v1_state=integrity_v1_state,
        integrity_v1_trigger_time=integrity_v1_trigger,
        integrity_v1_score=integrity_v1_score,
        shiro_state=shiro_state,
        shiro_trigger_path=shiro_trigger_path,
    )
