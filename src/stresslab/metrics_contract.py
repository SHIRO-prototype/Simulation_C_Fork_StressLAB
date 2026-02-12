"""Metrics Contract - frozen definitions of all StressLAB metrics.

This module is the single source of truth for how every metric is
defined and computed. All other modules must use the functions here
rather than ad-hoc reimplementations. Changing a metric definition
here constitutes a contract change and must bump the schema version.

Contract version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Contract version
# ---------------------------------------------------------------------------
METRICS_CONTRACT_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Covariance metrics
# ---------------------------------------------------------------------------

def covariance_norm(P: np.ndarray) -> float:
    """Covariance norm defined as trace of the full 6x6 covariance.

    Args:
        P: (6,6) covariance matrix in ECI [km^2, km^2/s, km^2/s^2].

    Returns:
        trace(P) in mixed units.
    """
    return float(np.trace(P))


def covariance_growth_rate(
    current_trace: float,
    previous_trace: float,
    dt: float,
) -> float:
    """Covariance growth rate: d(trace)/dt via first-order finite difference.

    Args:
        current_trace: trace(P) at current time.
        previous_trace: trace(P) at previous timestep.
        dt: timestep duration [s].

    Returns:
        Growth rate [km^2/s] (mixed-unit trace rate).
    """
    if dt <= 0.0:
        return 0.0
    return (current_trace - previous_trace) / dt


# ---------------------------------------------------------------------------
# Staleness metrics
# ---------------------------------------------------------------------------

def staleness(t_now: float, last_update_time: float) -> float:
    """Time since last measurement update.

    Args:
        t_now: current epoch time [s].
        last_update_time: epoch time of last successful update [s].

    Returns:
        Staleness in seconds (always >= 0).
    """
    return max(t_now - last_update_time, 0.0)


def freshness_score(
    staleness_s: float,
    staleness_ref: float = 86400.0,
) -> float:
    """Normalized freshness score in [0, 1].

    1.0 = perfectly fresh (staleness = 0)
    0.0 = maximally stale (staleness >= staleness_ref)

    Args:
        staleness_s: staleness in seconds.
        staleness_ref: reference staleness for normalization [s].

    Returns:
        Freshness in [0, 1].
    """
    return float(max(1.0 - staleness_s / max(staleness_ref, 1e-30), 0.0))


# ---------------------------------------------------------------------------
# Decision timing metrics
# ---------------------------------------------------------------------------

def compression_window(
    threshold_v1_trigger_time: Optional[float],
    integrity_v1_trigger_time: Optional[float],
) -> Optional[float]:
    """Decision compression window (DCW).

    Defined as the time difference between the threshold-v1 trigger
    and the integrity-v1 trigger. Positive means integrity-v1 triggered
    first (earlier warning).

    Args:
        threshold_v1_trigger_time: epoch time of threshold-v1 trigger [s].
        integrity_v1_trigger_time: epoch time of integrity-v1 trigger [s].

    Returns:
        DCW in seconds, or None if either model did not trigger.
    """
    if threshold_v1_trigger_time is None or integrity_v1_trigger_time is None:
        return None
    return threshold_v1_trigger_time - integrity_v1_trigger_time


# ---------------------------------------------------------------------------
# Ground truth and error classification
# ---------------------------------------------------------------------------

def is_true_danger(miss_distance: float, combined_hard_body_radius: float) -> bool:
    """Ground truth: is the conjunction actually dangerous?

    A conjunction is classified as dangerous if the miss distance
    is less than the combined hard-body radius.

    Args:
        miss_distance: closest approach distance [km].
        combined_hard_body_radius: sum of object radii [km].

    Returns:
        True if dangerous.
    """
    return miss_distance < combined_hard_body_radius


def is_false_safe(
    predicted_safe: bool,
    miss_distance: float,
    combined_hard_body_radius: float,
) -> bool:
    """False-safe classification.

    A false-safe occurs when the decision model declares Safe but
    the ground truth is dangerous (miss < hard_body_radius).

    Args:
        predicted_safe: True if decision model says Safe.
        miss_distance: ground-truth miss distance [km].
        combined_hard_body_radius: combined hard-body radius [km].

    Returns:
        True if this is a false-safe.
    """
    return predicted_safe and is_true_danger(miss_distance, combined_hard_body_radius)


def is_false_alert(
    predicted_alert: bool,
    miss_distance: float,
    combined_hard_body_radius: float,
) -> bool:
    """False-alert classification.

    A false-alert occurs when the decision model declares Alert but
    the ground truth is safe (miss >= hard_body_radius).

    Args:
        predicted_alert: True if decision model says Alert.
        miss_distance: ground-truth miss distance [km].
        combined_hard_body_radius: combined hard-body radius [km].

    Returns:
        True if this is a false-alert.
    """
    return predicted_alert and not is_true_danger(miss_distance, combined_hard_body_radius)


# ---------------------------------------------------------------------------
# Outage sensitivity
# ---------------------------------------------------------------------------

def outage_sensitivity_score(max_staleness: float) -> float:
    """Outage sensitivity score: maximum staleness observed during a run.

    This is the simplest form of outage sensitivity. Phase 2 will
    extend this to include gradients (d(trigger_time)/d(outage_duration)).

    Args:
        max_staleness: maximum staleness observed [s].

    Returns:
        Score in seconds.
    """
    return max_staleness


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricsSummary:
    """Frozen summary of all metrics for one simulation run.

    This is the canonical output contract. All fields are defined
    by the functions above.
    """
    contract_version: str
    threshold_v1_trigger_time: Optional[float]
    integrity_v1_trigger_time: Optional[float]
    decision_compression_window: Optional[float]
    false_safe_rate: float
    false_alert_rate: float
    outage_sensitivity_score: float
    max_pc_degraded: float
    max_pc_reference: float
    max_cov_trace: float
    max_staleness: float
    total_timesteps: int
