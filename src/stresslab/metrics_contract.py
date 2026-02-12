"""Metrics Contract - frozen definitions of all StressLAB metrics.

This module is the single source of truth for how every metric is
defined and computed. All other modules must use the functions here
rather than ad-hoc reimplementations. Changing a metric definition
here constitutes a contract change and must bump the contract version.

Contract version: 2.0.0
  v1.0.0 - Initial metrics (covariance, staleness, DCW, classification)
  v2.0.0 - Phase 2 elevation (outage gradient, instability index,
           DCW distributions, false-safe frequency, Pc drift, correlation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Contract version
# ---------------------------------------------------------------------------
METRICS_CONTRACT_VERSION = "2.0.0"


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
# Pc drift (degraded vs reference)
# ---------------------------------------------------------------------------

def pc_drift(
    pc_degraded: float,
    pc_reference: float,
) -> float:
    """Relative Pc drift between degraded and reference streams.

    Measures how much the degraded-stream collision probability deviates
    from the reference. Values near 0 mean minimal drift; large values
    indicate that tracking degradation is inflating/deflating Pc.

    Args:
        pc_degraded: Pc from degraded (outage-affected) stream.
        pc_reference: Pc from reference (ideal) stream.

    Returns:
        |Pc_degraded - Pc_reference| / max(Pc_reference, 1e-30)
    """
    return abs(pc_degraded - pc_reference) / max(pc_reference, 1e-30)


# ---------------------------------------------------------------------------
# Staleness-Pc correlation
# ---------------------------------------------------------------------------

def staleness_pc_correlation(
    staleness_series: Sequence[float],
    pc_drift_series: Sequence[float],
) -> float:
    """Pearson correlation between staleness and Pc drift time series.

    A high positive correlation means tracking staleness directly
    inflates collision probability uncertainty. This is the core
    scientific signal that StressLAB quantifies.

    Args:
        staleness_series: per-timestep staleness values [s].
        pc_drift_series: per-timestep pc_drift values (dimensionless).

    Returns:
        Pearson r in [-1, 1], or 0.0 if insufficient variance.
    """
    s = np.asarray(staleness_series, dtype=np.float64)
    d = np.asarray(pc_drift_series, dtype=np.float64)
    if len(s) < 2 or len(s) != len(d):
        return 0.0
    s_std = float(np.std(s))
    d_std = float(np.std(d))
    if s_std < 1e-30 or d_std < 1e-30:
        return 0.0
    corr_matrix = np.corrcoef(s, d)
    return float(corr_matrix[0, 1])


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


def compression_window_distribution(
    dcw_values: Sequence[float],
) -> dict:
    """Compression window distribution statistics from a batch of runs.

    Args:
        dcw_values: list of DCW values from multiple Monte Carlo runs [s].

    Returns:
        dict with keys: p10, p25, p50, p75, p90, mean, std, min, max, count
        Returns empty dict if no values provided.
    """
    if len(dcw_values) == 0:
        return {}
    arr = np.asarray(dcw_values, dtype=np.float64)
    return {
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(arr),
    }


# ---------------------------------------------------------------------------
# Decision instability
# ---------------------------------------------------------------------------

def decision_transitions_per_hour(
    state_sequence: Sequence[str],
    dt: float,
) -> float:
    """Count decision state transitions per hour.

    A transition occurs when the state at timestep i differs from
    timestep i-1. High values indicate the decision model is
    oscillating between states.

    Args:
        state_sequence: ordered list of state labels (e.g. "Monitor",
            "Watch", ...) for each timestep.
        dt: timestep duration [s].

    Returns:
        Transitions per hour. 0.0 if fewer than 2 timesteps.
    """
    if len(state_sequence) < 2 or dt <= 0.0:
        return 0.0
    transitions = sum(
        1 for i in range(1, len(state_sequence))
        if state_sequence[i] != state_sequence[i - 1]
    )
    total_hours = (len(state_sequence) - 1) * dt / 3600.0
    if total_hours <= 0.0:
        return 0.0
    return transitions / total_hours


def decision_entropy(state_sequence: Sequence[str]) -> float:
    """Shannon entropy of the decision state distribution.

    Measures how uniformly the simulation time is spent across
    different decision states. Higher entropy means more uniform
    distribution (more uncertainty in the decision).

    Uses natural log (nats). Maximum entropy for N states = ln(N).

    Args:
        state_sequence: ordered list of state labels.

    Returns:
        Shannon entropy in nats. 0.0 if sequence is empty or uniform.
    """
    if len(state_sequence) == 0:
        return 0.0
    _, counts = np.unique(state_sequence, return_counts=True)
    probs = counts / counts.sum()
    # Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def decision_instability_index(
    state_sequence: Sequence[str],
    dt: float,
    w_transitions: float = 0.5,
    w_entropy: float = 0.5,
    transitions_ref: float = 10.0,
    entropy_ref: float = 1.5,
) -> float:
    """Combined decision instability index.

    Weighted combination of normalized transitions-per-hour and
    normalized entropy, producing a dimensionless score in [0, inf).
    Higher values indicate more unstable decision behavior.

    Args:
        state_sequence: ordered list of state labels.
        dt: timestep duration [s].
        w_transitions: weight for transitions component.
        w_entropy: weight for entropy component.
        transitions_ref: reference transitions/hour for normalization.
        entropy_ref: reference entropy for normalization.

    Returns:
        Instability index (dimensionless).
    """
    tph = decision_transitions_per_hour(state_sequence, dt)
    ent = decision_entropy(state_sequence)
    return (
        w_transitions * tph / max(transitions_ref, 1e-30)
        + w_entropy * ent / max(entropy_ref, 1e-30)
    )


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


def false_safe_frequency(
    predicted_safe_flags: Sequence[bool],
    miss_distances: Sequence[float],
    combined_hard_body_radius: float,
) -> float:
    """False-safe frequency across a time series.

    Number of timesteps classified as false-safe divided by the
    total number of truly dangerous timesteps. Returns 0.0 if
    there are no dangerous timesteps.

    Args:
        predicted_safe_flags: per-timestep boolean (True if model said Safe).
        miss_distances: per-timestep miss distance [km].
        combined_hard_body_radius: combined HBR [km].

    Returns:
        False-safe frequency in [0, 1].
    """
    if len(predicted_safe_flags) == 0:
        return 0.0
    n_false_safe = 0
    n_danger = 0
    for safe, md in zip(predicted_safe_flags, miss_distances):
        if is_true_danger(md, combined_hard_body_radius):
            n_danger += 1
            if safe:
                n_false_safe += 1
    return n_false_safe / max(n_danger, 1)


# ---------------------------------------------------------------------------
# Outage sensitivity
# ---------------------------------------------------------------------------

def outage_sensitivity_score(max_staleness: float) -> float:
    """Outage sensitivity score: maximum staleness observed during a run.

    Args:
        max_staleness: maximum staleness observed [s].

    Returns:
        Score in seconds.
    """
    return max_staleness


def outage_sensitivity_gradient(
    trigger_times: Sequence[Optional[float]],
    outage_durations: Sequence[float],
) -> Optional[float]:
    """Outage sensitivity gradient: d(trigger_time)/d(outage_duration).

    Estimates how much earlier/later the decision model triggers per
    unit increase in outage duration. Computed via linear regression
    on (outage_duration, trigger_time) pairs from Monte Carlo runs.

    Only uses runs where the model actually triggered (non-None).
    Requires at least 2 valid data points with distinct outage durations.

    Args:
        trigger_times: per-run trigger times (None if model didn't trigger).
        outage_durations: per-run total outage duration [s].

    Returns:
        Gradient in [s/s] (dimensionless), or None if insufficient data.
    """
    valid_x = []
    valid_y = []
    for tt, od in zip(trigger_times, outage_durations):
        if tt is not None:
            valid_x.append(od)
            valid_y.append(tt)
    if len(valid_x) < 2:
        return None
    x = np.asarray(valid_x, dtype=np.float64)
    y = np.asarray(valid_y, dtype=np.float64)
    if np.std(x) < 1e-30:
        return None  # All same outage duration — can't compute gradient
    # Linear regression: y = a*x + b, return a
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])


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

    # Decision timing
    threshold_v1_trigger_time: Optional[float]
    integrity_v1_trigger_time: Optional[float]
    decision_compression_window: Optional[float]

    # Error classification
    false_safe_rate: float
    false_alert_rate: float

    # Outage sensitivity
    outage_sensitivity_score: float
    max_staleness: float

    # Peak values
    max_pc_degraded: float
    max_pc_reference: float
    max_cov_trace: float

    # Phase 2 elevated metrics
    decision_instability_index: float
    decision_transitions_per_hour: float
    decision_entropy: float
    mean_pc_drift: float
    max_pc_drift: float
    staleness_pc_correlation: float
    mean_freshness: float
    min_freshness: float

    # Count
    total_timesteps: int
