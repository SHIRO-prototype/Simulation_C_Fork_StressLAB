"""Monte Carlo Report - aggregate analysis for batch simulation runs.

Consumes the Monte Carlo DataFrame (one row per run) and uses the
metrics contract functions (compression_window_distribution,
outage_sensitivity_gradient) to produce structured batch reports.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from stresslab.metrics_contract import (
    compression_window_distribution,
    outage_sensitivity_gradient,
)


# ---------------------------------------------------------------------------
# Trigger timing statistics
# ---------------------------------------------------------------------------

def trigger_timing_stats(mc_df: pd.DataFrame) -> dict:
    """Compute trigger timing statistics across Monte Carlo runs.

    Args:
        mc_df: Monte Carlo DataFrame with columns threshold_v1_trigger_time,
            integrity_v1_trigger_time.

    Returns:
        dict with per-model trigger counts, means, stds, and medians.
    """
    good = _filter_good(mc_df)
    if len(good) == 0:
        return {"total_runs": len(mc_df), "successful_runs": 0}

    tv1 = good["threshold_v1_trigger_time"].dropna()
    iv1 = good["integrity_v1_trigger_time"].dropna()

    return {
        "total_runs": len(mc_df),
        "successful_runs": len(good),
        "threshold_v1": _series_stats(tv1, "trigger_time_s"),
        "integrity_v1": _series_stats(iv1, "trigger_time_s"),
    }


# ---------------------------------------------------------------------------
# DCW distribution (using contract function)
# ---------------------------------------------------------------------------

def dcw_distribution_report(mc_df: pd.DataFrame) -> dict:
    """Compute the Decision Compression Window distribution.

    Uses metrics_contract.compression_window_distribution() as the
    single source of truth.

    Args:
        mc_df: Monte Carlo DataFrame with column decision_compression_window.

    Returns:
        dict with percentiles, mean, std, min, max, count.
    """
    good = _filter_good(mc_df)
    if len(good) == 0 or "decision_compression_window" not in good.columns:
        return {}

    dcw_vals = good["decision_compression_window"].dropna().tolist()
    return compression_window_distribution(dcw_vals)


# ---------------------------------------------------------------------------
# Outage sensitivity gradient (using contract function)
# ---------------------------------------------------------------------------

def outage_gradient_report(mc_df: pd.DataFrame) -> dict:
    """Compute the outage sensitivity gradient from Monte Carlo data.

    Uses metrics_contract.outage_sensitivity_gradient() to estimate
    d(trigger_time) / d(outage_duration) via linear regression.

    Args:
        mc_df: Monte Carlo DataFrame with columns
            threshold_v1_trigger_time, integrity_v1_trigger_time,
            total_outage_duration.

    Returns:
        dict with gradients for each decision model, or None values
        if insufficient data.
    """
    good = _filter_good(mc_df)
    result: dict = {}

    if len(good) == 0 or "total_outage_duration" not in good.columns:
        return result

    outage_durations = good["total_outage_duration"].tolist()

    # Threshold-v1 gradient
    if "threshold_v1_trigger_time" in good.columns:
        tv1_triggers = [
            t if pd.notna(t) else None
            for t in good["threshold_v1_trigger_time"].tolist()
        ]
        grad = outage_sensitivity_gradient(tv1_triggers, outage_durations)
        result["threshold_v1_gradient"] = grad

    # Integrity-v1 gradient
    if "integrity_v1_trigger_time" in good.columns:
        iv1_triggers = [
            t if pd.notna(t) else None
            for t in good["integrity_v1_trigger_time"].tolist()
        ]
        grad = outage_sensitivity_gradient(iv1_triggers, outage_durations)
        result["integrity_v1_gradient"] = grad

    return result


# ---------------------------------------------------------------------------
# Pc degradation statistics
# ---------------------------------------------------------------------------

def pc_degradation_stats(mc_df: pd.DataFrame) -> dict:
    """Compute collision probability degradation statistics.

    Args:
        mc_df: Monte Carlo DataFrame with column max_pc_degraded.

    Returns:
        dict with mean, std, max, min, median of max Pc across runs.
    """
    good = _filter_good(mc_df)
    if len(good) == 0 or "max_pc_degraded" not in good.columns:
        return {}
    return _series_stats(good["max_pc_degraded"], "max_pc_degraded")


# ---------------------------------------------------------------------------
# Comparative model table (threshold-v1 vs integrity-v1)
# ---------------------------------------------------------------------------

def model_comparison_table(mc_df: pd.DataFrame) -> dict:
    """Side-by-side comparison of threshold-v1 and integrity-v1 models.

    For each model, reports the trigger rate, mean trigger time,
    and how often each triggered first (the early-warning model).

    Returns:
        dict with model-level statistics and early-warning counts.
    """
    good = _filter_good(mc_df)
    if len(good) == 0:
        return {}

    tv1 = good["threshold_v1_trigger_time"]
    iv1 = good["integrity_v1_trigger_time"]

    tv1_triggered = tv1.dropna()
    iv1_triggered = iv1.dropna()

    both = good.dropna(subset=["threshold_v1_trigger_time",
                                "integrity_v1_trigger_time"])

    n_integrity_first = 0
    n_threshold_first = 0
    n_simultaneous = 0

    if len(both) > 0:
        dcw = both["threshold_v1_trigger_time"] - both["integrity_v1_trigger_time"]
        n_integrity_first = int((dcw > 0).sum())
        n_threshold_first = int((dcw < 0).sum())
        n_simultaneous = int((dcw == 0).sum())

    return {
        "threshold_v1": {
            "trigger_rate": len(tv1_triggered) / max(len(good), 1),
            "mean_trigger_time_s": float(tv1_triggered.mean()) if len(tv1_triggered) > 0 else None,
            "std_trigger_time_s": float(tv1_triggered.std()) if len(tv1_triggered) > 0 else None,
        },
        "integrity_v1": {
            "trigger_rate": len(iv1_triggered) / max(len(good), 1),
            "mean_trigger_time_s": float(iv1_triggered.mean()) if len(iv1_triggered) > 0 else None,
            "std_trigger_time_s": float(iv1_triggered.std()) if len(iv1_triggered) > 0 else None,
        },
        "early_warning": {
            "integrity_v1_first": n_integrity_first,
            "threshold_v1_first": n_threshold_first,
            "simultaneous": n_simultaneous,
            "both_triggered_runs": len(both),
        },
    }


# ---------------------------------------------------------------------------
# Full Monte Carlo report
# ---------------------------------------------------------------------------

def generate_monte_carlo_report(mc_df: pd.DataFrame) -> dict:
    """Generate a complete Monte Carlo batch report.

    Args:
        mc_df: Monte Carlo DataFrame (one row per run).

    Returns:
        Nested dict containing all batch report sections.
    """
    return {
        "report_type": "monte_carlo_batch",
        "trigger_timing": trigger_timing_stats(mc_df),
        "dcw_distribution": dcw_distribution_report(mc_df),
        "outage_gradient": outage_gradient_report(mc_df),
        "pc_degradation": pc_degradation_stats(mc_df),
        "model_comparison": model_comparison_table(mc_df),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_good(mc_df: pd.DataFrame) -> pd.DataFrame:
    """Filter Monte Carlo DataFrame to successful runs only."""
    if "error" in mc_df.columns:
        return mc_df[mc_df["error"].isna()].copy()
    return mc_df.copy()


def _series_stats(s: pd.Series, name: str) -> dict:
    """Compute standard statistics for a numeric series."""
    if len(s) == 0:
        return {"count": 0}
    return {
        "count": len(s),
        "mean": float(s.mean()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
    }
