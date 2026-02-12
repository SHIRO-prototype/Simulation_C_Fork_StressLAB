"""Benchmark Report - structured report generation for a single simulation run.

Takes a MetricsSummary and/or a time-series DataFrame and produces a
structured report dict that can be exported in any format (JSON, CSV,
LaTeX) by the exporters module.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd

from stresslab.metrics_contract import MetricsSummary


def _fmt_time(t: Optional[float], unit: str = "s") -> str:
    """Format a time value for display, handling None."""
    if t is None:
        return "N/A"
    if unit == "h":
        return f"{t / 3600.0:.2f} h"
    return f"{t:.1f} s"


def _fmt_pc(pc: float) -> str:
    """Format a collision probability for display."""
    if pc < 1e-30:
        return "0"
    return f"{pc:.3e}"


def _fmt_pct(val: float) -> str:
    """Format a fraction as a percentage string."""
    return f"{val * 100:.2f}%"


# ---------------------------------------------------------------------------
# Timing shift table
# ---------------------------------------------------------------------------

def timing_shift_table(summary: MetricsSummary) -> dict:
    """Build the trigger timing shift table.

    Compares threshold-v1 and integrity-v1 trigger times and
    computes the decision compression window.

    Returns:
        dict with keys: threshold_v1_trigger, integrity_v1_trigger,
        dcw_seconds, dcw_description, early_warning_model.
    """
    tv1 = summary.threshold_v1_trigger_time
    iv1 = summary.integrity_v1_trigger_time
    dcw = summary.decision_compression_window

    if dcw is not None and dcw > 0:
        early = "integrity_v1"
        desc = f"integrity-v1 triggered {dcw:.1f}s earlier"
    elif dcw is not None and dcw < 0:
        early = "threshold_v1"
        desc = f"threshold-v1 triggered {abs(dcw):.1f}s earlier"
    elif dcw is not None:
        early = "simultaneous"
        desc = "Both models triggered simultaneously"
    else:
        early = "incomplete"
        desc = "One or both models did not trigger"

    return {
        "threshold_v1_trigger_s": tv1,
        "integrity_v1_trigger_s": iv1,
        "dcw_seconds": dcw,
        "dcw_description": desc,
        "early_warning_model": early,
    }


# ---------------------------------------------------------------------------
# Decision performance table
# ---------------------------------------------------------------------------

def decision_performance_table(summary: MetricsSummary) -> dict:
    """Build the decision performance table.

    Returns:
        dict with error rates, instability metrics, and freshness.
    """
    return {
        "false_safe_rate": summary.false_safe_rate,
        "false_alert_rate": summary.false_alert_rate,
        "decision_instability_index": summary.decision_instability_index,
        "decision_transitions_per_hour": summary.decision_transitions_per_hour,
        "decision_entropy_nats": summary.decision_entropy,
    }


# ---------------------------------------------------------------------------
# Uncertainty and degradation table
# ---------------------------------------------------------------------------

def uncertainty_table(summary: MetricsSummary) -> dict:
    """Build the uncertainty / degradation impact table.

    Returns:
        dict with covariance, staleness, Pc drift, and correlation metrics.
    """
    return {
        "max_cov_trace_km2": summary.max_cov_trace,
        "max_staleness_s": summary.max_staleness,
        "outage_sensitivity_score_s": summary.outage_sensitivity_score,
        "mean_pc_drift": summary.mean_pc_drift,
        "max_pc_drift": summary.max_pc_drift,
        "staleness_pc_correlation": summary.staleness_pc_correlation,
        "mean_freshness": summary.mean_freshness,
        "min_freshness": summary.min_freshness,
        "max_pc_degraded": summary.max_pc_degraded,
        "max_pc_reference": summary.max_pc_reference,
    }


# ---------------------------------------------------------------------------
# Peak values table
# ---------------------------------------------------------------------------

def peak_values_table(summary: MetricsSummary) -> dict:
    """Build the peak / extreme values table."""
    return {
        "max_pc_degraded": summary.max_pc_degraded,
        "max_pc_reference": summary.max_pc_reference,
        "max_cov_trace_km2": summary.max_cov_trace,
        "max_staleness_s": summary.max_staleness,
        "total_timesteps": summary.total_timesteps,
    }


# ---------------------------------------------------------------------------
# Time-series summary curves
# ---------------------------------------------------------------------------

def timeseries_curves(df: pd.DataFrame) -> dict:
    """Extract key time-series curves from the simulation DataFrame.

    Returns a dict of arrays suitable for plotting or CSV export:
        timestamp, pc_reference, pc_degraded, pc_drift,
        cov_trace_obj1, staleness_obj1, freshness_score,
        integrity_v1_score, miss_distance.
    """
    columns = [
        "timestamp", "pc_reference", "pc_degraded", "pc_drift",
        "cov_trace_obj1", "staleness_obj1", "freshness_score",
        "integrity_v1_score", "miss_distance",
    ]
    out = {}
    for col in columns:
        if col in df.columns:
            out[col] = df[col].tolist()
    return out


# ---------------------------------------------------------------------------
# Full single-run report
# ---------------------------------------------------------------------------

def generate_benchmark_report(
    summary: MetricsSummary,
    df: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Generate a complete benchmark report for a single simulation run.

    Args:
        summary: MetricsSummary from the logging engine.
        df: optional time-series DataFrame for curve data.
        run_id: optional run identifier.

    Returns:
        Nested dict containing all report sections.
    """
    report: dict = {
        "report_type": "single_run_benchmark",
        "contract_version": summary.contract_version,
    }
    if run_id is not None:
        report["run_id"] = run_id

    report["timing_shift"] = timing_shift_table(summary)
    report["decision_performance"] = decision_performance_table(summary)
    report["uncertainty_degradation"] = uncertainty_table(summary)
    report["peak_values"] = peak_values_table(summary)

    if df is not None:
        report["timeseries_curves"] = timeseries_curves(df)

    return report
