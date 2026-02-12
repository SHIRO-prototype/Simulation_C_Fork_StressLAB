"""Comparative Analysis - side-by-side comparison of simulation runs.

Supports comparing two individual run summaries (JSON files), or
analyzing the threshold-v1 vs integrity-v1 model performance across
a Monte Carlo batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from stresslab.metrics_contract import MetricsSummary


# ---------------------------------------------------------------------------
# Single-run comparison
# ---------------------------------------------------------------------------

def compare_summaries(
    summary_a: Union[dict, MetricsSummary],
    summary_b: Union[dict, MetricsSummary],
    label_a: str = "run_a",
    label_b: str = "run_b",
) -> dict:
    """Compare two individual simulation run summaries.

    Computes absolute and relative deltas for all numeric metrics.

    Args:
        summary_a: first run summary (dict from JSON or MetricsSummary).
        summary_b: second run summary (dict from JSON or MetricsSummary).
        label_a: label for the first run.
        label_b: label for the second run.

    Returns:
        dict with keys: label_a, label_b, metrics (list of per-metric dicts
        with name, value_a, value_b, delta, rel_delta_pct).
    """
    a = _to_flat(summary_a)
    b = _to_flat(summary_b)

    # Collect all numeric keys present in both
    numeric_keys = sorted(
        k for k in set(a.keys()) & set(b.keys())
        if isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))
        and a[k] is not None and b[k] is not None
    )

    metrics = []
    for key in numeric_keys:
        va = a[key]
        vb = b[key]
        delta = vb - va
        denom = abs(va) if abs(va) > 1e-30 else 1.0
        rel_pct = (delta / denom) * 100.0

        metrics.append({
            "metric": key,
            f"value_{label_a}": va,
            f"value_{label_b}": vb,
            "delta": delta,
            "rel_delta_pct": rel_pct,
        })

    return {
        "comparison_type": "single_run_pair",
        "label_a": label_a,
        "label_b": label_b,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Load summaries from JSON files
# ---------------------------------------------------------------------------

def load_summary_json(path: Path) -> dict:
    """Load a run summary from a JSON file.

    Handles both raw summary files and schema-enveloped report files.

    Args:
        path: path to JSON file.

    Returns:
        Summary dict (flat, not enveloped).
    """
    with open(path) as f:
        data = json.load(f)

    # If it's an enveloped report, unwrap it
    if "report" in data and isinstance(data["report"], dict):
        return data["report"]
    return data


# ---------------------------------------------------------------------------
# Compare from file paths
# ---------------------------------------------------------------------------

def compare_from_files(
    path_a: Path,
    path_b: Path,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
) -> dict:
    """Compare two run summaries loaded from JSON files.

    Args:
        path_a: path to first summary JSON.
        path_b: path to second summary JSON.
        label_a: optional label (defaults to filename stem).
        label_b: optional label (defaults to filename stem).

    Returns:
        Comparison dict from compare_summaries().
    """
    a = load_summary_json(path_a)
    b = load_summary_json(path_b)

    if label_a is None:
        label_a = path_a.stem
    if label_b is None:
        label_b = path_b.stem

    return compare_summaries(a, b, label_a, label_b)


# ---------------------------------------------------------------------------
# Comparison to flat table
# ---------------------------------------------------------------------------

def comparison_to_dataframe(comparison: dict) -> pd.DataFrame:
    """Convert a comparison dict to a DataFrame for export.

    Args:
        comparison: result from compare_summaries() or compare_from_files().

    Returns:
        DataFrame with one row per metric.
    """
    return pd.DataFrame(comparison.get("metrics", []))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_flat(summary: Union[dict, MetricsSummary]) -> dict:
    """Convert a MetricsSummary or dict to a flat dict."""
    if isinstance(summary, MetricsSummary):
        from dataclasses import asdict
        return asdict(summary)
    return dict(summary)
