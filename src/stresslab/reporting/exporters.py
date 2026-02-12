"""Report Exporters - multi-format output for StressLAB reports.

Supported formats:
  - JSON   (schema-validated, indent=2)
  - CSV    (flat tables, suitable for spreadsheet import)
  - LaTeX  (publication-ready tables)
  - Parquet (columnar, schema-embedded)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from stresslab import __version__ as STRESSLAB_VERSION
from stresslab.metrics_contract import METRICS_CONTRACT_VERSION
from stresslab.modules.logging_engine import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# JSON export (with schema validation envelope)
# ---------------------------------------------------------------------------

def export_json(
    report: dict,
    output_path: Path,
) -> Path:
    """Export a report dict to a schema-validated JSON file.

    Wraps the report in an envelope containing schema_version,
    metrics_contract_version, and stresslab_version for provenance.

    Args:
        report: report dict (from benchmark or MC report generators).
        output_path: destination file path.

    Returns:
        The written file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "metrics_contract_version": METRICS_CONTRACT_VERSION,
        "stresslab_version": STRESSLAB_VERSION,
        "report": report,
    }

    with open(output_path, "w") as f:
        json.dump(envelope, f, indent=2, default=_json_default)

    return output_path


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(
    data: Union[dict, pd.DataFrame],
    output_path: Path,
) -> Path:
    """Export data to CSV.

    If data is a dict, it is flattened to a single-row DataFrame.
    If data is a DataFrame, it is written directly.

    Args:
        data: flat dict or DataFrame.
        output_path: destination file path.

    Returns:
        The written file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_csv(output_path, index=False)
    else:
        flat = _flatten_dict(data)
        df = pd.DataFrame([flat])
        df.to_csv(output_path, index=False)

    return output_path


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------

def export_latex_table(
    rows: Sequence[tuple[str, str]],
    output_path: Path,
    caption: str = "StressLAB Report",
    label: str = "tab:stresslab",
    column_headers: tuple[str, str] = ("Metric", "Value"),
) -> Path:
    """Export a list of (metric, value) rows as a LaTeX table.

    Produces a standalone tabular environment wrapped in a table float
    with caption and label suitable for inclusion in LaTeX documents.

    Args:
        rows: sequence of (metric_name, formatted_value) tuples.
        output_path: destination .tex file path.
        caption: table caption.
        label: LaTeX label for cross-referencing.
        column_headers: two-element tuple for header row.

    Returns:
        The written file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{_escape_latex(caption)}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{l r}",
        r"\toprule",
        f"{_escape_latex(column_headers[0])} & {_escape_latex(column_headers[1])} \\\\",
        r"\midrule",
    ]

    for metric, value in rows:
        lines.append(f"{_escape_latex(metric)} & {_escape_latex(value)} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return output_path


def report_to_latex_rows(report: dict) -> list[tuple[str, str]]:
    """Convert a benchmark report dict to LaTeX-ready (metric, value) rows.

    Flattens the report sections into human-readable metric names
    and formatted values.

    Args:
        report: benchmark report from generate_benchmark_report().

    Returns:
        List of (metric_name, formatted_value) tuples.
    """
    rows: list[tuple[str, str]] = []

    # Timing shift
    ts = report.get("timing_shift", {})
    rows.append(("Threshold-v1 trigger", _fmt_opt_float(ts.get("threshold_v1_trigger_s"), "s")))
    rows.append(("Integrity-v1 trigger", _fmt_opt_float(ts.get("integrity_v1_trigger_s"), "s")))
    rows.append(("DCW", _fmt_opt_float(ts.get("dcw_seconds"), "s")))
    rows.append(("Early warning model", str(ts.get("early_warning_model", "N/A"))))

    # Decision performance
    dp = report.get("decision_performance", {})
    rows.append(("False-safe rate", _fmt_pct(dp.get("false_safe_rate", 0.0))))
    rows.append(("False-alert rate", _fmt_pct(dp.get("false_alert_rate", 0.0))))
    rows.append(("Instability index", f"{dp.get('decision_instability_index', 0.0):.4f}"))
    rows.append(("Transitions/hour", f"{dp.get('decision_transitions_per_hour', 0.0):.2f}"))
    rows.append(("Decision entropy (nats)", f"{dp.get('decision_entropy_nats', 0.0):.4f}"))

    # Uncertainty
    ud = report.get("uncertainty_degradation", {})
    rows.append(("Max cov trace (km$^2$)", f"{ud.get('max_cov_trace_km2', 0.0):.4e}"))
    rows.append(("Max staleness (s)", f"{ud.get('max_staleness_s', 0.0):.1f}"))
    rows.append(("Mean Pc drift", f"{ud.get('mean_pc_drift', 0.0):.4e}"))
    rows.append(("Max Pc drift", f"{ud.get('max_pc_drift', 0.0):.4e}"))
    rows.append(("Staleness-Pc correlation", f"{ud.get('staleness_pc_correlation', 0.0):.4f}"))
    rows.append(("Mean freshness", f"{ud.get('mean_freshness', 0.0):.4f}"))
    rows.append(("Min freshness", f"{ud.get('min_freshness', 0.0):.4f}"))

    # Peaks
    pv = report.get("peak_values", {})
    rows.append(("Max Pc degraded", f"{pv.get('max_pc_degraded', 0.0):.3e}"))
    rows.append(("Max Pc reference", f"{pv.get('max_pc_reference', 0.0):.3e}"))
    rows.append(("Total timesteps", str(pv.get("total_timesteps", 0))))

    return rows


# ---------------------------------------------------------------------------
# Parquet export for time-series curves
# ---------------------------------------------------------------------------

def export_timeseries_csv(
    df: pd.DataFrame,
    output_path: Path,
    columns: Optional[list[str]] = None,
) -> Path:
    """Export time-series DataFrame (or selected columns) to CSV.

    Args:
        df: simulation time-series DataFrame.
        output_path: destination file path.
        columns: optional subset of columns to export.

    Returns:
        The written file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if columns is not None:
        available = [c for c in columns if c in df.columns]
        df = df[available]

    df.to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict into dot-separated keys."""
    items: list[tuple[str, object]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            # Skip list fields (timeseries curves, etc.) for flat export
            continue
        else:
            items.append((new_key, v))
    return dict(items)


def _json_default(obj: object) -> object:
    """JSON serialization fallback for numpy/pandas types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def _fmt_opt_float(val: object, unit: str = "") -> str:
    """Format an optional float value."""
    if val is None:
        return "N/A"
    suffix = f" {unit}" if unit else ""
    return f"{val:.1f}{suffix}"


def _fmt_pct(val: float) -> str:
    """Format a fraction as a percentage string."""
    return f"{val * 100:.2f}\\%"
