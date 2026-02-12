"""Plotting module for StressLAB reports.

Generates publication-quality matplotlib figures for single runs,
parameter sweeps, and Monte Carlo campaigns. All functions accept
output paths and call plt.close() to avoid memory leaks.

No seaborn dependency. Axes are clearly labeled. Labels come from
parameters, not hard-coded names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------
_STYLE = {
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
}

# Decision state color maps
_THRESHOLD_V1_COLORS = {
    "Safe": "#4CAF50",   # green
    "Alert": "#F44336",  # red
}

_INTEGRITY_V1_COLORS = {
    "Monitor":  "#4CAF50",  # green
    "Watch":    "#FFC107",  # amber
    "Warning":  "#FF9800",  # orange
    "Critical": "#F44336",  # red
}


def _apply_style():
    """Apply the shared matplotlib rcParams."""
    plt.rcParams.update(_STYLE)


def _hours(seconds):
    """Convert seconds array/scalar to hours."""
    return np.asarray(seconds, dtype=np.float64) / 3600.0


# ---------------------------------------------------------------------------
# 1. Posture Timeline Overlay (single run)
# ---------------------------------------------------------------------------

def plot_posture_timeline(ts_df, output_path, title=None):
    """Decision posture timeline overlay for a single simulation run.

    Shows threshold-v1 and integrity-v1 decision states over time as
    colored bands, with Pc (degraded) and miss distance overlaid.

    Args:
        ts_df: pandas DataFrame from timeseries parquet. Expected columns:
            timestamp, threshold_v1_alert_state, integrity_v1_state,
            pc_degraded, pc_reference, miss_distance.
        output_path: Path to save the PNG.
        title: Optional plot title override.
    """
    _apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title or "Decision Posture Timeline", fontsize=14, fontweight="bold")

    t_hours = _hours(ts_df["timestamp"].values)

    # --- Panel 1: Threshold-v1 state bands + Pc ---
    _draw_state_bands(
        ax1, t_hours, ts_df["threshold_v1_alert_state"].values,
        _THRESHOLD_V1_COLORS, alpha=0.25,
    )
    ax1.semilogy(t_hours, ts_df["pc_degraded"].values, color="#1976D2",
                 label="Pc (degraded)", linewidth=1.5)
    ax1.semilogy(t_hours, ts_df["pc_reference"].values, color="#90CAF9",
                 label="Pc (reference)", linewidth=1.0, linestyle="--")
    ax1.set_ylabel("Collision Probability")
    ax1.set_title("Threshold-v1 Decision + Collision Probability")
    _add_state_legend(ax1, _THRESHOLD_V1_COLORS, extra_handles=ax1.get_legend_handles_labels()[0])
    ax1.set_ylim(bottom=max(ts_df["pc_reference"].min() * 0.1, 1e-30))

    # --- Panel 2: Integrity-v1 state bands + integrity score ---
    _draw_state_bands(
        ax2, t_hours, ts_df["integrity_v1_state"].values,
        _INTEGRITY_V1_COLORS, alpha=0.25,
    )
    if "integrity_v1_score" in ts_df.columns:
        ax2.plot(t_hours, ts_df["integrity_v1_score"].values, color="#6A1B9A",
                 label="Integrity score", linewidth=1.5)
    ax2.set_ylabel("Integrity Score")
    ax2.set_title("Integrity-v1 Decision + Composite Score")
    _add_state_legend(ax2, _INTEGRITY_V1_COLORS, extra_handles=ax2.get_legend_handles_labels()[0])

    # --- Panel 3: Staleness + miss distance ---
    ax3_twin = ax3.twinx()
    ax3.plot(t_hours, ts_df["staleness_obj1"].values / 3600.0, color="#E65100",
             label="Staleness (hours)", linewidth=1.5)
    ax3_twin.plot(t_hours, ts_df["miss_distance"].values, color="#0D47A1",
                  label="Miss distance (km)", linewidth=1.0, linestyle="--")
    ax3.set_xlabel("Time (hours)")
    ax3.set_ylabel("Staleness (hours)")
    ax3_twin.set_ylabel("Miss Distance (km)")
    ax3.set_title("Tracking Staleness + Miss Distance")

    # Combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_state_bands(ax, t_hours, state_series, color_map, alpha=0.25):
    """Draw colored vertical bands for each contiguous block of the same state."""
    if len(t_hours) == 0:
        return
    prev_state = state_series[0]
    block_start = t_hours[0]
    for i in range(1, len(state_series)):
        if state_series[i] != prev_state or i == len(state_series) - 1:
            end = t_hours[i]
            color = color_map.get(prev_state, "#CCCCCC")
            ax.axvspan(block_start, end, color=color, alpha=alpha)
            block_start = end
            prev_state = state_series[i]


def _add_state_legend(ax, color_map, extra_handles=None):
    """Add a legend combining state-color patches + any extra line handles."""
    handles = list(extra_handles) if extra_handles else []
    for state_name, color in color_map.items():
        handles.append(mpatches.Patch(color=color, alpha=0.4, label=state_name))
    ax.legend(handles=handles, loc="upper right", ncol=2)


# ---------------------------------------------------------------------------
# 2. Pc Drift vs Staleness (single run)
# ---------------------------------------------------------------------------

def plot_pc_drift_vs_staleness(ts_df, output_path, title=None):
    """Scatter/line of Pc drift vs staleness for a single run.

    Visualizes the relationship between tracking data staleness and
    the divergence of collision probability between degraded and
    reference knowledge streams.

    Args:
        ts_df: pandas DataFrame with columns: staleness_obj1, pc_drift.
        output_path: Path to save the PNG.
        title: Optional plot title override.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title or "Pc Drift vs Tracking Staleness", fontsize=14, fontweight="bold")

    staleness_h = ts_df["staleness_obj1"].values / 3600.0
    pc_drift_vals = ts_df["pc_drift"].values

    # Left: scatter plot
    scatter = ax1.scatter(staleness_h, pc_drift_vals, c=_hours(ts_df["timestamp"].values),
                          cmap="viridis", s=8, alpha=0.7, edgecolors="none")
    ax1.set_xlabel("Staleness (hours)")
    ax1.set_ylabel("Pc Drift (|Pc_deg - Pc_ref| / Pc_ref)")
    ax1.set_title("Scatter: Colored by Simulation Time")
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("Time (hours)")

    # Right: time series overlay
    t_hours = _hours(ts_df["timestamp"].values)
    ax2_twin = ax2.twinx()
    ln1 = ax2.plot(t_hours, staleness_h, color="#E65100", label="Staleness (hours)", linewidth=1.2)
    ln2 = ax2_twin.plot(t_hours, pc_drift_vals, color="#1976D2", label="Pc drift", linewidth=1.2)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Staleness (hours)")
    ax2_twin.set_ylabel("Pc Drift")
    ax2.set_title("Time Series Overlay")
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Outage Duration vs Trigger Time Shift (sweep)
# ---------------------------------------------------------------------------

def plot_outage_vs_trigger_shift(results, param_name, output_path, title=None):
    """Sweep plot: parameter value vs decision trigger times.

    Shows how the swept parameter affects when threshold-v1 and
    integrity-v1 trigger, and the Decision Compression Window.

    Args:
        results: list of dicts from sweep_summary.json["results"].
            Each dict must have: param_value, threshold_v1_trigger_time,
            integrity_v1_trigger_time, decision_compression_window.
        param_name: name of the swept parameter for axis labels.
        output_path: Path to save the PNG.
        title: Optional plot title override.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title or f"Sweep: {param_name} vs Decision Timing",
                 fontsize=14, fontweight="bold")

    param_vals = [r["param_value"] for r in results]
    labels = [r.get("param_value_raw", str(r["param_value"])) for r in results]
    tv1_times = [r.get("threshold_v1_trigger_time") for r in results]
    iv1_times = [r.get("integrity_v1_trigger_time") for r in results]
    dcw_vals = [r.get("decision_compression_window") for r in results]

    x = np.arange(len(param_vals))

    # Panel 1: trigger times
    tv1_plot = [v / 3600.0 if v is not None else np.nan for v in tv1_times]
    iv1_plot = [v / 3600.0 if v is not None else np.nan for v in iv1_times]

    ax1.plot(x, tv1_plot, "s-", color="#F44336", label="Threshold-v1 trigger",
             markersize=8, linewidth=2)
    ax1.plot(x, iv1_plot, "o-", color="#1976D2", label="Integrity-v1 trigger",
             markersize=8, linewidth=2)
    ax1.set_ylabel("Trigger Time (hours)")
    ax1.set_title("Decision Trigger Times")
    ax1.legend()

    # Panel 2: Decision Compression Window
    dcw_plot = [v / 3600.0 if v is not None else 0.0 for v in dcw_vals]
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in dcw_plot]
    ax2.bar(x, dcw_plot, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("DCW (hours)")
    ax2.set_title("Decision Compression Window (positive = integrity-v1 triggered earlier)")
    ax2.set_xlabel(param_name)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Outage Duration vs False-Safe Rate (sweep)
# ---------------------------------------------------------------------------

def plot_outage_vs_false_safe_rate(results, param_name, output_path, title=None):
    """Sweep plot: parameter value vs false-safe rate and other risk metrics.

    Args:
        results: list of dicts from sweep_summary.json["results"].
            Each dict must have: param_value, false_safe_rate,
            max_pc_degraded, decision_instability_index.
        param_name: name of the swept parameter for axis labels.
        output_path: Path to save the PNG.
        title: Optional plot title override.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title or f"Sweep: {param_name} vs Risk Metrics",
                 fontsize=14, fontweight="bold")

    labels = [r.get("param_value_raw", str(r["param_value"])) for r in results]
    x = np.arange(len(results))

    fsr = [r.get("false_safe_rate", 0) for r in results]
    max_pc = [r.get("max_pc_degraded", 0) for r in results]
    instability = [r.get("decision_instability_index", 0) for r in results]

    # Panel 1: False-safe rate + max Pc
    ax1.bar(x - 0.2, fsr, width=0.4, color="#FF9800", alpha=0.85,
            label="False-safe rate", edgecolor="black", linewidth=0.5)
    ax1_twin = ax1.twinx()
    ax1_twin.semilogy(x, max_pc, "D-", color="#9C27B0", markersize=7,
                      label="Max Pc (degraded)", linewidth=2)
    ax1.set_ylabel("False-Safe Rate")
    ax1_twin.set_ylabel("Max Pc (degraded)")
    ax1.set_title("False-Safe Rate + Peak Collision Probability")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    # Panel 2: Decision instability
    ax2.bar(x, instability, color="#1976D2", alpha=0.85,
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Decision Instability Index")
    ax2.set_title("Decision Instability")
    ax2.set_xlabel(param_name)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Compression Window Distribution (Monte Carlo)
# ---------------------------------------------------------------------------

def plot_compression_window_distribution(dcw_vals, output_path, title=None):
    """Histogram of Decision Compression Window values from MC campaign.

    Args:
        dcw_vals: list/array of DCW values in seconds.
        output_path: Path to save the PNG.
        title: Optional plot title override.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title or "Decision Compression Window Distribution",
                 fontsize=14, fontweight="bold")

    arr = np.asarray(dcw_vals, dtype=np.float64)
    arr_hours = arr / 3600.0

    # Left: histogram
    n_bins = min(50, max(10, len(arr) // 5))
    ax1.hist(arr_hours, bins=n_bins, color="#1976D2", alpha=0.8,
             edgecolor="black", linewidth=0.5)
    ax1.axvline(np.mean(arr_hours), color="#F44336", linestyle="--",
                linewidth=2, label=f"Mean: {np.mean(arr_hours):.2f}h")
    ax1.axvline(np.median(arr_hours), color="#4CAF50", linestyle="-.",
                linewidth=2, label=f"Median: {np.median(arr_hours):.2f}h")
    ax1.set_xlabel("DCW (hours)")
    ax1.set_ylabel("Count")
    ax1.set_title("Histogram")
    ax1.legend()

    # Right: box plot + percentile annotations
    bp = ax2.boxplot(arr_hours, vert=True, patch_artist=True,
                     boxprops=dict(facecolor="#BBDEFB", edgecolor="black"),
                     medianprops=dict(color="#F44336", linewidth=2))
    ax2.set_ylabel("DCW (hours)")
    ax2.set_title("Box Plot")
    ax2.set_xticklabels(["DCW"])

    # Annotate percentiles
    p10, p25, p75, p90 = np.percentile(arr_hours, [10, 25, 75, 90])
    ax2.annotate(f"P10: {p10:.2f}h", xy=(1.15, p10), fontsize=9, color="#666666")
    ax2.annotate(f"P90: {p90:.2f}h", xy=(1.15, p90), fontsize=9, color="#666666")

    # Stats text box
    stats_text = (
        f"N = {len(arr)}\n"
        f"Mean = {np.mean(arr_hours):.3f}h\n"
        f"Std = {np.std(arr_hours):.3f}h\n"
        f"Min = {np.min(arr_hours):.3f}h\n"
        f"Max = {np.max(arr_hours):.3f}h"
    )
    ax2.text(0.55, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Decision Instability Visualization (single run)
# ---------------------------------------------------------------------------

def plot_decision_instability(ts_df, output_path, title=None):
    """Decision instability visualization showing state transitions
    and the integrity-v1 score over time.

    Highlights transition points with vertical markers and shows
    state duration fractions as a stacked bar on the right.

    Args:
        ts_df: pandas DataFrame with columns: timestamp,
            integrity_v1_state, integrity_v1_score, threshold_v1_alert_state.
        output_path: Path to save the PNG.
        title: Optional plot title override.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                             gridspec_kw={"width_ratios": [4, 1]})
    fig.suptitle(title or "Decision Instability Analysis",
                 fontsize=14, fontweight="bold")

    t_hours = _hours(ts_df["timestamp"].values)
    iv1_states = ts_df["integrity_v1_state"].values
    tv1_states = ts_df["threshold_v1_alert_state"].values

    # --- Top-left: Integrity-v1 state timeline with transitions ---
    ax_iv1 = axes[0, 0]
    _draw_state_bands(ax_iv1, t_hours, iv1_states, _INTEGRITY_V1_COLORS, alpha=0.3)

    # Mark transitions
    transitions_iv1 = []
    for i in range(1, len(iv1_states)):
        if iv1_states[i] != iv1_states[i - 1]:
            transitions_iv1.append(t_hours[i])
            ax_iv1.axvline(t_hours[i], color="#333333", linewidth=0.7,
                           linestyle=":", alpha=0.6)

    if "integrity_v1_score" in ts_df.columns:
        ax_iv1.plot(t_hours, ts_df["integrity_v1_score"].values, color="#6A1B9A",
                    linewidth=1.5, label="Integrity score")
    ax_iv1.set_ylabel("Integrity Score")
    ax_iv1.set_title(f"Integrity-v1 States ({len(transitions_iv1)} transitions)")
    _add_state_legend(ax_iv1, _INTEGRITY_V1_COLORS,
                      extra_handles=ax_iv1.get_legend_handles_labels()[0])

    # --- Top-right: State fraction pie/bar for integrity-v1 ---
    ax_pie_iv1 = axes[0, 1]
    _draw_state_fraction_bar(ax_pie_iv1, iv1_states, _INTEGRITY_V1_COLORS,
                             "Integrity-v1 State Fractions")

    # --- Bottom-left: Threshold-v1 state timeline with transitions ---
    ax_tv1 = axes[1, 0]
    _draw_state_bands(ax_tv1, t_hours, tv1_states, _THRESHOLD_V1_COLORS, alpha=0.3)

    transitions_tv1 = []
    for i in range(1, len(tv1_states)):
        if tv1_states[i] != tv1_states[i - 1]:
            transitions_tv1.append(t_hours[i])
            ax_tv1.axvline(t_hours[i], color="#333333", linewidth=0.7,
                           linestyle=":", alpha=0.6)

    ax_tv1.plot(t_hours, ts_df["pc_degraded"].values, color="#1976D2",
                linewidth=1.2, label="Pc (degraded)")
    ax_tv1.set_yscale("log")
    ax_tv1.set_xlabel("Time (hours)")
    ax_tv1.set_ylabel("Collision Probability")
    ax_tv1.set_title(f"Threshold-v1 States ({len(transitions_tv1)} transitions)")
    _add_state_legend(ax_tv1, _THRESHOLD_V1_COLORS,
                      extra_handles=ax_tv1.get_legend_handles_labels()[0])

    # --- Bottom-right: State fraction for threshold-v1 ---
    ax_pie_tv1 = axes[1, 1]
    _draw_state_fraction_bar(ax_pie_tv1, tv1_states, _THRESHOLD_V1_COLORS,
                             "Threshold-v1 State Fractions")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_state_fraction_bar(ax, state_series, color_map, title):
    """Draw a horizontal stacked bar showing fraction of time in each state."""
    unique_states, counts = np.unique(state_series, return_counts=True)
    total = counts.sum()
    fractions = counts / total

    # Sort by the canonical order in color_map
    order = list(color_map.keys())
    sorted_pairs = []
    for state in order:
        idx = np.where(unique_states == state)[0]
        if len(idx) > 0:
            sorted_pairs.append((state, fractions[idx[0]]))
    # Add any states not in the order
    for state, frac in zip(unique_states, fractions):
        if state not in order:
            sorted_pairs.append((state, frac))

    if not sorted_pairs:
        ax.set_visible(False)
        return

    states_sorted = [p[0] for p in sorted_pairs]
    fracs_sorted = [p[1] for p in sorted_pairs]
    colors_sorted = [color_map.get(s, "#CCCCCC") for s in states_sorted]

    left = 0.0
    for state, frac, color in zip(states_sorted, fracs_sorted, colors_sorted):
        ax.barh(0, frac, left=left, color=color, edgecolor="black",
                linewidth=0.5, height=0.5)
        if frac > 0.05:
            ax.text(left + frac / 2, 0, f"{frac:.0%}",
                    ha="center", va="center", fontsize=8, fontweight="bold")
        left += frac

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of Time")
    ax.set_title(title, fontsize=10)

    # Legend below
    patches = [mpatches.Patch(color=color_map.get(s, "#CCCCCC"), label=s)
               for s in states_sorted]
    ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=8)
