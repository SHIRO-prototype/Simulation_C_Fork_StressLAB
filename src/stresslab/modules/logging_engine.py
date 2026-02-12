"""Logging Engine - records time-series data and experiment summaries.

Collects per-timestep data into a DataFrame and writes:
  - time_series.parquet  (full time series)
  - run_summary.json     (aggregate metrics)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from stresslab.types import (
    TimeStep,
    AlertState,
    ShiroState,
    SimulationConfig,
)


class LoggingEngine:
    """Accumulates timestep data and writes outputs."""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def record(self, step: TimeStep) -> None:
        """Record one timestep."""
        self._records.append({
            "timestamp": step.t,
            "rel_pos_x": step.propagation.rel_position[0],
            "rel_pos_y": step.propagation.rel_position[1],
            "rel_pos_z": step.propagation.rel_position[2],
            "rel_vel_x": step.propagation.rel_velocity[0],
            "rel_vel_y": step.propagation.rel_velocity[1],
            "rel_vel_z": step.propagation.rel_velocity[2],
            "miss_distance": step.geometry.miss_distance,
            "time_to_tca": step.geometry.time_to_tca,
            "rel_velocity_at_tca": step.geometry.rel_velocity_at_tca,
            "cov_trace_obj1": step.covariance_obj1.trace,
            "cov_trace_obj2": step.covariance_obj2.trace,
            "cov_frobenius_obj1": step.covariance_obj1.frobenius,
            "cov_frobenius_obj2": step.covariance_obj2.frobenius,
            "cov_max_eig_obj1": step.covariance_obj1.max_eigenvalue,
            "cov_max_eig_obj2": step.covariance_obj2.max_eigenvalue,
            "cov_growth_rate_obj1": step.covariance_obj1.growth_rate,
            "cov_growth_rate_obj2": step.covariance_obj2.growth_rate,
            "pc_baseline": step.risk.pc_baseline,
            "pc_degraded": step.risk.pc_degraded,
            "risk_ratio": step.risk.risk_ratio,
            "baseline_alert_state": step.decision.baseline_alert.value,
            "shiro_alert_state": step.decision.shiro_state.value,
            "shiro_score": step.decision.shiro_score,
            "measurement_applied_obj1": step.measurement_obj1.applied,
            "measurement_applied_obj2": step.measurement_obj2.applied,
            "staleness_obj1": step.measurement_obj1.time_since_last_update,
            "staleness_obj2": step.measurement_obj2.time_since_last_update,
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Convert accumulated records to DataFrame."""
        return pd.DataFrame(self._records)

    def write_timeseries(self, output_dir: Path, run_id: str) -> Path:
        """Write time-series data to parquet."""
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        path = output_dir / f"timeseries_{run_id}.parquet"
        df.to_parquet(path, index=False)
        return path

    def write_summary(
        self,
        output_dir: Path,
        run_id: str,
        config: SimulationConfig,
        baseline_trigger_time: Optional[float],
        shiro_trigger_time: Optional[float],
    ) -> Path:
        """Write run summary to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute aggregate metrics
        df = self.to_dataframe()

        # Decision compression window
        dcw = None
        if baseline_trigger_time is not None and shiro_trigger_time is not None:
            dcw = baseline_trigger_time - shiro_trigger_time

        # False rates (require ground truth: miss_distance < hard_body_radius)
        hbr = config.combined_hard_body_radius
        true_danger = df["miss_distance"] < hbr

        # False safe: baseline said Safe but was actually dangerous
        baseline_safe = df["baseline_alert_state"] == AlertState.SAFE.value
        false_safe_baseline = float((baseline_safe & true_danger).sum()) / max(true_danger.sum(), 1)

        # False alert: baseline said Alert but was not dangerous
        baseline_alert = df["baseline_alert_state"] == AlertState.ALERT.value
        true_safe = ~true_danger
        false_alert_baseline = float((baseline_alert & true_safe).sum()) / max(true_safe.sum(), 1)

        # Outage sensitivity score: max staleness observed
        max_staleness = float(df["staleness_obj1"].max())

        summary = {
            "run_id": run_id,
            "seed": config.seed,
            "baseline_trigger_time": baseline_trigger_time,
            "shiro_trigger_time": shiro_trigger_time,
            "decision_compression_window": dcw,
            "false_safe_rate": false_safe_baseline,
            "false_alert_rate": false_alert_baseline,
            "outage_sensitivity_score": max_staleness,
            "total_timesteps": len(df),
            "dynamics_model": config.dynamics_model.value,
        }

        path = output_dir / f"summary_{run_id}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        return path
