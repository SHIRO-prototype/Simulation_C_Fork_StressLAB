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
import pyarrow as pa
import pyarrow.parquet as pq

from stresslab import __version__ as STRESSLAB_VERSION
from stresslab.types import (
    TimeStep,
    AlertState,
    IntegrityV1State,
    SimulationConfig,
)
from stresslab.metrics_contract import (
    METRICS_CONTRACT_VERSION,
    compression_window,
    freshness_score as compute_freshness,
    pc_drift as compute_pc_drift,
    staleness_pc_correlation,
    decision_transitions_per_hour,
    decision_entropy,
    decision_instability_index,
    is_true_danger,
    outage_sensitivity_score as compute_outage_sensitivity,
    MetricsSummary,
)

# Schema version tracks the structure of output artifacts (columns, keys).
# Bump this when the output schema changes (new columns, renamed keys, etc.)
SCHEMA_VERSION = "2.0.0"


class LoggingEngine:
    """Accumulates timestep data and writes outputs."""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def record(self, step: TimeStep) -> None:
        """Record one timestep."""
        # Compute per-timestep Phase 2 metrics via the contract
        _freshness = compute_freshness(step.measurement_obj1.time_since_last_update)
        _pc_drift = compute_pc_drift(step.risk.pc_degraded, step.risk.pc_reference)

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
            "pc_reference": step.risk.pc_reference,
            "pc_degraded": step.risk.pc_degraded,
            "risk_ratio": step.risk.risk_ratio,
            "threshold_v1_alert_state": step.decision.threshold_v1_alert.value,
            "integrity_v1_state": step.decision.integrity_v1_state.value,
            "integrity_v1_score": step.decision.integrity_v1_score,
            "measurement_applied_obj1": step.measurement_obj1.applied,
            "measurement_applied_obj2": step.measurement_obj2.applied,
            "staleness_obj1": step.measurement_obj1.time_since_last_update,
            "staleness_obj2": step.measurement_obj2.time_since_last_update,
            "freshness_score": _freshness,
            "pc_drift": _pc_drift,
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Convert accumulated records to DataFrame."""
        return pd.DataFrame(self._records)

    def write_timeseries(self, output_dir: Path, run_id: str) -> Path:
        """Write time-series data to parquet with embedded schema metadata."""
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        path = output_dir / f"timeseries_{run_id}.parquet"

        # Convert to pyarrow table so we can inject file-level metadata
        table = pa.Table.from_pandas(df, preserve_index=False)
        existing_meta = table.schema.metadata or {}
        extra = {
            b"schema_version": SCHEMA_VERSION.encode(),
            b"metrics_contract_version": METRICS_CONTRACT_VERSION.encode(),
            b"stresslab_version": STRESSLAB_VERSION.encode(),
            b"run_id": run_id.encode(),
        }
        merged = {**existing_meta, **extra}
        table = table.replace_schema_metadata(merged)
        pq.write_table(table, str(path))
        return path

    def compute_summary(
        self,
        config: SimulationConfig,
        threshold_v1_trigger_time: Optional[float],
        integrity_v1_trigger_time: Optional[float],
    ) -> MetricsSummary:
        """Compute a frozen MetricsSummary using the metrics contract."""
        df = self.to_dataframe()

        # Decision compression window (via contract)
        dcw = compression_window(threshold_v1_trigger_time, integrity_v1_trigger_time)

        # Ground-truth classification (via contract)
        hbr = config.combined_hard_body_radius
        true_danger = df["miss_distance"].apply(
            lambda md: is_true_danger(md, hbr)
        )

        # False safe: threshold-v1 said Safe but was actually dangerous
        threshold_v1_safe = df["threshold_v1_alert_state"] == AlertState.SAFE.value
        n_danger = int(true_danger.sum())
        false_safe_rate = float((threshold_v1_safe & true_danger).sum()) / max(n_danger, 1)

        # False alert: threshold-v1 said Alert but was not dangerous
        threshold_v1_alert = df["threshold_v1_alert_state"] == AlertState.ALERT.value
        true_safe = ~true_danger
        n_safe = int(true_safe.sum())
        false_alert_rate = float((threshold_v1_alert & true_safe).sum()) / max(n_safe, 1)

        # Outage sensitivity (via contract)
        max_stale = float(df["staleness_obj1"].max())

        # --- Phase 2 elevated metrics ---
        state_seq = df["integrity_v1_state"].tolist()
        dt = float(config.dt)

        # Decision instability metrics (via contract)
        instability = decision_instability_index(state_seq, dt)
        trans_per_hour = decision_transitions_per_hour(state_seq, dt)
        entropy = decision_entropy(state_seq)

        # Pc drift (via contract, from per-timestep column)
        _mean_pc_drift = float(df["pc_drift"].mean()) if len(df) > 0 else 0.0
        _max_pc_drift = float(df["pc_drift"].max()) if len(df) > 0 else 0.0

        # Staleness-Pc correlation (via contract)
        _staleness_pc_corr = staleness_pc_correlation(
            df["staleness_obj1"].tolist(), df["pc_drift"].tolist(),
        )

        # Freshness (via contract, from per-timestep column)
        _mean_freshness = float(df["freshness_score"].mean()) if len(df) > 0 else 0.0
        _min_freshness = float(df["freshness_score"].min()) if len(df) > 0 else 0.0

        return MetricsSummary(
            contract_version=METRICS_CONTRACT_VERSION,
            threshold_v1_trigger_time=threshold_v1_trigger_time,
            integrity_v1_trigger_time=integrity_v1_trigger_time,
            decision_compression_window=dcw,
            false_safe_rate=false_safe_rate,
            false_alert_rate=false_alert_rate,
            outage_sensitivity_score=compute_outage_sensitivity(max_stale),
            max_pc_degraded=float(df["pc_degraded"].max()),
            max_pc_reference=float(df["pc_reference"].max()),
            max_cov_trace=float(df["cov_trace_obj1"].max()),
            max_staleness=max_stale,
            decision_instability_index=instability,
            decision_transitions_per_hour=trans_per_hour,
            decision_entropy=entropy,
            mean_pc_drift=_mean_pc_drift,
            max_pc_drift=_max_pc_drift,
            staleness_pc_correlation=_staleness_pc_corr,
            mean_freshness=_mean_freshness,
            min_freshness=_min_freshness,
            total_timesteps=len(df),
        )

    @staticmethod
    def _build_sanity(
        config: SimulationConfig,
        max_pc_drift: float,
        max_staleness: float,
    ) -> dict:
        """Build degradation sanity flags from config and metrics."""
        has_outages = len(config.measurement.outage_windows) > 0
        return {
            "degradation_active": has_outages,
            "pc_diverged": max_pc_drift > 1e-6,
            "staleness_ramped": max_staleness > config.measurement.update_interval,
        }

    @staticmethod
    def _build_config_snapshot(config: SimulationConfig) -> dict:
        """Extract key config parameters for the summary envelope."""
        return {
            "update_interval": config.measurement.update_interval,
            "outage_windows": [
                {"start": w["start"], "end": w["end"]}
                for w in config.measurement.outage_windows
            ],
            "process_noise_scale": config.process_noise.scale,
            "pc_threshold": config.threshold_v1.pc_threshold,
            "t_start": config.t_start,
            "t_end": config.t_end,
            "dt": config.dt,
            "combined_hard_body_radius": config.combined_hard_body_radius,
        }

    @staticmethod
    def _build_story(
        config: SimulationConfig,
        metrics: MetricsSummary,
        sanity: dict,
    ) -> dict:
        """Generate a human-readable narrative for the run.

        Returns a dict with a ``bullets`` list of 6 narrative strings
        covering the key aspects of the simulation outcome.
        """
        bullets: list[str] = []

        # 1 — Tracking conditions
        n_outages = len(config.measurement.outage_windows)
        interval = config.measurement.update_interval
        if n_outages > 0:
            windows_desc = ", ".join(
                f"{w['start']:.0f}-{w['end']:.0f}s"
                for w in config.measurement.outage_windows
            )
            bullets.append(
                f"Tracking operated at {interval}s cadence with "
                f"{n_outages} outage window(s) ({windows_desc}), "
                f"reaching a peak staleness of {metrics.max_staleness:.0f}s."
            )
        else:
            bullets.append(
                f"Tracking operated continuously at {interval}s cadence "
                f"with no outages; peak staleness was {metrics.max_staleness:.0f}s."
            )

        # 2 — Uncertainty behaviour
        if metrics.max_cov_trace > 1e-2:
            unc_level = "significant"
        elif metrics.max_cov_trace > 1e-4:
            unc_level = "moderate"
        else:
            unc_level = "minimal"
        bullets.append(
            f"Position uncertainty showed {unc_level} growth "
            f"(peak covariance trace {metrics.max_cov_trace:.3e} km^2)."
        )

        # 3 — Risk divergence
        if sanity.get("pc_diverged"):
            ratio_text = (
                f"Pc diverged between reference ({metrics.max_pc_reference:.3e}) "
                f"and degraded ({metrics.max_pc_degraded:.3e}) streams, "
                f"with mean drift {metrics.mean_pc_drift:.3e} and "
                f"staleness-Pc correlation {metrics.staleness_pc_correlation:.3f}."
            )
        else:
            ratio_text = (
                "Reference and degraded Pc streams remained closely aligned; "
                "no meaningful risk divergence detected."
            )
        bullets.append(ratio_text)

        # 4 — Decision timing
        tv1 = metrics.threshold_v1_trigger_time
        iv1 = metrics.integrity_v1_trigger_time
        dcw = metrics.decision_compression_window
        if tv1 is not None and iv1 is not None:
            earlier = "Integrity V1" if iv1 > tv1 else "Threshold V1"
            bullets.append(
                f"Threshold V1 triggered at T-{tv1:.0f}s, "
                f"Integrity V1 at T-{iv1:.0f}s "
                f"(compression window {dcw:+.0f}s). "
                f"{earlier} provided earlier warning."
            )
        elif tv1 is not None:
            bullets.append(
                f"Only Threshold V1 triggered (T-{tv1:.0f}s); "
                "Integrity V1 never escalated."
            )
        elif iv1 is not None:
            bullets.append(
                f"Only Integrity V1 triggered (T-{iv1:.0f}s); "
                "Threshold V1 never alerted."
            )
        else:
            bullets.append(
                "Neither decision model triggered during this scenario."
            )

        # 5 — Reliability
        instab = metrics.decision_instability_index
        trans = metrics.decision_transitions_per_hour
        if instab < 0.01:
            stability_desc = "very stable"
        elif instab < 0.05:
            stability_desc = "moderately stable"
        else:
            stability_desc = "unstable"
        bullets.append(
            f"Decision states were {stability_desc} "
            f"(instability index {instab:.4f}, "
            f"{trans:.2f} transitions/hr, "
            f"entropy {metrics.decision_entropy:.4f} nats). "
            f"False safe rate {metrics.false_safe_rate:.3f}, "
            f"false alert rate {metrics.false_alert_rate:.3f}."
        )

        # 6 — Overall assessment
        degradation_active = sanity.get("degradation_active", False)
        pc_diverged = sanity.get("pc_diverged", False)
        freshness = metrics.mean_freshness
        if degradation_active and pc_diverged and freshness < 0.5:
            assessment = (
                "Tracking degradation materially impacted collision risk "
                "classification and decision timing in this scenario."
            )
        elif degradation_active and pc_diverged:
            assessment = (
                "Degradation produced measurable risk divergence, "
                "though freshness remained partially adequate."
            )
        elif degradation_active:
            assessment = (
                "Despite active degradation, risk divergence was negligible; "
                "the scenario was resilient to the configured outages."
            )
        else:
            assessment = (
                "No degradation was configured; this run serves as a nominal "
                "control case for comparison."
            )
        bullets.append(assessment)

        return {"bullets": bullets}

    def write_summary(
        self,
        output_dir: Path,
        run_id: str,
        config: SimulationConfig,
        threshold_v1_trigger_time: Optional[float],
        integrity_v1_trigger_time: Optional[float],
        scenario_metadata: Optional[dict] = None,
    ) -> Path:
        """Write run summary to JSON.

        Args:
            scenario_metadata: optional presentation-only metadata dict with
                run_label, scenario_title, scenario_purpose, scenario_takeaway,
                tags.  These are embedded in the JSON but do not affect
                simulation results or run_id.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = self.compute_summary(
            config, threshold_v1_trigger_time, integrity_v1_trigger_time,
        )

        sanity = self._build_sanity(config, metrics.max_pc_drift, metrics.max_staleness)
        config_snapshot = self._build_config_snapshot(config)
        story = self._build_story(config, metrics, sanity)

        # Extract scenario metadata fields (default to None / empty list)
        sm = scenario_metadata or {}
        run_label = sm.get("run_label")
        scenario_title = sm.get("scenario_title")
        scenario_purpose = sm.get("scenario_purpose")
        scenario_takeaway = sm.get("scenario_takeaway")
        tags = sm.get("tags", [])

        summary = {
            "run_id": run_id,
            "seed": config.seed,
            "schema_version": SCHEMA_VERSION,
            "metrics_contract_version": metrics.contract_version,
            "stresslab_version": STRESSLAB_VERSION,
            "run_label": run_label,
            "scenario_title": scenario_title,
            "scenario_purpose": scenario_purpose,
            "scenario_takeaway": scenario_takeaway,
            "tags": tags,
            "threshold_v1_trigger_time": metrics.threshold_v1_trigger_time,
            "integrity_v1_trigger_time": metrics.integrity_v1_trigger_time,
            "decision_compression_window": metrics.decision_compression_window,
            "false_safe_rate": metrics.false_safe_rate,
            "false_alert_rate": metrics.false_alert_rate,
            "outage_sensitivity_score": metrics.outage_sensitivity_score,
            "max_pc_degraded": metrics.max_pc_degraded,
            "max_pc_reference": metrics.max_pc_reference,
            "max_cov_trace": metrics.max_cov_trace,
            "max_staleness": metrics.max_staleness,
            "decision_instability_index": metrics.decision_instability_index,
            "decision_transitions_per_hour": metrics.decision_transitions_per_hour,
            "decision_entropy": metrics.decision_entropy,
            "mean_pc_drift": metrics.mean_pc_drift,
            "max_pc_drift": metrics.max_pc_drift,
            "staleness_pc_correlation": metrics.staleness_pc_correlation,
            "mean_freshness": metrics.mean_freshness,
            "min_freshness": metrics.min_freshness,
            "total_timesteps": metrics.total_timesteps,
            "dynamics_model": config.dynamics_model.value,
            "sanity": sanity,
            "config": config_snapshot,
            "story": story,
        }

        path = output_dir / f"summary_{run_id}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        return path
