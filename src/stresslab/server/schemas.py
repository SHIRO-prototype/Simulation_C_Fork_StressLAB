"""Pydantic response schemas for the StressLAB API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from stresslab import __version__ as STRESSLAB_VERSION
from stresslab.metrics_contract import METRICS_CONTRACT_VERSION
from stresslab.modules.logging_engine import SCHEMA_VERSION


class VersionMixin(BaseModel):
    """Fields present on every API response."""
    schema_version: str = SCHEMA_VERSION
    contract_version: str = METRICS_CONTRACT_VERSION
    code_version: str = STRESSLAB_VERSION


class HealthResponse(VersionMixin):
    status: str = "ok"


class RunIndex(BaseModel):
    """Summary entry for one simulation run."""
    run_id: str
    path: str  # relative to workspace
    created_at: Optional[str] = None
    seed: Optional[int] = None
    dynamics_model: Optional[str] = None
    threshold_v1_trigger_time: Optional[float] = None
    integrity_v1_trigger_time: Optional[float] = None
    decision_compression_window: Optional[float] = None
    false_safe_rate: Optional[float] = None
    false_alert_rate: Optional[float] = None
    max_pc_degraded: Optional[float] = None
    max_staleness: Optional[float] = None
    decision_instability_index: Optional[float] = None
    total_timesteps: Optional[int] = None
    outage_sensitivity_score: Optional[float] = None
    max_pc_reference: Optional[float] = None
    max_cov_trace: Optional[float] = None
    decision_transitions_per_hour: Optional[float] = None
    decision_entropy: Optional[float] = None
    mean_pc_drift: Optional[float] = None
    max_pc_drift: Optional[float] = None
    staleness_pc_correlation: Optional[float] = None
    mean_freshness: Optional[float] = None
    min_freshness: Optional[float] = None


class RunListResponse(VersionMixin):
    runs: list[RunIndex]
    total: int


class RunSummaryResponse(VersionMixin):
    summary: dict[str, Any]


class TimeseriesResponse(VersionMixin):
    meta: dict[str, Any]
    columns: list[str]
    rows: list[list[Any]]


class SweepIndex(BaseModel):
    sweep_id: str
    path: str
    sweep_param: Optional[str] = None
    n_values: Optional[int] = None
    seed: Optional[int] = None
    created_at: Optional[str] = None


class SweepListResponse(VersionMixin):
    sweeps: list[SweepIndex]
    total: int


class SweepSummaryResponse(VersionMixin):
    summary: dict[str, Any]


class MCBatchIndex(BaseModel):
    batch_id: str
    path: str
    n_runs: Optional[int] = None
    successful_runs: Optional[int] = None
    created_at: Optional[str] = None


class MCListResponse(VersionMixin):
    batches: list[MCBatchIndex]
    total: int


class MCSummaryResponse(VersionMixin):
    summary: dict[str, Any]
