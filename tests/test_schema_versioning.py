"""Tests for schema versioning in output artifacts."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from stresslab.modules.logging_engine import LoggingEngine, SCHEMA_VERSION
from stresslab.metrics_contract import METRICS_CONTRACT_VERSION
from stresslab import __version__ as STRESSLAB_VERSION
from stresslab.types import (
    TimeStep,
    PropagationResult,
    StateVector,
    CovarianceResult,
    MeasurementResult,
    GeometryResult,
    RiskResult,
    DecisionResult,
    AlertState,
    IntegrityV1State,
    SimulationConfig,
)


def _make_timestep(t: float = 60.0) -> TimeStep:
    """Create a minimal valid TimeStep for testing."""
    prop = PropagationResult(
        state_obj1=StateVector(np.array([6878.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0])),
        state_obj2=StateVector(np.array([6878.1, 0.0, 0.0]), np.array([0.0, 7.5, 0.0])),
        rel_position=np.array([0.1, 0.0, 0.0]),
        rel_velocity=np.array([0.0, 0.0, 0.0]),
        stm_obj1=np.eye(6),
        stm_obj2=np.eye(6),
        estimated_tca=1000.0,
    )
    cov = CovarianceResult(
        covariance=np.eye(6) * 1e-4,
        trace=6e-4,
        frobenius=6e-4,
        max_eigenvalue=1e-4,
        growth_rate=0.0,
    )
    meas = MeasurementResult(applied=False, time_since_last_update=60.0)
    geom = GeometryResult(
        miss_distance=0.1,
        rel_velocity_at_tca=1.0,
        time_to_tca=1000.0,
        b_plane_xi=np.array([1.0, 0.0, 0.0]),
        b_plane_eta=np.array([0.0, 1.0, 0.0]),
        b_plane_zeta=np.array([0.0, 0.0, 1.0]),
    )
    risk = RiskResult(pc_reference=1e-6, pc_degraded=1e-5, risk_ratio=10.0)
    decision = DecisionResult(
        threshold_v1_alert=AlertState.SAFE,
        threshold_v1_trigger_time=None,
        integrity_v1_state=IntegrityV1State.MONITOR,
        integrity_v1_trigger_time=None,
        integrity_v1_score=0.1,
    )
    return TimeStep(
        t=t,
        propagation=prop,
        covariance_obj1=cov,
        covariance_obj2=cov,
        measurement_obj1=meas,
        measurement_obj2=meas,
        geometry=geom,
        risk=risk,
        decision=decision,
    )


class TestParquetSchemaVersion:
    def test_metadata_present(self, tmp_path: Path):
        logger = LoggingEngine()
        logger.record(_make_timestep(60.0))
        logger.record(_make_timestep(120.0))

        path = logger.write_timeseries(tmp_path, "test_run")
        meta = pq.read_metadata(str(path)).metadata

        assert b"schema_version" in meta
        assert meta[b"schema_version"] == SCHEMA_VERSION.encode()

    def test_contract_version_in_parquet(self, tmp_path: Path):
        logger = LoggingEngine()
        logger.record(_make_timestep(60.0))

        path = logger.write_timeseries(tmp_path, "test_run")
        meta = pq.read_metadata(str(path)).metadata

        assert b"metrics_contract_version" in meta
        assert meta[b"metrics_contract_version"] == METRICS_CONTRACT_VERSION.encode()

    def test_stresslab_version_in_parquet(self, tmp_path: Path):
        logger = LoggingEngine()
        logger.record(_make_timestep(60.0))

        path = logger.write_timeseries(tmp_path, "test_run")
        meta = pq.read_metadata(str(path)).metadata

        assert b"stresslab_version" in meta
        assert meta[b"stresslab_version"] == STRESSLAB_VERSION.encode()

    def test_run_id_in_parquet(self, tmp_path: Path):
        logger = LoggingEngine()
        logger.record(_make_timestep(60.0))

        path = logger.write_timeseries(tmp_path, "abc123")
        meta = pq.read_metadata(str(path)).metadata

        assert b"run_id" in meta
        assert meta[b"run_id"] == b"abc123"

    def test_data_preserved(self, tmp_path: Path):
        """Verify the actual data columns survive the pyarrow round-trip."""
        logger = LoggingEngine()
        logger.record(_make_timestep(60.0))
        logger.record(_make_timestep(120.0))

        path = logger.write_timeseries(tmp_path, "test_run")
        table = pq.read_table(str(path))
        assert len(table) == 2
        assert "timestamp" in table.column_names
        assert "pc_degraded" in table.column_names


class TestJSONSchemaVersion:
    def test_schema_version_in_json(self, tmp_path: Path):
        logger = LoggingEngine()
        logger.record(_make_timestep(60.0))

        config = SimulationConfig()
        path = logger.write_summary(tmp_path, "test_run", config, None, None)

        with open(path) as f:
            data = json.load(f)

        assert "schema_version" in data
        assert data["schema_version"] == SCHEMA_VERSION

    def test_contract_version_in_json(self, tmp_path: Path):
        logger = LoggingEngine()
        logger.record(_make_timestep(60.0))

        config = SimulationConfig()
        path = logger.write_summary(tmp_path, "test_run", config, None, None)

        with open(path) as f:
            data = json.load(f)

        assert "metrics_contract_version" in data
        assert data["metrics_contract_version"] == METRICS_CONTRACT_VERSION

    def test_stresslab_version_in_json(self, tmp_path: Path):
        logger = LoggingEngine()
        logger.record(_make_timestep(60.0))

        config = SimulationConfig()
        path = logger.write_summary(tmp_path, "test_run", config, None, None)

        with open(path) as f:
            data = json.load(f)

        assert "stresslab_version" in data
        assert data["stresslab_version"] == STRESSLAB_VERSION

    def test_all_expected_keys_in_json(self, tmp_path: Path):
        """Verify the JSON summary contains all required top-level keys."""
        logger = LoggingEngine()
        logger.record(_make_timestep(60.0))

        config = SimulationConfig()
        path = logger.write_summary(tmp_path, "test_run", config, 100.0, 80.0)

        with open(path) as f:
            data = json.load(f)

        required_keys = {
            "run_id", "seed",
            "schema_version", "metrics_contract_version", "stresslab_version",
            "threshold_v1_trigger_time", "integrity_v1_trigger_time",
            "decision_compression_window",
            "false_safe_rate", "false_alert_rate",
            "outage_sensitivity_score",
            "max_pc_degraded", "max_pc_reference", "max_cov_trace",
            "max_staleness", "total_timesteps", "dynamics_model",
            "decision_instability_index", "decision_transitions_per_hour",
            "decision_entropy",
            "mean_pc_drift", "max_pc_drift",
            "staleness_pc_correlation",
            "mean_freshness", "min_freshness",
            "sanity", "config", "story",
        }
        assert required_keys.issubset(set(data.keys()))


class TestSchemaVersionConstants:
    def test_schema_version_format(self):
        parts = SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_contract_and_schema_versions_are_semver(self):
        for ver in [SCHEMA_VERSION, METRICS_CONTRACT_VERSION]:
            parts = ver.split(".")
            assert len(parts) == 3
