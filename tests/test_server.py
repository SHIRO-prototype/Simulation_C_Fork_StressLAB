"""Tests for the StressLAB dashboard server API.

Uses FastAPI's TestClient (backed by httpx) to exercise every endpoint.
Generates a short simulation run as test fixture data.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from stresslab import __version__
from stresslab.metrics_contract import METRICS_CONTRACT_VERSION
from stresslab.modules.logging_engine import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixture: generate a short simulation run into a temp workspace
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Run a short simulation and return the workspace path."""
    ws = tmp_path_factory.mktemp("server_ws")

    from stresslab.modules.scenario_generator import generate_default_scenario
    from stresslab.modules.simulation_runner import run_simulation

    config = generate_default_scenario(
        seed=99, miss_distance_km=0.5, t_end=3600.0, dt=120.0,
    )
    result = run_simulation(config, output_dir=ws, verbose=False)
    # Stash run_id for later use
    ws_run_id = result["run_id"]
    (ws / ".run_id").write_text(ws_run_id)
    return ws


@pytest.fixture(scope="module")
def run_id(workspace):
    return (workspace / ".run_id").read_text()


@pytest.fixture(scope="module")
def client(workspace):
    """Create a TestClient for the StressLAB API."""
    from fastapi.testclient import TestClient
    from stresslab.server.app import create_app

    app = create_app(workspace=workspace, dev_mode=True)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_status(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    def test_health_has_versions(self, client):
        data = client.get("/api/health").json()
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["contract_version"] == METRICS_CONTRACT_VERSION
        assert data["code_version"] == __version__


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class TestRuns:
    def test_list_runs(self, client):
        r = client.get("/api/runs")
        assert r.status_code == 200
        data = r.json()
        assert "runs" in data
        assert "total" in data
        assert data["total"] >= 1
        # Version fields on list response
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["contract_version"] == METRICS_CONTRACT_VERSION
        assert data["code_version"] == __version__

    def test_list_runs_search(self, client, run_id):
        r = client.get(f"/api/runs?q={run_id[:8]}")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1
        found_ids = [run["run_id"] for run in data["runs"]]
        assert run_id in found_ids

    def test_get_summary(self, client, run_id):
        r = client.get(f"/api/runs/{run_id}/summary")
        assert r.status_code == 200
        data = r.json()
        assert "summary" in data
        summary = data["summary"]
        assert summary["run_id"] == run_id
        assert "false_safe_rate" in summary
        assert "max_pc_degraded" in summary
        assert "decision_instability_index" in summary
        # Version envelope
        assert data["schema_version"] == SCHEMA_VERSION

    def test_get_summary_not_found(self, client):
        r = client.get("/api/runs/nonexistent_id_xyz/summary")
        assert r.status_code == 404

    def test_timeseries_json(self, client, run_id):
        r = client.get(f"/api/runs/{run_id}/timeseries?format=json")
        assert r.status_code == 200
        data = r.json()
        assert "columns" in data
        assert "rows" in data
        assert "timestamp" in data["columns"]
        assert len(data["rows"]) > 0
        assert data["meta"]["run_id"] == run_id
        assert "run_label" in data["meta"]
        assert "loaded_path" in data["meta"]
        assert "n_rows" in data["meta"]
        assert "key_field_ranges" in data["meta"]
        # Version fields
        assert data["schema_version"] == SCHEMA_VERSION

    def test_timeseries_json_with_column_filter(self, client, run_id):
        r = client.get(
            f"/api/runs/{run_id}/timeseries?format=json&cols=pc_reference,pc_degraded"
        )
        assert r.status_code == 200
        data = r.json()
        cols = data["columns"]
        # timestamp always included + requested columns
        assert "timestamp" in cols
        assert "pc_reference" in cols
        assert "pc_degraded" in cols
        # Should not have columns we didn't request
        assert "miss_distance" not in cols

    def test_timeseries_json_downsample(self, client, run_id):
        # downsample param must be >= 100 per API contract
        r = client.get(f"/api/runs/{run_id}/timeseries?format=json&downsample=100")
        assert r.status_code == 200
        data = r.json()
        # With dt=120 over 3600s that's 31 points, all below 100, so no reduction
        assert len(data["rows"]) > 0

    def test_timeseries_arrow(self, client, run_id):
        r = client.get(f"/api/runs/{run_id}/timeseries?format=arrow")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/vnd.apache.arrow.stream"
        assert r.headers["x-schema-version"] == SCHEMA_VERSION
        assert r.headers["x-contract-version"] == METRICS_CONTRACT_VERSION
        assert r.headers["x-code-version"] == __version__
        # Should be valid Arrow IPC bytes
        assert len(r.content) > 0

    def test_timeseries_not_found(self, client):
        r = client.get("/api/runs/nonexistent_id_xyz/timeseries")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

class TestDownload:
    def test_download_summary_json(self, client, run_id):
        filename = f"summary_{run_id}.json"
        r = client.get(f"/api/runs/{run_id}/download/{filename}")
        assert r.status_code == 200
        data = json.loads(r.content)
        assert data["run_id"] == run_id

    def test_download_timeseries_parquet(self, client, run_id):
        filename = f"timeseries_{run_id}.parquet"
        r = client.get(f"/api/runs/{run_id}/download/{filename}")
        assert r.status_code == 200
        assert len(r.content) > 100  # non-trivial parquet file

    def test_path_traversal_blocked(self, client, run_id):
        # URL-encoded traversal — router may decode before handler.
        # When SPA catch-all is active, the decoded path may not match
        # the API route, so Starlette serves index.html instead (200).
        # This is still safe: no secret data is leaked.
        r = client.get(f"/api/runs/{run_id}/download/..%2F..%2Fsecret.json")
        if r.status_code == 200:
            # SPA catch-all served HTML — no data leakage
            assert "text/html" in r.headers.get("content-type", "")
        else:
            assert r.status_code in (400, 404)

    def test_path_traversal_dotdot(self, client, run_id):
        # Direct .. in filename should be caught by our handler
        r = client.get(f"/api/runs/{run_id}/download/summary_..test.json")
        assert r.status_code == 400

    def test_invalid_prefix(self, client, run_id):
        r = client.get(f"/api/runs/{run_id}/download/malicious_file.json")
        assert r.status_code == 400

    def test_invalid_suffix(self, client, run_id):
        r = client.get(f"/api/runs/{run_id}/download/summary_test.exe")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Sweeps (empty workspace, should return empty lists)
# ---------------------------------------------------------------------------

class TestSweeps:
    def test_list_sweeps_empty(self, client):
        r = client.get("/api/sweeps")
        assert r.status_code == 200
        data = r.json()
        assert data["sweeps"] == []
        assert data["total"] == 0
        assert data["schema_version"] == SCHEMA_VERSION

    def test_sweep_not_found(self, client):
        r = client.get("/api/sweeps/nonexistent/summary")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Monte Carlo (empty workspace, should return empty lists)
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_list_mc_empty(self, client):
        r = client.get("/api/mc")
        assert r.status_code == 200
        data = r.json()
        assert data["batches"] == []
        assert data["total"] == 0
        assert data["schema_version"] == SCHEMA_VERSION

    def test_mc_not_found(self, client):
        r = client.get("/api/mc/nonexistent/summary")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Reindex
# ---------------------------------------------------------------------------

class TestReindex:
    def test_reindex(self, client):
        r = client.post("/api/reindex")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "reindexed"
        assert data["runs"] >= 1


# ---------------------------------------------------------------------------
# Indexer unit tests
# ---------------------------------------------------------------------------

class TestIndexer:
    def test_workspace_indexer_finds_run(self, workspace, run_id):
        from stresslab.server.indexer import WorkspaceIndexer

        idx = WorkspaceIndexer(workspace)
        idx.reindex()
        runs = idx.list_runs()
        assert len(runs) >= 1
        ids = [r.run_id for r in runs]
        assert run_id in ids

    def test_resolve_run_path(self, workspace, run_id):
        from stresslab.server.indexer import WorkspaceIndexer

        idx = WorkspaceIndexer(workspace)
        path = idx.resolve_run_path(run_id)
        assert path is not None
        assert (path / f"summary_{run_id}.json").exists()

    def test_search_filter(self, workspace, run_id):
        from stresslab.server.indexer import WorkspaceIndexer

        idx = WorkspaceIndexer(workspace)
        # Search with first 8 chars should find it
        runs = idx.list_runs(q=run_id[:8])
        assert len(runs) >= 1

        # Search with garbage should find nothing
        runs = idx.list_runs(q="zzzzzzzzzzz_no_match")
        assert len(runs) == 0


# ---------------------------------------------------------------------------
# Parquet adapter unit tests
# ---------------------------------------------------------------------------

class TestParquetAdapter:
    def test_load_timeseries(self, workspace, run_id):
        from stresslab.server.parquet_adapter import load_timeseries

        ts_path = workspace / f"timeseries_{run_id}.parquet"
        df = load_timeseries(ts_path)
        assert "timestamp" in df.columns
        assert "pc_reference" in df.columns
        assert len(df) > 0

    def test_load_timeseries_column_select(self, workspace, run_id):
        from stresslab.server.parquet_adapter import load_timeseries

        ts_path = workspace / f"timeseries_{run_id}.parquet"
        df = load_timeseries(ts_path, columns=["pc_reference", "miss_distance"])
        assert "timestamp" in df.columns  # always included
        assert "pc_reference" in df.columns
        assert "miss_distance" in df.columns

    def test_downsample(self):
        import pandas as pd
        from stresslab.server.parquet_adapter import downsample

        df = pd.DataFrame({"timestamp": range(100), "value": range(100)})
        ds = downsample(df, max_points=10)
        assert len(ds) <= 15  # stride-based, not exact
        # First and last rows preserved
        assert ds.iloc[0]["timestamp"] == 0
        assert ds.iloc[-1]["timestamp"] == 99

    def test_downsample_no_op(self):
        import pandas as pd
        from stresslab.server.parquet_adapter import downsample

        df = pd.DataFrame({"timestamp": range(5), "value": range(5)})
        ds = downsample(df, max_points=10)
        assert len(ds) == 5  # unchanged

    def test_to_json_payload_nan_handling(self):
        import pandas as pd
        import numpy as np
        from stresslab.server.parquet_adapter import to_json_payload

        df = pd.DataFrame({
            "a": [1.0, float("nan"), float("inf")],
            "b": [10, 20, 30],
        })
        payload = to_json_payload(df)
        assert payload["columns"] == ["a", "b"]
        # NaN and inf should become None
        assert payload["rows"][1][0] is None
        assert payload["rows"][2][0] is None

    def test_read_parquet_metadata(self, workspace, run_id):
        from stresslab.server.parquet_adapter import read_parquet_metadata

        ts_path = workspace / f"timeseries_{run_id}.parquet"
        meta = read_parquet_metadata(ts_path)
        # Should have at least schema_version from our logging_engine
        assert isinstance(meta, dict)

    def test_to_arrow_ipc(self):
        import pandas as pd
        import pyarrow as pa
        from stresslab.server.parquet_adapter import to_arrow_ipc

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4, 5, 6]})
        ipc_bytes = to_arrow_ipc(df)
        assert isinstance(ipc_bytes, bytes)
        assert len(ipc_bytes) > 0

        # Verify it's valid Arrow IPC
        reader = pa.ipc.open_stream(ipc_bytes)
        table = reader.read_all()
        assert table.num_rows == 3
        assert "x" in table.column_names


# ---------------------------------------------------------------------------
# Label fields and scenario metadata tests (Phase 8)
# ---------------------------------------------------------------------------

class TestRunLabelFields:
    """Verify that run_label, scenario_title, and tags are surfaced."""

    def test_run_index_has_label_fields(self, client):
        """RunIndex in /api/runs should have label-related fields."""
        r = client.get("/api/runs")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1
        run = data["runs"][0]
        # Fields exist (may be null for runs without metadata)
        assert "run_label" in run
        assert "scenario_title" in run
        assert "tags" in run
        assert isinstance(run["tags"], list)

    def test_summary_has_scenario_metadata_keys(self, client, run_id):
        """Summary JSON should contain all scenario_metadata fields."""
        r = client.get(f"/api/runs/{run_id}/summary")
        assert r.status_code == 200
        summary = r.json()["summary"]
        for key in ["run_label", "scenario_title", "scenario_purpose",
                     "scenario_takeaway", "tags"]:
            assert key in summary, f"Missing key: {key}"

    def test_tags_is_list_in_summary(self, client, run_id):
        """Tags field in summary should be a list."""
        r = client.get(f"/api/runs/{run_id}/summary")
        summary = r.json()["summary"]
        assert isinstance(summary["tags"], list)


class TestScenarioMetadataInSummary:
    """Test that scenario_metadata round-trips through write_summary."""

    def test_metadata_embedded_when_provided(self, tmp_path):
        from stresslab.modules.logging_engine import LoggingEngine
        from stresslab.types import SimulationConfig

        logger = LoggingEngine()
        # Need at least one timestep
        from tests.test_schema_versioning import _make_timestep
        logger.record(_make_timestep(60.0))

        config = SimulationConfig()
        metadata = {
            "run_label": "test_label",
            "scenario_title": "Test Title",
            "scenario_purpose": "Testing purpose",
            "scenario_takeaway": "Testing takeaway",
            "tags": ["test", "unit"],
        }
        path = logger.write_summary(
            tmp_path, "test_run", config, None, None,
            scenario_metadata=metadata,
        )
        data = json.loads(path.read_text())
        assert data["run_label"] == "test_label"
        assert data["scenario_title"] == "Test Title"
        assert data["scenario_purpose"] == "Testing purpose"
        assert data["scenario_takeaway"] == "Testing takeaway"
        assert data["tags"] == ["test", "unit"]

    def test_metadata_defaults_when_none(self, tmp_path):
        from stresslab.modules.logging_engine import LoggingEngine
        from stresslab.types import SimulationConfig

        logger = LoggingEngine()
        from tests.test_schema_versioning import _make_timestep
        logger.record(_make_timestep(60.0))

        config = SimulationConfig()
        path = logger.write_summary(
            tmp_path, "test_run", config, None, None,
        )
        data = json.loads(path.read_text())
        assert data["run_label"] is None
        assert data["scenario_title"] is None
        assert data["scenario_purpose"] is None
        assert data["scenario_takeaway"] is None
        assert data["tags"] == []


class TestIndexerLabelSearch:
    """Test that the indexer can search by run_label and scenario_title."""

    def test_search_by_label_finds_nothing_for_default_run(self, workspace, run_id):
        """Default scenario has no run_label, so label search should not match."""
        from stresslab.server.indexer import WorkspaceIndexer

        idx = WorkspaceIndexer(workspace)
        idx.reindex()
        # Search for a label that does not exist
        runs = idx.list_runs(q="D0_nominal")
        found_ids = [r.run_id for r in runs]
        # Our test run was generated without metadata, so should not match
        assert run_id not in found_ids

