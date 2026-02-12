"""FastAPI application for the StressLAB local dashboard.

Serves a REST API for run indexing and artifact retrieval, plus the
built frontend as static files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from stresslab import __version__ as STRESSLAB_VERSION
from stresslab.metrics_contract import METRICS_CONTRACT_VERSION
from stresslab.modules.logging_engine import SCHEMA_VERSION
from stresslab.server.indexer import WorkspaceIndexer
from stresslab.server.parquet_adapter import (
    load_timeseries,
    read_parquet_metadata,
    downsample,
    to_json_payload,
    to_arrow_ipc,
)
from stresslab.server.schemas import (
    HealthResponse,
    RunListResponse,
    RunSummaryResponse,
    TimeseriesResponse,
    SweepListResponse,
    SweepSummaryResponse,
    MCListResponse,
    MCSummaryResponse,
)


def create_app(workspace: Path, dev_mode: bool = False) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        workspace: root directory to scan for artifacts.
        dev_mode: if True, enables CORS for Vite dev server.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="StressLAB Dashboard API",
        version=STRESSLAB_VERSION,
        description="Local REST API for browsing StressLAB simulation artifacts.",
    )

    indexer = WorkspaceIndexer(workspace)

    # CORS for Vite dev server
    if dev_mode:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:5174",
                "http://127.0.0.1:5174",
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/api/health", response_model=HealthResponse)
    def health():
        return HealthResponse()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    @app.get("/api/runs", response_model=RunListResponse)
    def list_runs(
        q: Optional[str] = Query(None, description="Search query"),
        limit: int = Query(200, ge=1, le=10000),
    ):
        runs = indexer.list_runs(q=q, limit=limit)
        return RunListResponse(runs=runs, total=len(runs))

    @app.get("/api/runs/{run_id}/summary", response_model=RunSummaryResponse)
    def get_run_summary(run_id: str):
        run_dir = indexer.resolve_run_path(run_id)
        if run_dir is None:
            raise HTTPException(404, f"Run not found: {run_id}")

        summary_path = run_dir / f"summary_{run_id}.json"
        if not summary_path.exists():
            raise HTTPException(404, f"Summary file not found for run: {run_id}")

        with open(summary_path) as f:
            data = json.load(f)

        return RunSummaryResponse(summary=data)

    @app.get("/api/runs/{run_id}/timeseries")
    def get_run_timeseries(
        run_id: str,
        format: str = Query("json", pattern="^(json|arrow)$"),
        cols: Optional[str] = Query(None, description="Comma-separated column names"),
        downsample_max: int = Query(5000, alias="downsample", ge=100, le=100000),
    ):
        run_dir = indexer.resolve_run_path(run_id)
        if run_dir is None:
            raise HTTPException(404, f"Run not found: {run_id}")

        ts_path = run_dir / f"timeseries_{run_id}.parquet"
        if not ts_path.exists():
            raise HTTPException(404, f"Timeseries file not found for run: {run_id}")

        columns = [c.strip() for c in cols.split(",") if c.strip()] if cols else None

        df = load_timeseries(ts_path, columns=columns)
        meta = read_parquet_metadata(ts_path)
        df = downsample(df, max_points=downsample_max)

        if format == "arrow":
            arrow_bytes = to_arrow_ipc(df)
            return Response(
                content=arrow_bytes,
                media_type="application/vnd.apache.arrow.stream",
                headers={
                    "X-Schema-Version": SCHEMA_VERSION,
                    "X-Contract-Version": METRICS_CONTRACT_VERSION,
                    "X-Code-Version": STRESSLAB_VERSION,
                },
            )

        payload = to_json_payload(df)
        return TimeseriesResponse(
            meta={
                **meta,
                "total_rows_before_downsample": len(df),
            },
            columns=payload["columns"],
            rows=payload["rows"],
        )

    # ------------------------------------------------------------------
    # Sweeps
    # ------------------------------------------------------------------

    @app.get("/api/sweeps", response_model=SweepListResponse)
    def list_sweeps():
        sweeps = indexer.list_sweeps()
        return SweepListResponse(sweeps=sweeps, total=len(sweeps))

    @app.get("/api/sweeps/{sweep_id}/summary", response_model=SweepSummaryResponse)
    def get_sweep_summary(sweep_id: str):
        sweep_dir = indexer.resolve_sweep_path(sweep_id)
        if sweep_dir is None:
            raise HTTPException(404, f"Sweep not found: {sweep_id}")

        sweep_json = sweep_dir / "sweep_summary.json"
        if not sweep_json.exists():
            raise HTTPException(404, f"Sweep summary not found: {sweep_id}")

        with open(sweep_json) as f:
            data = json.load(f)

        return SweepSummaryResponse(summary=data)

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    @app.get("/api/mc", response_model=MCListResponse)
    def list_mc():
        batches = indexer.list_mc_batches()
        return MCListResponse(batches=batches, total=len(batches))

    @app.get("/api/mc/{batch_id}/summary", response_model=MCSummaryResponse)
    def get_mc_summary(batch_id: str):
        mc_dir = indexer.resolve_mc_path(batch_id)
        if mc_dir is None:
            raise HTTPException(404, f"MC batch not found: {batch_id}")

        # Try stats JSON first, then CSV
        stats_path = mc_dir / "monte_carlo_stats.json"
        csv_path = mc_dir / "monte_carlo_summary.csv"

        data: dict = {}
        if stats_path.exists():
            with open(stats_path) as f:
                data = json.load(f)

        if csv_path.exists():
            import pandas as pd
            mc_df = pd.read_csv(csv_path)
            data["csv_rows"] = len(mc_df)
            data["csv_columns"] = mc_df.columns.tolist()

        if not data:
            raise HTTPException(404, f"No MC summary found for: {batch_id}")

        return MCSummaryResponse(summary=data)

    # ------------------------------------------------------------------
    # Reindex (dev only)
    # ------------------------------------------------------------------

    @app.post("/api/reindex")
    def reindex():
        indexer.reindex()
        return {
            "status": "reindexed",
            "runs": len(indexer.list_runs()),
            "sweeps": len(indexer.list_sweeps()),
            "mc_batches": len(indexer.list_mc_batches()),
        }

    # ------------------------------------------------------------------
    # Artifact download
    # ------------------------------------------------------------------

    @app.get("/api/runs/{run_id}/download/{filename}")
    def download_artifact(run_id: str, filename: str):
        """Download a raw artifact file (summary JSON or timeseries parquet)."""
        # Security: only allow specific filenames
        allowed_prefixes = ("summary_", "timeseries_")
        allowed_suffixes = (".json", ".parquet")
        if not any(filename.startswith(p) for p in allowed_prefixes):
            raise HTTPException(400, "Invalid filename prefix")
        if not any(filename.endswith(s) for s in allowed_suffixes):
            raise HTTPException(400, "Invalid filename suffix")
        # Prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(400, "Invalid filename")

        run_dir = indexer.resolve_run_path(run_id)
        if run_dir is None:
            raise HTTPException(404, f"Run not found: {run_id}")

        file_path = run_dir / filename
        if not file_path.exists():
            raise HTTPException(404, f"File not found: {filename}")

        content = file_path.read_bytes()
        media = "application/json" if filename.endswith(".json") else "application/octet-stream"
        return Response(
            content=content,
            media_type=media,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    # ------------------------------------------------------------------
    # Static file serving (built frontend)
    # ------------------------------------------------------------------

    # Try to serve built frontend from web/dist
    web_dist = Path(__file__).parent.parent.parent.parent / "web" / "dist"
    if web_dist.is_dir():
        # Serve index.html for SPA routes (catch-all)
        from fastapi.responses import FileResponse

        @app.get("/assets/{path:path}")
        def serve_asset(path: str):
            asset = web_dist / "assets" / path
            if asset.exists() and asset.is_file():
                return FileResponse(asset)
            raise HTTPException(404)

        @app.get("/{path:path}")
        def serve_spa(path: str):
            # Try exact file first
            file = web_dist / path
            if file.exists() and file.is_file():
                return FileResponse(file)
            # Fallback to index.html for SPA routing
            index = web_dist / "index.html"
            if index.exists():
                return FileResponse(index)
            raise HTTPException(404)

    return app
