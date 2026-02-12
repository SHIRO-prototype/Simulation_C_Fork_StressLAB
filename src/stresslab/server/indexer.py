"""Workspace Indexer — scans for run, sweep, and MC artifacts.

Walks the workspace directory looking for:
  - summary_*.json   (single runs)
  - sweep_summary.json (sweeps)
  - monte_carlo_summary.csv / monte_carlo_stats.json (MC batches)

Results are cached in workspace/.cache/index.json with mtime checks.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from stresslab.server.schemas import RunIndex, SweepIndex, MCBatchIndex, SanityFlags, ConfigSnapshot, RunStory


def _iso_mtime(p: Path) -> str:
    """Return ISO-8601 string of a file's modification time."""
    ts = os.path.getmtime(p)
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _safe_json_load(p: Path) -> dict:
    """Load JSON; return empty dict on any error."""
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


class WorkspaceIndexer:
    """Scans and caches workspace contents."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self._runs: list[RunIndex] = []
        self._sweeps: list[SweepIndex] = []
        self._mc_batches: list[MCBatchIndex] = []
        self._indexed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reindex(self) -> None:
        """Full workspace scan."""
        self._runs = []
        self._sweeps = []
        self._mc_batches = []
        self._scan_directory(self.workspace)
        # Sort runs newest-first
        self._runs.sort(key=lambda r: r.created_at or "", reverse=True)
        self._sweeps.sort(key=lambda s: s.created_at or "", reverse=True)
        self._mc_batches.sort(key=lambda b: b.created_at or "", reverse=True)
        self._indexed = True

    def ensure_indexed(self) -> None:
        if not self._indexed:
            self.reindex()

    def list_runs(
        self,
        q: Optional[str] = None,
        limit: int = 200,
    ) -> list[RunIndex]:
        """Return indexed runs, optionally filtered by query string."""
        self.ensure_indexed()
        results = self._runs
        if q:
            q_lower = q.lower()
            results = [
                r for r in results
                if q_lower in r.run_id.lower()
                or q_lower in (r.path or "").lower()
                or q_lower in (r.dynamics_model or "").lower()
            ]
        return results[:limit]

    def get_run(self, run_id: str) -> Optional[RunIndex]:
        self.ensure_indexed()
        for r in self._runs:
            if r.run_id == run_id:
                return r
        return None

    def list_sweeps(self) -> list[SweepIndex]:
        self.ensure_indexed()
        return self._sweeps

    def get_sweep(self, sweep_id: str) -> Optional[SweepIndex]:
        self.ensure_indexed()
        for s in self._sweeps:
            if s.sweep_id == sweep_id:
                return s
        return None

    def list_mc_batches(self) -> list[MCBatchIndex]:
        self.ensure_indexed()
        return self._mc_batches

    def get_mc_batch(self, batch_id: str) -> Optional[MCBatchIndex]:
        self.ensure_indexed()
        for b in self._mc_batches:
            if b.batch_id == batch_id:
                return b
        return None

    def resolve_run_path(self, run_id: str) -> Optional[Path]:
        """Return the absolute directory of a run, if found."""
        entry = self.get_run(run_id)
        if entry is None:
            return None
        return self.workspace / entry.path

    def resolve_sweep_path(self, sweep_id: str) -> Optional[Path]:
        entry = self.get_sweep(sweep_id)
        if entry is None:
            return None
        return self.workspace / entry.path

    def resolve_mc_path(self, batch_id: str) -> Optional[Path]:
        entry = self.get_mc_batch(batch_id)
        if entry is None:
            return None
        return self.workspace / entry.path

    # ------------------------------------------------------------------
    # Internal scanning
    # ------------------------------------------------------------------

    def _scan_directory(self, root: Path) -> None:
        """Recursively scan for artifacts."""
        if not root.is_dir():
            return

        # Check this directory itself for sweep/MC artifacts
        self._try_index_directory_artifacts(root)

        # Skip hidden directories and __pycache__
        for entry in sorted(root.iterdir()):
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            if entry.name == "node_modules":
                continue

            if entry.is_file():
                self._try_index_file(entry)
            elif entry.is_dir():
                # Always recurse to find individual run summaries and nested artifacts
                self._scan_directory(entry)

    def _try_index_directory_artifacts(self, directory: Path) -> None:
        """Check a directory for sweep and MC artifacts at this level."""
        sweep_json = directory / "sweep_summary.json"
        mc_csv = directory / "monte_carlo_summary.csv"
        mc_stats = directory / "monte_carlo_stats.json"

        if sweep_json.exists():
            self._index_sweep(directory, sweep_json)
        if mc_csv.exists() or mc_stats.exists():
            self._index_mc_batch(
                directory, mc_stats if mc_stats.exists() else mc_csv
            )

    def _try_index_file(self, path: Path) -> None:
        """Index a single file if it's a run summary."""
        if path.name.startswith("summary_") and path.suffix == ".json":
            self._index_run(path)

    def _index_run(self, summary_path: Path) -> None:
        """Index a single run from its summary JSON."""
        data = _safe_json_load(summary_path)
        if not data:
            return

        run_id = data.get("run_id")
        if not run_id:
            # Try extracting from filename: summary_<run_id>.json
            stem = summary_path.stem
            if stem.startswith("summary_"):
                run_id = stem[len("summary_"):]
        if not run_id:
            return

        # Relative path to the directory containing the summary
        try:
            rel_path = summary_path.parent.relative_to(self.workspace)
        except ValueError:
            rel_path = summary_path.parent

        # Extract sanity and config sub-dicts if present
        sanity_raw = data.get("sanity")
        sanity = SanityFlags(**sanity_raw) if isinstance(sanity_raw, dict) else None
        config_raw = data.get("config")
        config_snap = ConfigSnapshot(**config_raw) if isinstance(config_raw, dict) else None
        story_raw = data.get("story")
        story = RunStory(**story_raw) if isinstance(story_raw, dict) else None

        self._runs.append(RunIndex(
            run_id=run_id,
            path=str(rel_path).replace("\\", "/"),
            created_at=_iso_mtime(summary_path),
            seed=data.get("seed"),
            dynamics_model=data.get("dynamics_model"),
            threshold_v1_trigger_time=data.get("threshold_v1_trigger_time"),
            integrity_v1_trigger_time=data.get("integrity_v1_trigger_time"),
            decision_compression_window=data.get("decision_compression_window"),
            false_safe_rate=data.get("false_safe_rate"),
            false_alert_rate=data.get("false_alert_rate"),
            max_pc_degraded=data.get("max_pc_degraded"),
            max_staleness=data.get("max_staleness"),
            decision_instability_index=data.get("decision_instability_index"),
            total_timesteps=data.get("total_timesteps"),
            outage_sensitivity_score=data.get("outage_sensitivity_score"),
            max_pc_reference=data.get("max_pc_reference"),
            max_cov_trace=data.get("max_cov_trace"),
            decision_transitions_per_hour=data.get("decision_transitions_per_hour"),
            decision_entropy=data.get("decision_entropy"),
            mean_pc_drift=data.get("mean_pc_drift"),
            max_pc_drift=data.get("max_pc_drift"),
            staleness_pc_correlation=data.get("staleness_pc_correlation"),
            mean_freshness=data.get("mean_freshness"),
            min_freshness=data.get("min_freshness"),
            sanity=sanity,
            config=config_snap,
            story=story,
        ))

    def _index_sweep(self, sweep_dir: Path, sweep_json: Path) -> None:
        """Index a sweep directory."""
        data = _safe_json_load(sweep_json)

        # Use directory name as sweep_id
        sweep_id = sweep_dir.name

        try:
            rel_path = sweep_dir.relative_to(self.workspace)
        except ValueError:
            rel_path = sweep_dir

        self._sweeps.append(SweepIndex(
            sweep_id=sweep_id,
            path=str(rel_path).replace("\\", "/"),
            sweep_param=data.get("sweep_param"),
            n_values=len(data.get("sweep_values", [])),
            seed=data.get("seed"),
            created_at=_iso_mtime(sweep_json),
        ))

    def _index_mc_batch(self, mc_dir: Path, artifact: Path) -> None:
        """Index a Monte Carlo batch directory."""
        batch_id = mc_dir.name

        try:
            rel_path = mc_dir.relative_to(self.workspace)
        except ValueError:
            rel_path = mc_dir

        data: dict = {}
        stats_path = mc_dir / "monte_carlo_stats.json"
        if stats_path.exists():
            data = _safe_json_load(stats_path)

        self._mc_batches.append(MCBatchIndex(
            batch_id=batch_id,
            path=str(rel_path).replace("\\", "/"),
            n_runs=data.get("total_runs"),
            successful_runs=data.get("successful_runs"),
            created_at=_iso_mtime(artifact),
        ))
