"""FastAPI application for the StressLAB local dashboard.

Serves a REST API for run indexing and artifact retrieval, plus the
built frontend as static files.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from stresslab import __version__ as STRESSLAB_VERSION
from stresslab.metrics_contract import METRICS_CONTRACT_VERSION
from stresslab.modules.logging_engine import SCHEMA_VERSION
from stresslab.modules.scenario_generator import export_scenario
from stresslab.modules.simulation_runner import run_simulation
from stresslab.modules.synth_case import generate_synthetic_case
from stresslab.server.cases import (
    CaseRepository,
    CaseSnapshot,
    StressKnobs,
    compare_from_summaries,
)
from stresslab.server.indexer import WorkspaceIndexer
from stresslab.server.parquet_adapter import (
    load_timeseries,
    read_parquet_metadata,
    downsample,
    resample_uniform,
    to_json_payload,
    to_arrow_ipc,
)
from stresslab.server.schemas import (
    CaseDetailResponse,
    CaseListResponse,
    CompareResponse,
    HealthResponse,
    RunListResponse,
    RunSummaryResponse,
    TimeseriesResponse,
    SweepListResponse,
    SweepSummaryResponse,
    MCListResponse,
    MCSummaryResponse,
)


def create_app(workspace: Path, dev_mode: bool = False, enable_launch: bool = False) -> FastAPI:
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
    case_repo = CaseRepository(workspace)
    run_status: dict[str, dict] = {}

    def _normalize_snapshot_payload(payload: dict) -> dict:
        if isinstance(payload.get("snapshot"), dict):
            return payload["snapshot"]
        if "state_obj1" in payload and "state_obj2" in payload:
            return payload
        raw_text = payload.get("raw_text")
        if isinstance(raw_text, str):
            source_format = str(payload.get("source_format", "auto")).lower()
            if source_format in {"json", "auto"}:
                try:
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            if source_format in {"yaml", "yml", "auto"}:
                try:
                    import yaml  # type: ignore

                    parsed = yaml.safe_load(raw_text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
        raise ValueError("Could not parse snapshot payload; provide snapshot dict or raw_text in JSON/YAML format")

    def _load_run_summary(run_id: str) -> dict:
        run_dir = indexer.resolve_run_path(run_id)
        if run_dir is None:
            raise HTTPException(404, f"Run not found: {run_id}")
        summary_path = run_dir / f"summary_{run_id}.json"
        if not summary_path.exists():
            raise HTTPException(404, f"Summary file not found for run: {run_id}")
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)

    def _run_tca_preview(run_id: str, epoch_utc: Optional[str]) -> dict:
        run_dir = indexer.resolve_run_path(run_id)
        if run_dir is None:
            return {"tca_utc": None, "miss_distance_at_tca": None, "relative_velocity_mps": None}
        ts_path = run_dir / f"timeseries_{run_id}.parquet"
        if not ts_path.exists():
            return {"tca_utc": None, "miss_distance_at_tca": None, "relative_velocity_mps": None}

        try:
            df = pd.read_parquet(
                ts_path,
                columns=["timestamp", "miss_distance", "time_to_tca", "rel_velocity_at_tca"],
            )
        except Exception:
            return {"tca_utc": None, "miss_distance_at_tca": None, "relative_velocity_mps": None}

        if len(df) == 0:
            return {"tca_utc": None, "miss_distance_at_tca": None, "relative_velocity_mps": None}

        if "time_to_tca" in df.columns:
            idx = (df["time_to_tca"].abs()).idxmin()
        else:
            idx = df["miss_distance"].idxmin()

        t_offset_s = float(df.loc[idx, "timestamp"])
        miss_km = float(df.loc[idx, "miss_distance"]) if "miss_distance" in df.columns else None
        rel_v_kms = float(df.loc[idx, "rel_velocity_at_tca"]) if "rel_velocity_at_tca" in df.columns else None

        tca_utc = None
        if epoch_utc:
            try:
                epoch = datetime.fromisoformat(str(epoch_utc).replace("Z", "+00:00"))
                tca_utc = (epoch + timedelta(seconds=t_offset_s)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            except Exception:
                tca_utc = None

        return {
            "tca_utc": tca_utc,
            "miss_distance_at_tca": miss_km,
            "relative_velocity_mps": rel_v_kms * 1000.0 if rel_v_kms is not None else None,
        }

    def _interp_linear(x_in: np.ndarray, y_in: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
        m = np.isfinite(x_in) & np.isfinite(y_in)
        x = x_in[m]
        y = y_in[m]
        if len(x) < 2:
            return np.full_like(x_grid, np.nan, dtype=float)
        order = np.argsort(x)
        return np.interp(x_grid, x[order], y[order], left=np.nan, right=np.nan)

    def _interp_step_prev(x_in: np.ndarray, y_in: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
        m = np.isfinite(x_in) & np.isfinite(y_in)
        x = x_in[m]
        y = y_in[m]
        if len(x) == 0:
            return np.full_like(x_grid, np.nan, dtype=float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        idx = np.searchsorted(x, x_grid, side="right") - 1
        idx = np.clip(idx, 0, len(y) - 1)
        out = y[idx]
        out[x_grid < x[0]] = np.nan
        return out

    def _growth_rate_smoothed(trace_series: np.ndarray, x_hours: np.ndarray, window_points: int = 9) -> np.ndarray:
        n = len(trace_series)
        if n < 5:
            return np.full(n, np.nan)
        w = max(5, min(window_points, n if n % 2 == 1 else n - 1))
        k = w // 2
        y = np.log(np.clip(trace_series, 1e-20, None))
        out = np.full(n, np.nan)
        for i in range(n):
            lo = max(0, i - k)
            hi = min(n, i + k + 1)
            if hi - lo < 5:
                continue
            xw = x_hours[lo:hi]
            yw = y[lo:hi]
            if np.nanmax(xw) - np.nanmin(xw) < 1e-9:
                continue
            coef = np.polyfit(xw, yw, 1)
            out[i] = coef[0]
        return out

    def _panel_series_for_run(run_id: str, x_grid: np.ndarray, horizon_hours: float, pc_floor: float = 1e-16) -> dict:
        run_dir = indexer.resolve_run_path(run_id)
        if run_dir is None:
            raise HTTPException(404, f"Run not found: {run_id}")
        ts_path = run_dir / f"timeseries_{run_id}.parquet"
        if not ts_path.exists():
            raise HTTPException(404, f"Timeseries file not found for run: {run_id}")

        summary_path = run_dir / f"summary_{run_id}.json"
        run_label = None
        if summary_path.exists():
            try:
                with open(summary_path, encoding="utf-8") as f:
                    run_label = (json.load(f) or {}).get("run_label")
            except Exception:
                run_label = None

        summary_path = run_dir / f"summary_{run_id}.json"
        run_label = None
        if summary_path.exists():
            try:
                with open(summary_path, encoding="utf-8") as f:
                    run_label = (json.load(f) or {}).get("run_label")
            except Exception:
                run_label = None

        needed = [
            "time_to_tca",
            "pc_reference",
            "pc_degraded",
            "staleness_obj1",
            "cov_trace_obj1",
            "cov_trace_obj2",
            "miss_distance",
            "rel_velocity_at_tca",
            "threshold_v1_alert_state",
            "integrity_v1_state",
            "measurement_applied_obj1",
            "measurement_applied_obj2",
        ]
        df = load_timeseries(ts_path, columns=needed)
        if "time_to_tca" not in df.columns:
            raise HTTPException(400, f"Run {run_id} missing time_to_tca")

        x = (df["time_to_tca"].to_numpy(dtype=float) / 3600.0)
        mask = np.isfinite(x) & (x >= 0.0) & (x <= horizon_hours)
        if np.count_nonzero(mask) < 2:
            mask = np.isfinite(x)
        x = x[mask]
        sub = df.loc[mask]

        # Keep last duplicate x.
        dedup = pd.DataFrame({"x": x})
        dedup["idx"] = np.arange(len(dedup))
        dedup = dedup.sort_values(["x", "idx"]).drop_duplicates(subset=["x"], keep="last")
        keep_idx = dedup["idx"].to_numpy(dtype=int)
        x = x[keep_idx]
        sub = sub.iloc[keep_idx]

        pc_ref = _interp_linear(x, sub.get("pc_reference", pd.Series(dtype=float)).to_numpy(dtype=float), x_grid)
        pc_deg = _interp_linear(x, sub.get("pc_degraded", pd.Series(dtype=float)).to_numpy(dtype=float), x_grid)
        pc_ref = np.where(np.isfinite(pc_ref), np.maximum(pc_ref, pc_floor), np.nan)
        pc_deg = np.where(np.isfinite(pc_deg), np.maximum(pc_deg, pc_floor), np.nan)

        staleness = _interp_step_prev(x, sub.get("staleness_obj1", pd.Series(dtype=float)).to_numpy(dtype=float), x_grid)
        cov1 = _interp_linear(x, sub.get("cov_trace_obj1", pd.Series(dtype=float)).to_numpy(dtype=float), x_grid)
        cov2 = _interp_linear(x, sub.get("cov_trace_obj2", pd.Series(dtype=float)).to_numpy(dtype=float), x_grid)
        cov_combined = cov1 + cov2
        miss_m = _interp_linear(x, sub.get("miss_distance", pd.Series(dtype=float)).to_numpy(dtype=float), x_grid) * 1000.0
        rel_v_mps = _interp_linear(x, sub.get("rel_velocity_at_tca", pd.Series(dtype=float)).to_numpy(dtype=float), x_grid) * 1000.0

        def _state_to_numeric(series: pd.Series) -> np.ndarray:
            if pd.api.types.is_numeric_dtype(series):
                return series.to_numpy(dtype=float)
            mapping = {
                "safe": 0.0,
                "monitor": 1.0,
                "watch": 2.0,
                "warning": 3.0,
                "critical": 4.0,
                "alert": 4.0,
            }
            out = []
            for v in series.to_numpy():
                if v is None:
                    out.append(np.nan)
                    continue
                key = str(v).strip().lower()
                out.append(mapping.get(key, np.nan))
            return np.asarray(out, dtype=float)

        th_raw = _state_to_numeric(sub.get("threshold_v1_alert_state", pd.Series(dtype=float)))
        int_raw = _state_to_numeric(sub.get("integrity_v1_state", pd.Series(dtype=float)))
        th_state = _interp_step_prev(x, th_raw, x_grid)
        int_state = _interp_step_prev(x, int_raw, x_grid)
        meas1 = _interp_step_prev(x, sub.get("measurement_applied_obj1", pd.Series(dtype=float)).to_numpy(dtype=float), x_grid)
        meas2 = _interp_step_prev(x, sub.get("measurement_applied_obj2", pd.Series(dtype=float)).to_numpy(dtype=float), x_grid)
        outage_flag = np.where((meas1 < 0.5) & (meas2 < 0.5), 1.0, 0.0)

        growth_rate = _growth_rate_smoothed(cov_combined, x_grid)

        def _safe(arr: np.ndarray) -> list:
            out = []
            for v in arr.tolist():
                if v is None:
                    out.append(None)
                elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    out.append(None)
                else:
                    out.append(float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            return out

        return {
            "run_id": run_id,
            "pc_reference": _safe(pc_ref),
            "pc_degraded": _safe(pc_deg),
            "staleness_seconds": _safe(staleness),
            "cov_trace_combined": _safe(cov_combined),
            "cov_growth_rate_per_hour": _safe(growth_rate),
            "miss_distance_m": _safe(miss_m),
            "relative_speed_mps": _safe(rel_v_mps),
            "decision_state_threshold_v1": _safe(th_state),
            "decision_state_integrity_v1": _safe(int_state),
            "outage_flag": _safe(outage_flag),
        }

    def _trigger_hours_before_tca(run_id: str, trigger_time_s: Optional[float]) -> Optional[float]:
        if trigger_time_s is None:
            return None
        run_dir = indexer.resolve_run_path(run_id)
        if run_dir is None:
            return None
        ts_path = run_dir / f"timeseries_{run_id}.parquet"
        if not ts_path.exists():
            return None
        try:
            df = load_timeseries(ts_path, columns=["timestamp", "time_to_tca"])
            if len(df) == 0 or "timestamp" not in df.columns or "time_to_tca" not in df.columns:
                return None
            ts = df["timestamp"].to_numpy(dtype=float)
            ttc = df["time_to_tca"].to_numpy(dtype=float)
            m = np.isfinite(ts) & np.isfinite(ttc)
            if np.count_nonzero(m) < 2:
                return None
            ts = ts[m]
            ttc = ttc[m]
            order = np.argsort(ts)
            ts = ts[order]
            ttc = ttc[order]
            h = float(np.interp(float(trigger_time_s), ts, ttc) / 3600.0)
            return h if np.isfinite(h) else None
        except Exception:
            return None

    def _stress_snapshot_to_case_snapshot(snapshot: dict) -> dict:
        case_id = str(snapshot.get("case_id") or "").strip()
        epoch_utc = str(snapshot.get("epoch_utc") or "").strip()
        primary = snapshot.get("primary") or {}
        secondary = snapshot.get("secondary") or {}
        if not case_id:
            raise ValueError("case_id is required")
        if not epoch_utc:
            raise ValueError("epoch_utc is required")
        if not isinstance(primary, dict) or not isinstance(secondary, dict):
            raise ValueError("primary and secondary must be objects")

        p_state = np.asarray(primary.get("state_eci_km_kms"), dtype=float)
        s_state = np.asarray(secondary.get("state_eci_km_kms"), dtype=float)
        p_cov = np.asarray(primary.get("cov_eci_6x6"), dtype=float)
        s_cov = np.asarray(secondary.get("cov_eci_6x6"), dtype=float)

        if p_state.shape != (6,) or s_state.shape != (6,):
            raise ValueError("primary/secondary state_eci_km_kms must be length=6")
        if p_cov.shape != (6, 6) or s_cov.shape != (6, 6):
            raise ValueError("primary/secondary cov_eci_6x6 must be shape=6x6")

        eps = 1e-10
        for name, cov in (("primary", p_cov), ("secondary", s_cov)):
            if not np.allclose(cov, cov.T, atol=eps):
                raise ValueError(f"{name} covariance symmetry check failed")
            eig = np.linalg.eigvalsh(cov)
            if float(np.min(eig)) < -eps:
                raise ValueError(f"{name} covariance PSD check failed")

        for name, state in (("primary", p_state), ("secondary", s_state)):
            rmag = float(np.linalg.norm(state[:3]))
            vmag = float(np.linalg.norm(state[3:]))
            if not (6000.0 <= rmag <= 80000.0):
                raise ValueError(f"{name} |r| out of bounds: {rmag:.2f} km")
            if not (0.0 <= vmag <= 20.0):
                raise ValueError(f"{name} |v| out of bounds: {vmag:.4f} km/s")

        meta = snapshot.get("meta") if isinstance(snapshot.get("meta"), dict) else {}
        target_tca_h = float(meta.get("target_tca_hours", 4.0))
        horizon_s = float(meta.get("horizon_seconds", 172800.0))
        t_end = float(min(max(target_tca_h * 3600.0 * 3.0, 21600.0), max(21600.0, horizon_s)))

        return {
            "state_obj1": p_state.tolist(),
            "state_obj2": s_state.tolist(),
            "cov_obj1": p_cov.tolist(),
            "cov_obj2": s_cov.tolist(),
            "seed": 42,
            "dynamics_model": "two_body_plus_J2",
            "t_start": 0.0,
            "t_end": t_end,
            "dt": 60.0,
            "combined_hard_body_radius": 0.02,
            "measurement_update_interval": 60.0,
            "measurement_noise_sigma_pos": 0.01,
            "process_noise_scale": 1.0,
            "process_noise_sigma_radial": 1e-9,
            "process_noise_sigma_tangential": 1e-9,
            "process_noise_sigma_normal": 1e-9,
            "threshold_pc": 1e-4,
            "threshold_time_to_tca_gate": 86400.0,
            "threshold_miss_distance": 1.0,
            "_operator_case_id": case_id,
            "_operator_epoch_utc": epoch_utc,
        }

    def _extract_stress_snapshot(payload: dict) -> dict:
        raw_snapshot = payload.get("snapshot")
        if isinstance(raw_snapshot, dict):
            return raw_snapshot

        raw_text = payload.get("raw_text")
        if isinstance(raw_text, str) and raw_text.strip():
            source_format = str(payload.get("source_format", "auto")).lower()
            if source_format in {"json", "auto"}:
                try:
                    parsed_json = json.loads(raw_text)
                    if isinstance(parsed_json, dict):
                        return parsed_json
                except Exception:
                    pass
            if source_format in {"yaml", "yml", "auto"}:
                try:
                    import yaml  # type: ignore

                    parsed_yaml = yaml.safe_load(raw_text)
                    if isinstance(parsed_yaml, dict):
                        return parsed_yaml
                except Exception:
                    pass
            raise ValueError("Unable to parse raw_text as JSON/YAML CaseSnapshot")

        if isinstance(payload, dict):
            return payload
        raise ValueError("Invalid payload")

    def _run_case_mode(case_id: str, mode: str, config, scenario_metadata: dict) -> dict:
        status_key = f"{case_id}:{mode}"
        run_status[status_key] = {"status": "queued", "mode": mode, "case_id": case_id}
        run_status[status_key]["status"] = "running"
        out_dir = case_repo.case_dir(case_id) / "runs" / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        export_scenario(config, out_dir)
        result = run_simulation(config, output_dir=out_dir, verbose=False, scenario_metadata=scenario_metadata)
        run_status[status_key] = {
            "status": "done",
            "mode": mode,
            "case_id": case_id,
            "run_id": result["run_id"],
        }
        if mode == "baseline":
            case_repo.update_case(case_id, baseline_run_id=result["run_id"])
        elif mode == "stress":
            case_repo.update_case(case_id, stress_run_id=result["run_id"])
        indexer.reindex()
        return result

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

    @app.post("/api/stress/validate")
    def stress_validate(payload: dict = Body(...)):
        snapshot = _extract_stress_snapshot(payload)
        try:
            case_snapshot = _stress_snapshot_to_case_snapshot(snapshot)
            normalized = {
                "case_id": case_snapshot.pop("_operator_case_id"),
                "epoch_utc": case_snapshot.pop("_operator_epoch_utc"),
                "primary": {
                    "state_eci_km_kms": case_snapshot["state_obj1"],
                    "cov_eci_6x6": case_snapshot["cov_obj1"],
                },
                "secondary": {
                    "state_eci_km_kms": case_snapshot["state_obj2"],
                    "cov_eci_6x6": case_snapshot["cov_obj2"],
                },
            }
            return {"ok": True, "errors": [], "normalized": normalized}
        except Exception as e:
            return {"ok": False, "errors": [str(e)]}

    @app.post("/api/stress/case")
    def stress_case(payload: dict = Body(...)):
        snapshot = _extract_stress_snapshot(payload)
        try:
            mapped = _stress_snapshot_to_case_snapshot(snapshot)
            operator_case_id = mapped.pop("_operator_case_id")
            operator_epoch_utc = mapped.pop("_operator_epoch_utc")
            cs = CaseSnapshot.from_dict(mapped)
            item = case_repo.create_case(
                cs,
                source="operator",
                notes=f"case_id={operator_case_id} epoch={operator_epoch_utc}",
                case_id_override=operator_case_id,
            )
            case_repo.update_case(item["case_id"], operator_case_id=operator_case_id, epoch_utc=operator_epoch_utc)
            return {"ok": True, "case_id": item["case_id"]}
        except Exception as e:
            return {"ok": False, "errors": [str(e)]}

    @app.post("/api/stress/synth-case")
    def stress_synth_case(payload: dict = Body(...)):
        mode = str(payload.get("mode", "A"))
        seed = int(payload.get("seed", 42))
        case_id = str(payload.get("case_id", f"OPS_SYNTH_{seed:04d}"))
        target_miss_m = float(payload.get("target_miss_m", 200.0))
        target_tca_hours = float(payload.get("target_tca_hours", 4.0))
        target_rel_speed_mps = float(payload.get("target_rel_speed_mps", 10000.0))
        sig = payload.get("sigmas") if isinstance(payload.get("sigmas"), dict) else {}

        result = generate_synthetic_case(
            case_id=case_id,
            mode=mode,
            seed=seed,
            target_miss_m=target_miss_m,
            target_tca_hours=target_tca_hours,
            target_rel_speed_mps=target_rel_speed_mps,
            primary_pos_sigma_m=float(sig.get("primary_pos_sigma_m", payload.get("primary_pos_sigma_m", 20.0))),
            secondary_pos_sigma_m=float(sig.get("secondary_pos_sigma_m", payload.get("secondary_pos_sigma_m", 100.0))),
            primary_vel_sigma_mps=float(sig.get("primary_vel_sigma_mps", payload.get("primary_vel_sigma_mps", 0.02))),
            secondary_vel_sigma_mps=float(sig.get("secondary_vel_sigma_mps", payload.get("secondary_vel_sigma_mps", 0.10))),
            hard_body_radius_m=float(payload.get("hbr_m", 5.0)),
            horizon_seconds=float(payload.get("horizon_seconds", 172800.0)),
        )

        synth_snapshot = result["case_snapshot"]
        case_res = stress_case({"snapshot": synth_snapshot})
        if not case_res.get("ok"):
            return case_res
        return {
            "case_id": case_res.get("case_id"),
            "case_snapshot": synth_snapshot,
            "preview": result["preview"],
        }

    @app.post("/api/stress/{case_id}/baseline")
    def stress_baseline(case_id: str):
        out = run_case_baseline(case_id)
        if out.get("status") != "done":
            return out
        run_id = out["run_id"]
        summary = _load_run_summary(run_id)
        case_item = case_repo.get_case(case_id)
        tca_preview = _run_tca_preview(run_id, case_item.get("epoch_utc") if isinstance(case_item, dict) else None)
        return {
            "baseline_run_id": run_id,
            "baseline_summary": {
                "tca_utc": tca_preview.get("tca_utc"),
                "miss_distance_at_tca": tca_preview.get("miss_distance_at_tca"),
                "relative_velocity_mps": tca_preview.get("relative_velocity_mps"),
                "max_pc_baseline": summary.get("max_pc_degraded"),
                "threshold_trigger": summary.get("threshold_v1_trigger_time"),
                "integrity_trigger": summary.get("integrity_v1_trigger_time"),
            },
        }

    @app.post("/api/stress/{case_id}/stress")
    def stress_run(case_id: str, payload: dict = Body(...)):
        knobs = payload.get("knobs") if isinstance(payload.get("knobs"), dict) else payload
        if not isinstance(knobs, dict):
            raise HTTPException(400, "knobs payload must be an object")
        try:
            tca_ref = float(case_repo.load_snapshot(case_id).baseline_config().t_end)
        except Exception:
            tca_ref = 259200.0
        mapped_knobs = {
            "measurement_cadence_s": int(knobs.get("measurement_cadence_seconds", 60)),
            "outage_windows": [],
            "process_noise_scale": float(knobs.get("process_noise_scale", 1.0)),
            "maneuver_enabled": False,
            "maneuver_delta_v_sigma": 0.001,
            "maneuver_execution_time": 0.0,
        }
        for i, w in enumerate(knobs.get("outage_windows", [])):
            if not isinstance(w, dict):
                raise HTTPException(400, f"outage_windows[{i}] must be an object")
            start_h = float(w.get("start_hours_before_tca", 3.0))
            dur_h = float(w.get("duration_hours", 2.0))
            start = max(0.0, tca_ref - start_h * 3600.0)
            end = min(tca_ref, start + max(0.0, dur_h * 3600.0))
            if end > start:
                mapped_knobs["outage_windows"].append({"start": start, "end": end})

        out = run_case_stress(case_id, mapped_knobs)
        if out.get("status") != "done":
            return out
        run_id = out["run_id"]
        case_item = case_repo.get_case(case_id) or {}
        baseline_run_id = case_item.get("baseline_run_id")
        if baseline_run_id and str(run_id) == str(baseline_run_id):
            raise HTTPException(500, "run_id collision")
        summary = _load_run_summary(run_id)
        cadence_seconds = int(mapped_knobs.get("measurement_cadence_s", 60))
        outage_windows_req = knobs.get("outage_windows", []) if isinstance(knobs.get("outage_windows", []), list) else []
        max_staleness_seconds = summary.get("max_staleness")
        if max_staleness_seconds is None:
            run_dir = indexer.resolve_run_path(run_id)
            if run_dir is not None:
                ts_path = run_dir / f"timeseries_{run_id}.parquet"
                if ts_path.exists():
                    try:
                        ts_df = load_timeseries(ts_path, columns=["staleness_obj1", "staleness_obj2"])
                        cand = []
                        for c in ["staleness_obj1", "staleness_obj2"]:
                            if c in ts_df.columns:
                                cand.append(float(np.nanmax(ts_df[c].to_numpy(dtype=float))))
                        if cand:
                            max_staleness_seconds = max(cand)
                    except Exception:
                        pass
        return {
            "stress_run_id": run_id,
            "stress_summary": {
                "measurement_cadence_seconds": cadence_seconds,
                "outage_windows": outage_windows_req,
                "process_noise_scale": mapped_knobs.get("process_noise_scale"),
                "max_staleness_seconds": max_staleness_seconds,
                "max_staleness": max_staleness_seconds,
                "max_pc_stress": summary.get("max_pc_degraded"),
                "threshold_trigger": summary.get("threshold_v1_trigger_time"),
                "integrity_trigger": summary.get("integrity_v1_trigger_time"),
                "sanity_outage_vs_staleness": (
                    bool(any(float(w.get("duration_hours", 0) or 0) > 0 for w in outage_windows_req if isinstance(w, dict)))
                    and float(max_staleness_seconds or 0.0) > float(cadence_seconds)
                ),
            },
        }

    @app.get("/api/stress/{case_id}/compare")
    def stress_compare(case_id: str):
        cmp = get_case_compare(case_id).compare
        baseline_run_id = cmp.get("baseline_run_id")
        stress_run_id = cmp.get("stress_run_id")
        if baseline_run_id and stress_run_id and str(baseline_run_id) == str(stress_run_id):
            raise HTTPException(500, "run_id collision")
        return {
            "case_id": case_id,
            "baseline_run_id": baseline_run_id,
            "stress_run_id": stress_run_id,
            "deltas": cmp.get("deltas", {}),
            "links": {"compare_ui": f"/cases/{case_id}/compare"},
        }

    @app.get("/api/stress/{case_id}/panel-data")
    def stress_panel_data(
        case_id: str,
        horizon_hours: float = Query(6.0, ge=0.5, le=72.0),
        max_points: int = Query(300, ge=50, le=500),
    ):
        item = case_repo.get_case(case_id)
        if item is None:
            raise HTTPException(404, f"Case not found: {case_id}")
        baseline_run_id = item.get("baseline_run_id")
        stress_run_id = item.get("stress_run_id")
        if not baseline_run_id:
            raise HTTPException(400, "Baseline run is required")
        if stress_run_id and str(baseline_run_id) == str(stress_run_id):
            raise HTTPException(500, "run_id collision")

        x_grid = np.linspace(0.0, float(horizon_hours), num=max_points)
        baseline = _panel_series_for_run(baseline_run_id, x_grid, horizon_hours=float(horizon_hours))
        stress = _panel_series_for_run(stress_run_id, x_grid, horizon_hours=float(horizon_hours)) if stress_run_id else None
        baseline_dir = indexer.resolve_run_path(baseline_run_id)
        stress_dir = indexer.resolve_run_path(stress_run_id) if stress_run_id else None

        baseline_summary = _load_run_summary(baseline_run_id)
        stress_summary = _load_run_summary(stress_run_id) if stress_run_id else None
        baseline_trigger_h = _trigger_hours_before_tca(
            baseline_run_id,
            baseline_summary.get("threshold_v1_trigger_time") if isinstance(baseline_summary, dict) else None,
        )
        shiro_trigger_h = _trigger_hours_before_tca(
            baseline_run_id,
            baseline_summary.get("integrity_v1_trigger_time") if isinstance(baseline_summary, dict) else None,
        )

        return {
            "case_id": case_id,
            "x_axis": {
                "name": "hours_before_tca",
                "grid": x_grid.tolist(),
                "direction": "countdown_to_zero",
            },
            "meta": {
                "sorted": True,
                "resampled": True,
                "n_out": int(max_points),
                "pc_floor": 1e-16,
                "baseline_run_id": baseline_run_id,
                "stress_run_id": stress_run_id,
                "baseline_loaded_path": str((baseline_dir / f"timeseries_{baseline_run_id}.parquet") if baseline_dir else ""),
                "stress_loaded_path": str((stress_dir / f"timeseries_{stress_run_id}.parquet") if (stress_dir and stress_run_id) else ""),
                "units": {
                    "x": "hours",
                    "miss_distance": "m",
                    "relative_speed": "m/s",
                    "staleness": "s",
                    "cov_trace": "km^2",
                },
            },
            "overlays": {
                "threshold_lines": [1e-4, 1e-5, 1e-6],
                "trigger_markers": {
                    "baseline_trigger_time": baseline_trigger_h,
                    "shiro_trigger_time": shiro_trigger_h,
                },
            },
            "baseline": baseline,
            "stress": stress,
            "summary": {
                "baseline": baseline_summary,
                "stress": stress_summary,
            },
        }

    @app.get("/api/cases", response_model=CaseListResponse)
    def list_cases():
        cases = case_repo.list_cases()
        return CaseListResponse(cases=cases, total=len(cases))

    @app.get("/api/cases/{case_id}", response_model=CaseDetailResponse)
    def get_case(case_id: str):
        item = case_repo.get_case(case_id)
        if item is None:
            raise HTTPException(404, f"Case not found: {case_id}")
        snapshot = case_repo.load_snapshot(case_id).to_dict()
        return CaseDetailResponse(case=item, snapshot=snapshot)

    @app.post("/api/cases/validate")
    def validate_case(payload: dict = Body(...)):
        try:
            snapshot = CaseSnapshot.from_dict(_normalize_snapshot_payload(payload))
            return {"valid": True, "snapshot": snapshot.to_dict()}
        except Exception as e:
            raise HTTPException(400, f"Invalid case snapshot: {e}")

    @app.post("/api/cases", response_model=CaseDetailResponse)
    def create_case(payload: dict = Body(...)):
        source = str(payload.get("source", "upload"))
        notes = payload.get("notes")
        try:
            snapshot = CaseSnapshot.from_dict(_normalize_snapshot_payload(payload))
        except Exception as e:
            raise HTTPException(400, f"Invalid case snapshot: {e}")
        item = case_repo.create_case(snapshot, source=source, notes=notes)
        return CaseDetailResponse(case=item, snapshot=snapshot.to_dict())

    @app.get("/api/cases/{case_id}/status")
    def get_case_status(case_id: str):
        keys = [f"{case_id}:baseline", f"{case_id}:stress"]
        return {
            "case_id": case_id,
            "statuses": {k.split(":", 1)[1]: run_status.get(k, {"status": "unknown"}) for k in keys},
        }

    @app.post("/api/cases/{case_id}/baseline")
    def run_case_baseline(case_id: str):
        item = case_repo.get_case(case_id)
        if item is None:
            raise HTTPException(404, f"Case not found: {case_id}")

        if not enable_launch:
            cmd = (
                f"python -m stresslab.cli serve --workspace {workspace} --enable-launch --port 5179"
            )
            return {
                "status": "launch_disabled",
                "message": "Server launch is disabled. Restart with --enable-launch.",
                "repro_command": cmd,
            }

        try:
            snapshot = case_repo.load_snapshot(case_id)
            config = snapshot.baseline_config()
            result = _run_case_mode(
                case_id,
                "baseline",
                config,
                {
                    "run_label": "BASELINE",
                    "scenario_title": "Case Baseline",
                    "case_id": case_id,
                },
            )
            return {"status": "done", "run_id": result["run_id"]}
        except Exception as e:
            run_status[f"{case_id}:baseline"] = {"status": "failed", "error": str(e)}
            raise HTTPException(500, f"Baseline run failed: {e}")

    @app.post("/api/cases/{case_id}/stress")
    def run_case_stress(case_id: str, payload: dict = Body(...)):
        item = case_repo.get_case(case_id)
        if item is None:
            raise HTTPException(404, f"Case not found: {case_id}")

        if not enable_launch:
            cmd = (
                f"python -m stresslab.cli serve --workspace {workspace} --enable-launch --port 5179"
            )
            return {
                "status": "launch_disabled",
                "message": "Server launch is disabled. Restart with --enable-launch.",
                "repro_command": cmd,
            }

        try:
            baseline_run_id = item.get("baseline_run_id") if isinstance(item, dict) else None
            knobs = StressKnobs.from_dict(payload)
            snapshot = case_repo.load_snapshot(case_id)
            config = deepcopy(snapshot.baseline_config())
            config = knobs.apply(config)
            case_repo.update_case(case_id, stress_knobs=knobs.to_dict())
            result = _run_case_mode(
                case_id,
                "stress",
                config,
                {
                    "run_label": "STRESS",
                    "scenario_title": "Case Stress",
                    "case_id": case_id,
                    "stress_knobs": knobs.to_dict(),
                },
            )
            if baseline_run_id and str(result["run_id"]) == str(baseline_run_id):
                raise HTTPException(500, "run_id collision")
            return {"status": "done", "run_id": result["run_id"], "knobs": knobs.to_dict()}
        except Exception as e:
            run_status[f"{case_id}:stress"] = {"status": "failed", "error": str(e)}
            raise HTTPException(500, f"Stress run failed: {e}")

    @app.get("/api/cases/{case_id}/compare", response_model=CompareResponse)
    def get_case_compare(case_id: str):
        item = case_repo.get_case(case_id)
        if item is None:
            raise HTTPException(404, f"Case not found: {case_id}")
        baseline_run_id = item.get("baseline_run_id")
        stress_run_id = item.get("stress_run_id")
        if not baseline_run_id or not stress_run_id:
            raise HTTPException(400, "Both baseline and stress runs are required")
        if str(baseline_run_id) == str(stress_run_id):
            raise HTTPException(500, "run_id collision")

        baseline_summary = _load_run_summary(baseline_run_id)
        stress_summary = _load_run_summary(stress_run_id)
        compare = compare_from_summaries(
            case_id=case_id,
            baseline=baseline_summary,
            stress=stress_summary,
            knobs=item.get("stress_knobs"),
        )
        compare_path = case_repo.case_dir(case_id) / "compare.json"
        with open(compare_path, "w", encoding="utf-8") as f:
            json.dump(compare, f, indent=2)
        return CompareResponse(compare=compare)

    @app.get("/api/cases/{case_id}/onepager")
    def case_onepager(case_id: str):
        compare_payload = get_case_compare(case_id).compare
        deltas = compare_payload.get("deltas", {})
        html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>StressLAB Case {case_id} One-Pager</title>
  <style>
    body {{ font-family: Helvetica, Arial, sans-serif; margin: 24px; color: #111827; }}
    h1 {{ margin: 0 0 8px 0; }}
    .muted {{ color: #6b7280; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 20px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }}
    .k {{ font-size: 12px; color: #6b7280; }}
    .v {{ font-size: 18px; font-weight: 700; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>Case Compare One-Pager</h1>
  <div class=\"muted\">case_id={case_id} | baseline={compare_payload.get("baseline_run_id")} | stress={compare_payload.get("stress_run_id")}</div>
  <div class=\"grid\">
    <div class=\"card\"><div class=\"k\">Trigger Shift</div><div class=\"v\">{deltas.get("trigger_shift")}</div></div>
    <div class=\"card\"><div class=\"k\">Compression Loss %</div><div class=\"v\">{deltas.get("compression_loss_pct")}</div></div>
    <div class=\"card\"><div class=\"k\">Max Pc Ratio</div><div class=\"v\">{deltas.get("max_pc_ratio")}</div></div>
    <div class=\"card\"><div class=\"k\">Max Cov Ratio</div><div class=\"v\">{deltas.get("max_cov_ratio")}</div></div>
    <div class=\"card\"><div class=\"k\">Max Staleness Delta</div><div class=\"v\">{deltas.get("max_staleness_delta")}</div></div>
    <div class=\"card\"><div class=\"k\">Instability Delta</div><div class=\"v\">{deltas.get("instability_delta")}</div></div>
  </div>
</body>
</html>
""".strip()
        return HTMLResponse(content=html)

    @app.get("/api/cases/{case_id}/export")
    def export_case_pack(case_id: str):
        item = case_repo.get_case(case_id)
        if item is None:
            raise HTTPException(404, f"Case not found: {case_id}")
        snapshot = case_repo.load_snapshot(case_id).to_dict()
        compare = get_case_compare(case_id).compare
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract_version": METRICS_CONTRACT_VERSION,
            "code_version": STRESSLAB_VERSION,
            "case": item,
            "snapshot": snapshot,
            "compare": compare,
        }
        return Response(
            content=json.dumps(payload, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=case_export_{case_id}.json"},
        )

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
        resample: str = Query("none", pattern="^(none|uniform)$"),
        max_points: int = Query(300, ge=50, le=100000),
    ):
        run_dir = indexer.resolve_run_path(run_id)
        if run_dir is None:
            raise HTTPException(404, f"Run not found: {run_id}")

        ts_path = run_dir / f"timeseries_{run_id}.parquet"
        if not ts_path.exists():
            raise HTTPException(404, f"Timeseries file not found for run: {run_id}")

        summary_path = run_dir / f"summary_{run_id}.json"
        run_label = None
        if summary_path.exists():
            try:
                with open(summary_path, encoding="utf-8") as f:
                    run_label = (json.load(f) or {}).get("run_label")
            except Exception:
                run_label = None

        columns = [c.strip() for c in cols.split(",") if c.strip()] if cols else None

        df = load_timeseries(ts_path, columns=columns)
        total_rows_before = len(df)
        meta = read_parquet_metadata(ts_path)

        def _minmax(col: str) -> Optional[dict]:
            if col not in df.columns or len(df) == 0:
                return None
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return None
            return {"min": float(np.min(arr)), "max": float(np.max(arr))}

        key_field_ranges = {
            "staleness_seconds": _minmax("staleness_obj1"),
            "cov_trace": _minmax("cov_trace_obj1"),
            "pc": _minmax("pc_degraded"),
        }

        sorted_before = False
        if "timestamp" in df.columns and len(df) >= 2:
            ts_vals = df["timestamp"].to_numpy()
            sorted_before = bool((ts_vals[1:] >= ts_vals[:-1]).all())

        if resample == "uniform":
            df = resample_uniform(df, max_points=max_points)
        else:
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
                "run_id": run_id,
                "loaded_path": str(ts_path),
                "run_label": run_label,
                "n_rows": len(df),
                "key_field_ranges": key_field_ranges,
                "total_rows_before_transform": total_rows_before,
                "total_rows_after_transform": len(df),
                "sorted": sorted_before,
                "resampled": resample == "uniform",
                "grid_points": len(df) if resample == "uniform" else None,
                "schema_version": SCHEMA_VERSION,
                "contract_version": METRICS_CONTRACT_VERSION,
                "code_version": STRESSLAB_VERSION,
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
            import numpy as np
            mc_df = pd.read_csv(csv_path)
            data["csv_columns"] = mc_df.columns.tolist()
            data["csv_total_rows"] = len(mc_df)
            # Send actual row data for histogram rendering (cap at 10k)
            rows = mc_df.head(10000).values.tolist()
            # Convert NaN/inf to None for JSON serialization
            data["csv_rows"] = [
                [None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v for v in row]
                for row in rows
            ]

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
            "cases": len(case_repo.list_cases()),
        }

    # ------------------------------------------------------------------
    # Artifact download
    # ------------------------------------------------------------------

    @app.get("/api/runs/{run_id}/download/{filename}")
    def download_artifact(run_id: str, filename: str):
        """Download a raw artifact file (summary JSON or timeseries parquet)."""
        # Security: only allow specific filenames
        allowed_prefixes = ("summary_", "timeseries_", "scenario_")
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

    # Try to serve built frontend from a few repo-local locations.
    env_web_dist = os.getenv("STRESSLAB_WEB_DIST")
    web_dist_candidates = [
        Path(env_web_dist) if env_web_dist else None,
        Path.cwd() / "web" / "dist",
        Path(__file__).parent.parent.parent.parent / "web" / "dist",
    ]
    web_dist = next((p for p in web_dist_candidates if p is not None and p.is_dir()), None)

    if web_dist is not None:
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
