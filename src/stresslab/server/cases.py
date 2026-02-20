"""Case snapshot registry and case-driven stress workflow helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from stresslab import __version__ as STRESSLAB_VERSION
from stresslab.metrics_contract import METRICS_CONTRACT_VERSION
from stresslab.modules.logging_engine import SCHEMA_VERSION
from stresslab.stresslab_types import (
    DynamicsModel,
    IntegrityV1Config,
    ManeuverConfig,
    MeasurementConfig,
    ProcessNoiseConfig,
    SimulationConfig,
    ThresholdV1Config,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _json_load(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _validate_vec6(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (6,):
        raise ValueError(f"{name} must be length-6; got shape {arr.shape}")
    return arr


def _validate_cov6(name: str, value: Any, psd_eps: float = 1e-10) -> np.ndarray:
    cov = np.asarray(value, dtype=float)
    if cov.shape != (6, 6):
        raise ValueError(f"{name} must be 6x6; got shape {cov.shape}")
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eig = np.linalg.eigvalsh(cov)
    if float(np.min(eig)) < -psd_eps:
        raise ValueError(
            f"{name} must be positive semi-definite within eps={psd_eps}; min eigenvalue={float(np.min(eig)):.3e}"
        )
    return cov


@dataclass(frozen=True)
class CaseSnapshot:
    """Serializable operator case snapshot."""

    state_obj1: np.ndarray
    state_obj2: np.ndarray
    cov_obj1: np.ndarray
    cov_obj2: np.ndarray
    seed: int
    dynamics_model: str
    t_start: float
    t_end: float
    dt: float
    combined_hard_body_radius: float
    measurement_update_interval: float
    measurement_noise_sigma_pos: float
    process_noise_scale: float
    process_noise_sigma_radial: float
    process_noise_sigma_tangential: float
    process_noise_sigma_normal: float
    threshold_pc: float
    threshold_time_to_tca_gate: float
    threshold_miss_distance: Optional[float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseSnapshot":
        state_obj1 = _validate_vec6("state_obj1", data.get("state_obj1"))
        state_obj2 = _validate_vec6("state_obj2", data.get("state_obj2"))
        cov_obj1 = _validate_cov6("cov_obj1", data.get("cov_obj1"))
        cov_obj2 = _validate_cov6("cov_obj2", data.get("cov_obj2"))

        t_start = float(data.get("t_start", 0.0))
        t_end = float(data.get("t_end", 259200.0))
        dt = float(data.get("dt", 60.0))
        if t_end <= t_start:
            raise ValueError(f"t_end must be > t_start; got {t_end} <= {t_start}")
        if dt <= 0:
            raise ValueError("dt must be > 0")

        measurement_update_interval = float(data.get("measurement_update_interval", 3600.0))
        measurement_noise_sigma_pos = float(data.get("measurement_noise_sigma_pos", 0.01))
        if measurement_update_interval <= 0:
            raise ValueError("measurement_update_interval must be > 0")
        if measurement_noise_sigma_pos <= 0:
            raise ValueError("measurement_noise_sigma_pos must be > 0")

        process_noise_scale = float(data.get("process_noise_scale", 1.0))
        if process_noise_scale <= 0:
            raise ValueError("process_noise_scale must be > 0")

        threshold_pc = float(data.get("threshold_pc", 1e-4))
        threshold_time_to_tca_gate = float(data.get("threshold_time_to_tca_gate", 86400.0))
        threshold_miss_distance_raw = data.get("threshold_miss_distance")
        threshold_miss_distance = (
            float(threshold_miss_distance_raw)
            if threshold_miss_distance_raw is not None
            else None
        )

        return cls(
            state_obj1=state_obj1,
            state_obj2=state_obj2,
            cov_obj1=cov_obj1,
            cov_obj2=cov_obj2,
            seed=int(data.get("seed", 42)),
            dynamics_model=str(data.get("dynamics_model", DynamicsModel.TWO_BODY_J2.value)),
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            combined_hard_body_radius=float(data.get("combined_hard_body_radius", 0.02)),
            measurement_update_interval=measurement_update_interval,
            measurement_noise_sigma_pos=measurement_noise_sigma_pos,
            process_noise_scale=process_noise_scale,
            process_noise_sigma_radial=float(data.get("process_noise_sigma_radial", 1e-9)),
            process_noise_sigma_tangential=float(data.get("process_noise_sigma_tangential", 1e-9)),
            process_noise_sigma_normal=float(data.get("process_noise_sigma_normal", 1e-9)),
            threshold_pc=threshold_pc,
            threshold_time_to_tca_gate=threshold_time_to_tca_gate,
            threshold_miss_distance=threshold_miss_distance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_obj1": self.state_obj1.tolist(),
            "state_obj2": self.state_obj2.tolist(),
            "cov_obj1": self.cov_obj1.tolist(),
            "cov_obj2": self.cov_obj2.tolist(),
            "seed": self.seed,
            "dynamics_model": self.dynamics_model,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "dt": self.dt,
            "combined_hard_body_radius": self.combined_hard_body_radius,
            "measurement_update_interval": self.measurement_update_interval,
            "measurement_noise_sigma_pos": self.measurement_noise_sigma_pos,
            "process_noise_scale": self.process_noise_scale,
            "process_noise_sigma_radial": self.process_noise_sigma_radial,
            "process_noise_sigma_tangential": self.process_noise_sigma_tangential,
            "process_noise_sigma_normal": self.process_noise_sigma_normal,
            "threshold_pc": self.threshold_pc,
            "threshold_time_to_tca_gate": self.threshold_time_to_tca_gate,
            "threshold_miss_distance": self.threshold_miss_distance,
        }

    def baseline_config(self) -> SimulationConfig:
        return SimulationConfig(
            state_obj1=self.state_obj1.copy(),
            state_obj2=self.state_obj2.copy(),
            cov_obj1=self.cov_obj1.copy(),
            cov_obj2=self.cov_obj2.copy(),
            dynamics_model=DynamicsModel(self.dynamics_model),
            process_noise=ProcessNoiseConfig(
                sigma_radial=self.process_noise_sigma_radial,
                sigma_tangential=self.process_noise_sigma_tangential,
                sigma_normal=self.process_noise_sigma_normal,
                scale=self.process_noise_scale,
            ),
            measurement=MeasurementConfig(
                update_interval=self.measurement_update_interval,
                noise_sigma_pos=self.measurement_noise_sigma_pos,
                outage_windows=[],
            ),
            maneuver=ManeuverConfig(enabled=False),
            threshold_v1=ThresholdV1Config(
                pc_threshold=self.threshold_pc,
                miss_distance_threshold=self.threshold_miss_distance,
                time_to_tca_gate=self.threshold_time_to_tca_gate,
            ),
            integrity_v1=IntegrityV1Config(),
            t_start=self.t_start,
            t_end=self.t_end,
            dt=self.dt,
            combined_hard_body_radius=self.combined_hard_body_radius,
            seed=self.seed,
        )


@dataclass(frozen=True)
class StressKnobs:
    measurement_cadence_s: Optional[float]
    outage_windows: list[dict[str, float]]
    process_noise_scale: Optional[float]
    maneuver_enabled: bool
    maneuver_delta_v_sigma: float
    maneuver_execution_time: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StressKnobs":
        cadence_raw = data.get("measurement_cadence_s")
        cadence = float(cadence_raw) if cadence_raw is not None else None
        if cadence is not None and cadence <= 0:
            raise ValueError("measurement_cadence_s must be > 0")

        outage_windows = data.get("outage_windows") or []
        if not isinstance(outage_windows, list):
            raise ValueError("outage_windows must be a list")
        cleaned_windows: list[dict[str, float]] = []
        for idx, window in enumerate(outage_windows):
            if not isinstance(window, dict) or "start" not in window or "end" not in window:
                raise ValueError(f"outage_windows[{idx}] must have start and end")
            start = float(window["start"])
            end = float(window["end"])
            if end <= start:
                raise ValueError(f"outage_windows[{idx}] end must be > start")
            cleaned_windows.append({"start": start, "end": end})

        pn_scale_raw = data.get("process_noise_scale")
        pn_scale = float(pn_scale_raw) if pn_scale_raw is not None else None
        if pn_scale is not None and pn_scale <= 0:
            raise ValueError("process_noise_scale must be > 0")

        maneuver_enabled = bool(data.get("maneuver_enabled", False))
        maneuver_delta_v_sigma = float(data.get("maneuver_delta_v_sigma", 0.001))
        maneuver_execution_time = float(data.get("maneuver_execution_time", 0.0))

        return cls(
            measurement_cadence_s=cadence,
            outage_windows=cleaned_windows,
            process_noise_scale=pn_scale,
            maneuver_enabled=maneuver_enabled,
            maneuver_delta_v_sigma=maneuver_delta_v_sigma,
            maneuver_execution_time=maneuver_execution_time,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement_cadence_s": self.measurement_cadence_s,
            "outage_windows": self.outage_windows,
            "process_noise_scale": self.process_noise_scale,
            "maneuver_enabled": self.maneuver_enabled,
            "maneuver_delta_v_sigma": self.maneuver_delta_v_sigma,
            "maneuver_execution_time": self.maneuver_execution_time,
        }

    def apply(self, config: SimulationConfig) -> SimulationConfig:
        cfg = config
        if self.measurement_cadence_s is not None:
            cfg.measurement.update_interval = self.measurement_cadence_s
        cfg.measurement.outage_windows = self.outage_windows
        if self.process_noise_scale is not None:
            cfg.process_noise.scale = self.process_noise_scale
        cfg.maneuver.enabled = self.maneuver_enabled
        cfg.maneuver.delta_v_sigma = self.maneuver_delta_v_sigma
        cfg.maneuver.execution_time = self.maneuver_execution_time
        return cfg


class CaseRepository:
    """Persistent case storage rooted at outputs/cases/."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.cases_root = workspace / "cases"
        self.index_path = self.cases_root / "index.json"

    def _ensure_index(self) -> dict[str, Any]:
        if self.index_path.exists():
            try:
                return _json_load(self.index_path)
            except Exception:
                pass
        data = {"schema_version": 1, "cases": []}
        _json_dump(self.index_path, data)
        return data

    def _write_index(self, data: dict[str, Any]) -> None:
        _json_dump(self.index_path, data)

    def list_cases(self) -> list[dict[str, Any]]:
        data = self._ensure_index()
        return list(data.get("cases", []))

    def get_case(self, case_id: str) -> Optional[dict[str, Any]]:
        for item in self.list_cases():
            if item.get("case_id") == case_id:
                return item
        return None

    def create_case(
        self,
        snapshot: CaseSnapshot,
        source: str,
        notes: Optional[str] = None,
        case_id_override: Optional[str] = None,
    ) -> dict[str, Any]:
        index = self._ensure_index()
        if case_id_override:
            case_id = case_id_override
        else:
            snapshot_blob = json.dumps(snapshot.to_dict(), sort_keys=True)
            digest = hashlib.sha256(snapshot_blob.encode()).hexdigest()[:10]
            case_id = f"case_{digest}_{int(datetime.now(timezone.utc).timestamp())}"
        case_dir = self.cases_root / case_id
        snapshot_path = case_dir / "snapshot.json"
        _json_dump(snapshot_path, snapshot.to_dict())

        existing = self.get_case(case_id)
        entry = {
            "case_id": case_id,
            "source": source,
            "created_at": existing.get("created_at") if isinstance(existing, dict) else _now_iso(),
            "notes": notes,
            "snapshot_path": str(snapshot_path.relative_to(self.workspace)).replace("\\", "/"),
            "baseline_run_id": existing.get("baseline_run_id") if isinstance(existing, dict) else None,
            "stress_run_id": existing.get("stress_run_id") if isinstance(existing, dict) else None,
        }
        cases = index.setdefault("cases", [])
        replaced = False
        for i, item in enumerate(cases):
            if item.get("case_id") == case_id:
                cases[i] = entry
                replaced = True
                break
        if not replaced:
            cases.append(entry)
        self._write_index(index)
        return entry

    def load_snapshot(self, case_id: str) -> CaseSnapshot:
        entry = self.get_case(case_id)
        if entry is None:
            raise ValueError(f"Unknown case_id: {case_id}")
        snapshot_path = self.workspace / entry["snapshot_path"]
        if not snapshot_path.exists():
            raise ValueError(f"Snapshot file missing for case {case_id}: {snapshot_path}")
        return CaseSnapshot.from_dict(_json_load(snapshot_path))

    def update_case(self, case_id: str, **fields: Any) -> dict[str, Any]:
        index = self._ensure_index()
        updated: Optional[dict[str, Any]] = None
        for item in index.get("cases", []):
            if item.get("case_id") == case_id:
                item.update(fields)
                updated = item
                break
        if updated is None:
            raise ValueError(f"Unknown case_id: {case_id}")
        self._write_index(index)
        return updated

    def case_dir(self, case_id: str) -> Path:
        return self.cases_root / case_id


def compare_from_summaries(case_id: str, baseline: dict[str, Any], stress: dict[str, Any], knobs: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Build deterministic compare payload for a case."""

    def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(b) - float(a)

    def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        if abs(float(a)) < 1e-30:
            return None
        return float(b) / float(a)

    baseline_trigger = baseline.get("threshold_v1_trigger_time")
    stress_trigger = stress.get("threshold_v1_trigger_time")
    baseline_dcw = baseline.get("decision_compression_window")
    stress_dcw = stress.get("decision_compression_window")
    compression_loss_pct = None
    if baseline_dcw is not None and stress_dcw is not None and abs(float(baseline_dcw)) > 1e-30:
        compression_loss_pct = ((float(stress_dcw) - float(baseline_dcw)) / abs(float(baseline_dcw))) * 100.0

    payload = {
        "case_id": case_id,
        "schema_version": SCHEMA_VERSION,
        "contract_version": METRICS_CONTRACT_VERSION,
        "code_version": STRESSLAB_VERSION,
        "generated_at": _now_iso(),
        "baseline_run_id": baseline.get("run_id"),
        "stress_run_id": stress.get("run_id"),
        "stress_knobs": knobs,
        "deltas": {
            "trigger_shift": _delta(baseline_trigger, stress_trigger),
            "compression_loss_pct": compression_loss_pct,
            "max_pc_ratio": _ratio(baseline.get("max_pc_degraded"), stress.get("max_pc_degraded")),
            "max_cov_ratio": _ratio(baseline.get("max_cov_trace"), stress.get("max_cov_trace")),
            "max_staleness_delta": _delta(baseline.get("max_staleness"), stress.get("max_staleness")),
            "instability_delta": _delta(
                baseline.get("decision_instability_index"),
                stress.get("decision_instability_index"),
            ),
        },
        "baseline_summary": baseline,
        "stress_summary": stress,
    }
    return payload
