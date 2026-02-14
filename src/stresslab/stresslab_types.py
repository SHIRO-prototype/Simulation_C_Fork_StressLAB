"""Core data types and contracts for StressLAB.

All modules communicate through these dataclasses. Coordinate frame
is Earth-Centered Inertial (ECI). State vector is [x, y, z, vx, vy, vz]
with position in km and velocity in km/s.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU_EARTH_KM3S2 = 398600.4418  # km^3/s^2
RE_EARTH_KM = 6378.137  # km  (WGS-84 equatorial)
J2 = 1.08263e-3


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class DynamicsModel(str, Enum):
    TWO_BODY = "two_body"
    TWO_BODY_J2 = "two_body_plus_J2"


class AlertState(str, Enum):
    """Threshold-v1 decision model alert states."""
    SAFE = "Safe"
    ALERT = "Alert"


class IntegrityV1State(str, Enum):
    """Integrity-v1 decision model states."""
    MONITOR = "Monitor"
    WATCH = "Watch"
    WARNING = "Warning"
    CRITICAL = "Critical"


class ShiroState(str, Enum):
    """SHIRO posture state machine (SAFE -> ELEVATED -> CRITICAL)."""
    SAFE = "SAFE"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Core data containers
# ---------------------------------------------------------------------------
@dataclass
class StateVector:
    """6-DOF Cartesian state in ECI [km, km/s]."""
    position: np.ndarray  # (3,) km
    velocity: np.ndarray  # (3,) km/s

    def as_array(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "StateVector":
        return cls(position=arr[:3].copy(), velocity=arr[3:6].copy())


@dataclass
class PropagationResult:
    """Output of the propagation engine at one timestep."""
    state_obj1: StateVector
    state_obj2: StateVector
    rel_position: np.ndarray  # (3,) km
    rel_velocity: np.ndarray  # (3,) km/s
    stm_obj1: np.ndarray  # (6,6)
    stm_obj2: np.ndarray  # (6,6)
    estimated_tca: float  # seconds from current epoch


@dataclass
class CovarianceResult:
    """Output of the covariance engine at one timestep per object."""
    covariance: np.ndarray  # (6,6) ECI
    trace: float
    frobenius: float
    max_eigenvalue: float
    growth_rate: float  # d(trace)/dt  [km^2/s]


@dataclass
class MeasurementResult:
    """Output of measurement model at one timestep."""
    applied: bool
    time_since_last_update: float  # seconds


@dataclass
class GeometryResult:
    """Encounter geometry at one timestep."""
    miss_distance: float  # km
    rel_velocity_at_tca: float  # km/s  (magnitude)
    time_to_tca: float  # seconds
    b_plane_xi: np.ndarray  # (3,) unit vector along relative velocity
    b_plane_eta: np.ndarray  # (3,) B-plane basis 1
    b_plane_zeta: np.ndarray  # (3,) B-plane basis 2


@dataclass
class RiskResult:
    """Output of collision probability computation."""
    pc_reference: float  # Pc with nominal (reference) covariance
    pc_degraded: float  # Pc with degraded (outage-affected) covariance
    risk_ratio: float  # pc_degraded / max(pc_reference, 1e-30)


@dataclass
class DecisionResult:
    """Output of the decision layer at one timestep."""
    threshold_v1_alert: AlertState
    threshold_v1_trigger_time: Optional[float]
    integrity_v1_state: IntegrityV1State
    integrity_v1_trigger_time: Optional[float]
    integrity_v1_score: float
    shiro_state: Optional["ShiroState"] = None
    shiro_trigger_path: Optional[str] = None


@dataclass
class TimeStep:
    """Aggregated data for one simulation timestep."""
    t: float  # seconds from epoch
    propagation: PropagationResult
    covariance_obj1: CovarianceResult
    covariance_obj2: CovarianceResult
    measurement_obj1: MeasurementResult
    measurement_obj2: MeasurementResult
    geometry: GeometryResult
    risk: RiskResult
    decision: DecisionResult


@dataclass
class KnowledgeStream:
    """Encapsulates covariance tracking state for one knowledge stream.

    A knowledge stream pairs per-object covariance matrices with
    derived metrics (norm, growth rate) and staleness tracking.
    Two streams run in parallel during simulation:
      - reference_stream: always receives measurement updates (no outages)
      - degraded_stream: subject to outage windows (actual tracking)
    """
    cov_obj1: np.ndarray        # (6,6) ECI covariance for object 1
    cov_obj2: np.ndarray        # (6,6) ECI covariance for object 2
    last_update_time_obj1: float  # seconds from epoch
    last_update_time_obj2: float  # seconds from epoch
    prev_trace_obj1: float      # previous trace for growth rate computation
    prev_trace_obj2: float      # previous trace for growth rate computation

    @classmethod
    def from_initial(
        cls,
        cov_obj1: np.ndarray,
        cov_obj2: np.ndarray,
        t_start: float,
    ) -> "KnowledgeStream":
        """Create a KnowledgeStream from initial covariances."""
        return cls(
            cov_obj1=cov_obj1.copy(),
            cov_obj2=cov_obj2.copy(),
            last_update_time_obj1=t_start,
            last_update_time_obj2=t_start,
            prev_trace_obj1=float(np.trace(cov_obj1)),
            prev_trace_obj2=float(np.trace(cov_obj2)),
        )

    @property
    def cov_norm(self) -> float:
        """Combined covariance norm (sum of traces)."""
        return float(np.trace(self.cov_obj1) + np.trace(self.cov_obj2))

    def staleness(self, t_now: float) -> float:
        """Max staleness across both objects at time t_now."""
        return max(
            t_now - self.last_update_time_obj1,
            t_now - self.last_update_time_obj2,
        )


# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------
class ConfigValidationError(ValueError):
    """Raised when a configuration dataclass has invalid field values."""
    pass


@dataclass
class ProcessNoiseConfig:
    """Process noise in RTN frame (acceleration noise, km^2/s^4)."""
    sigma_radial: float = 1e-9
    sigma_tangential: float = 1e-9
    sigma_normal: float = 1e-9
    scale: float = 1.0

    def __post_init__(self):
        if self.sigma_radial < 0:
            raise ConfigValidationError("sigma_radial must be >= 0")
        if self.sigma_tangential < 0:
            raise ConfigValidationError("sigma_tangential must be >= 0")
        if self.sigma_normal < 0:
            raise ConfigValidationError("sigma_normal must be >= 0")
        if self.scale <= 0:
            raise ConfigValidationError("process noise scale must be > 0")

    def Q_rtn(self) -> np.ndarray:
        """3x3 diagonal acceleration noise in RTN."""
        s = self.scale
        return np.diag([
            s * self.sigma_radial ** 2,
            s * self.sigma_tangential ** 2,
            s * self.sigma_normal ** 2,
        ])


@dataclass
class MeasurementConfig:
    """Measurement schedule configuration."""
    update_interval: float = 3600.0  # seconds
    noise_sigma_pos: float = 0.01  # km (1-sigma per axis)
    outage_windows: list = field(default_factory=list)
    # Each outage: {"start": float, "end": float} in seconds from epoch

    def __post_init__(self):
        if self.update_interval <= 0:
            raise ConfigValidationError("update_interval must be > 0")
        if self.noise_sigma_pos <= 0:
            raise ConfigValidationError("noise_sigma_pos must be > 0")
        for i, w in enumerate(self.outage_windows):
            if not isinstance(w, dict) or "start" not in w or "end" not in w:
                raise ConfigValidationError(
                    f"outage_windows[{i}] must be a dict with 'start' and 'end' keys"
                )
            if w["end"] <= w["start"]:
                raise ConfigValidationError(
                    f"outage_windows[{i}]: end ({w['end']}) must be > start ({w['start']})"
                )


@dataclass
class ManeuverConfig:
    """Optional maneuver uncertainty injection."""
    enabled: bool = False
    delta_v_sigma: float = 0.001  # km/s 1-sigma per axis
    execution_time: float = 0.0  # seconds from epoch

    def __post_init__(self):
        if self.delta_v_sigma < 0:
            raise ConfigValidationError("delta_v_sigma must be >= 0")
        if self.execution_time < 0:
            raise ConfigValidationError("execution_time must be >= 0")


@dataclass
class ThresholdV1Config:
    """Threshold-v1 decision model parameters."""
    pc_threshold: float = 1e-4
    miss_distance_threshold: Optional[float] = None  # km
    time_to_tca_gate: float = 86400.0  # seconds (trigger only within this)

    def __post_init__(self):
        if self.pc_threshold <= 0:
            raise ConfigValidationError("pc_threshold must be > 0")
        if self.miss_distance_threshold is not None and self.miss_distance_threshold <= 0:
            raise ConfigValidationError("miss_distance_threshold must be > 0 when set")
        if self.time_to_tca_gate <= 0:
            raise ConfigValidationError("time_to_tca_gate must be > 0")


@dataclass
class ShiroConfig:
    """SHIRO posture state machine: geometry gating + freshness/sigma elevated triggers."""
    d_watch_km: float = 8.0  # geometry becomes relevant inside this (m in spec -> km)
    d_act_km: float = 1.0  # CRITICAL threshold (miss distance)
    pc_critical: float = 1e-4
    dt_max_s: float = 3600.0  # staleness >= this -> ELEVATED when geometry relevant

    def __post_init__(self):
        if self.d_watch_km <= 0 or self.d_act_km <= 0:
            raise ConfigValidationError("d_watch_km and d_act_km must be > 0")
        if self.d_act_km > self.d_watch_km:
            raise ConfigValidationError("d_act_km must be <= d_watch_km")
        if self.pc_critical <= 0:
            raise ConfigValidationError("pc_critical must be > 0")
        if self.dt_max_s <= 0:
            raise ConfigValidationError("dt_max_s must be > 0")


@dataclass
class IntegrityV1Config:
    """Integrity-v1 decision model parameters."""
    weights: np.ndarray = field(
        default_factory=lambda: np.array([0.4, 0.2, 0.2, 0.2])
    )  # [w_pc, w_cov_norm, w_growth_rate, w_staleness]
    threshold_monitor_to_watch: float = 0.25
    threshold_watch_to_warning: float = 0.50
    threshold_warning_to_critical: float = 0.75
    # Normalization references
    pc_ref: float = 1e-4
    cov_norm_ref: float = 100.0  # km^2
    growth_rate_ref: float = 1.0  # km^2/s
    staleness_ref: float = 86400.0  # seconds

    def __post_init__(self):
        w = np.asarray(self.weights)
        if w.shape != (4,):
            raise ConfigValidationError("weights must have exactly 4 elements")
        if np.any(w < 0):
            raise ConfigValidationError("all weights must be >= 0")
        if not (self.threshold_monitor_to_watch
                < self.threshold_watch_to_warning
                < self.threshold_warning_to_critical):
            raise ConfigValidationError(
                "integrity-v1 thresholds must be strictly increasing: "
                "monitor_to_watch < watch_to_warning < warning_to_critical"
            )
        if self.pc_ref <= 0:
            raise ConfigValidationError("pc_ref must be > 0")
        if self.cov_norm_ref <= 0:
            raise ConfigValidationError("cov_norm_ref must be > 0")
        if self.growth_rate_ref <= 0:
            raise ConfigValidationError("growth_rate_ref must be > 0")
        if self.staleness_ref <= 0:
            raise ConfigValidationError("staleness_ref must be > 0")


@dataclass
class SimulationConfig:
    """Master configuration for one simulation run."""
    # Initial states
    state_obj1: np.ndarray = field(default_factory=lambda: np.zeros(6))
    state_obj2: np.ndarray = field(default_factory=lambda: np.zeros(6))
    cov_obj1: np.ndarray = field(default_factory=lambda: np.eye(6) * 1e-4)
    cov_obj2: np.ndarray = field(default_factory=lambda: np.eye(6) * 1e-4)

    # Dynamics
    dynamics_model: DynamicsModel = DynamicsModel.TWO_BODY_J2
    process_noise: ProcessNoiseConfig = field(default_factory=ProcessNoiseConfig)

    # Measurement
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)

    # Maneuver
    maneuver: ManeuverConfig = field(default_factory=ManeuverConfig)

    # Decision
    threshold_v1: ThresholdV1Config = field(default_factory=ThresholdV1Config)
    integrity_v1: IntegrityV1Config = field(default_factory=IntegrityV1Config)
    shiro: Optional[ShiroConfig] = None

    # Timeline
    t_start: float = 0.0
    t_end: float = 259200.0  # 3 days in seconds
    dt: float = 60.0  # seconds

    # Risk
    combined_hard_body_radius: float = 0.02  # km  (20 meters)

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        if self.dt <= 0:
            raise ConfigValidationError("dt must be > 0")
        if self.t_end <= self.t_start:
            raise ConfigValidationError(
                f"t_end ({self.t_end}) must be > t_start ({self.t_start})"
            )
        if self.combined_hard_body_radius <= 0:
            raise ConfigValidationError("combined_hard_body_radius must be > 0")
        if self.state_obj1.shape != (6,):
            raise ConfigValidationError("state_obj1 must be a 6-element array")
        if self.state_obj2.shape != (6,):
            raise ConfigValidationError("state_obj2 must be a 6-element array")
        if self.cov_obj1.shape != (6, 6):
            raise ConfigValidationError("cov_obj1 must be a 6x6 matrix")
        if self.cov_obj2.shape != (6, 6):
            raise ConfigValidationError("cov_obj2 must be a 6x6 matrix")

    def run_id(self) -> str:
        """SHA-256 hash of the fully-resolved config + code version.

        Every field that can affect simulation output is included.
        Changing *any* parameter produces a different run_id, ensuring
        that cached results are never confused across configurations.
        """
        from stresslab import __version__
        blob = json.dumps(self._serializable(__version__), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _serializable(self, code_version: str = "") -> dict:
        """Convert the full config to a JSON-safe dict for hashing.

        Args:
            code_version: stresslab package version tag (included in hash).
        """
        return {
            "code_version": code_version,
            # Initial states & covariances
            "state_obj1": self.state_obj1.tolist(),
            "state_obj2": self.state_obj2.tolist(),
            "cov_obj1": self.cov_obj1.tolist(),
            "cov_obj2": self.cov_obj2.tolist(),
            # Dynamics
            "dynamics_model": self.dynamics_model.value,
            "process_noise": {
                "sigma_radial": self.process_noise.sigma_radial,
                "sigma_tangential": self.process_noise.sigma_tangential,
                "sigma_normal": self.process_noise.sigma_normal,
                "scale": self.process_noise.scale,
            },
            # Measurement
            "measurement": {
                "update_interval": self.measurement.update_interval,
                "noise_sigma_pos": self.measurement.noise_sigma_pos,
                "outage_windows": self.measurement.outage_windows,
            },
            # Maneuver
            "maneuver": {
                "enabled": self.maneuver.enabled,
                "delta_v_sigma": self.maneuver.delta_v_sigma,
                "execution_time": self.maneuver.execution_time,
            },
            # Decision models
            "threshold_v1": {
                "pc_threshold": self.threshold_v1.pc_threshold,
                "miss_distance_threshold": self.threshold_v1.miss_distance_threshold,
                "time_to_tca_gate": self.threshold_v1.time_to_tca_gate,
            },
            "integrity_v1": {
                "weights": self.integrity_v1.weights.tolist(),
                "threshold_monitor_to_watch": self.integrity_v1.threshold_monitor_to_watch,
                "threshold_watch_to_warning": self.integrity_v1.threshold_watch_to_warning,
                "threshold_warning_to_critical": self.integrity_v1.threshold_warning_to_critical,
                "pc_ref": self.integrity_v1.pc_ref,
                "cov_norm_ref": self.integrity_v1.cov_norm_ref,
                "growth_rate_ref": self.integrity_v1.growth_rate_ref,
                "staleness_ref": self.integrity_v1.staleness_ref,
            },
            "shiro": (
                {
                    "d_watch_km": self.shiro.d_watch_km,
                    "d_act_km": self.shiro.d_act_km,
                    "pc_critical": self.shiro.pc_critical,
                    "dt_max_s": self.shiro.dt_max_s,
                }
                if self.shiro is not None
                else None
            ),
            # Timeline
            "t_start": self.t_start,
            "t_end": self.t_end,
            "dt": self.dt,
            # Risk
            "combined_hard_body_radius": self.combined_hard_body_radius,
            # Reproducibility
            "seed": self.seed,
        }
