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
    """Baseline model alert states."""
    SAFE = "Safe"
    ALERT = "Alert"


class ShiroState(str, Enum):
    """SHIRO confidence-integrity model states."""
    MONITOR = "Monitor"
    WATCH = "Watch"
    WARNING = "Warning"
    CRITICAL = "Critical"


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
    pc_baseline: float  # Pc with nominal covariance
    pc_degraded: float  # Pc with degraded (outage-affected) covariance
    risk_ratio: float  # pc_degraded / max(pc_baseline, 1e-30)


@dataclass
class DecisionResult:
    """Output of the decision layer at one timestep."""
    baseline_alert: AlertState
    baseline_trigger_time: Optional[float]
    shiro_state: ShiroState
    shiro_trigger_time: Optional[float]
    shiro_score: float


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


# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------
@dataclass
class ProcessNoiseConfig:
    """Process noise in RTN frame (acceleration noise, km^2/s^4)."""
    sigma_radial: float = 1e-9
    sigma_tangential: float = 1e-9
    sigma_normal: float = 1e-9
    scale: float = 1.0

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


@dataclass
class ManeuverConfig:
    """Optional maneuver uncertainty injection."""
    enabled: bool = False
    delta_v_sigma: float = 0.001  # km/s 1-sigma per axis
    execution_time: float = 0.0  # seconds from epoch


@dataclass
class BaselineThresholdConfig:
    """Baseline decision model parameters."""
    pc_threshold: float = 1e-4
    miss_distance_threshold: Optional[float] = None  # km
    time_to_tca_gate: float = 86400.0  # seconds (trigger only within this)


@dataclass
class ShiroConfig:
    """SHIRO confidence-integrity model parameters."""
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
    baseline: BaselineThresholdConfig = field(default_factory=BaselineThresholdConfig)
    shiro: ShiroConfig = field(default_factory=ShiroConfig)

    # Timeline
    t_start: float = 0.0
    t_end: float = 259200.0  # 3 days in seconds
    dt: float = 60.0  # seconds

    # Risk
    combined_hard_body_radius: float = 0.02  # km  (20 meters)

    # Reproducibility
    seed: int = 42

    def run_id(self) -> str:
        """SHA-256 hash of config + seed."""
        blob = json.dumps(self._serializable(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _serializable(self) -> dict:
        """Convert to JSON-safe dict for hashing."""
        return {
            "state_obj1": self.state_obj1.tolist(),
            "state_obj2": self.state_obj2.tolist(),
            "dynamics_model": self.dynamics_model.value,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "dt": self.dt,
            "seed": self.seed,
            "combined_hard_body_radius": self.combined_hard_body_radius,
        }
