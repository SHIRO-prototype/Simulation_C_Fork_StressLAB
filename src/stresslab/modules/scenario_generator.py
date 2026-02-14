"""Scenario Generator - creates reproducible conjunction scenarios.

Generates initial conditions for two objects on near-collision orbits
with deterministic seeds. Exports a full SimulationConfig.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from stresslab.stresslab_types import (
    DynamicsModel,
    MeasurementConfig,
    ManeuverConfig,
    ProcessNoiseConfig,
    ThresholdV1Config,
    IntegrityV1Config,
    SimulationConfig,
    MU_EARTH_KM3S2,
    RE_EARTH_KM,
)


def _circular_orbit_state(
    semi_major_axis_km: float,
    inclination_rad: float,
    raan_rad: float,
    arg_lat_rad: float,
) -> np.ndarray:
    """Generate ECI state vector for a circular orbit."""
    r = semi_major_axis_km
    v = np.sqrt(MU_EARTH_KM3S2 / r)

    # Position and velocity in perifocal frame
    r_pqw = np.array([r * np.cos(arg_lat_rad), r * np.sin(arg_lat_rad), 0.0])
    v_pqw = np.array([-v * np.sin(arg_lat_rad), v * np.cos(arg_lat_rad), 0.0])

    # Rotation from perifocal to ECI
    cos_O, sin_O = np.cos(raan_rad), np.sin(raan_rad)
    cos_i, sin_i = np.cos(inclination_rad), np.sin(inclination_rad)

    R = np.array([
        [cos_O, -sin_O * cos_i, sin_O * sin_i],
        [sin_O, cos_O * cos_i, -cos_O * sin_i],
        [0.0, sin_i, cos_i],
    ])

    pos = R @ r_pqw
    vel = R @ v_pqw
    return np.concatenate([pos, vel])


def generate_default_scenario(
    seed: int = 42,
    miss_distance_km: float = 0.5,
    t_end: float = 259200.0,
    dt: float = 60.0,
    dynamics: DynamicsModel = DynamicsModel.TWO_BODY_J2,
    t_tca_frac: float = 0.6,
) -> SimulationConfig:
    """Generate a default LEO conjunction scenario.

    Strategy: start both objects at the same position with a tiny
    velocity difference. Object 1 is in a circular LEO orbit.
    Object 2 has a small relative velocity in the cross-track
    direction, creating a slow flyby with minimum separation
    equal to miss_distance_km. The encounter happens near t=0
    and the objects slowly drift apart, providing a realistic
    conjunction timeline where Pc is initially high and decays.

    To stress-test the decision models, we reverse this:
    place the objects so they *converge* toward closest approach
    at t_tca = t_end * t_tca_frac (default 0.6).
    """
    rng = np.random.default_rng(seed)

    sma = RE_EARTH_KM + 500.0
    v_circ = np.sqrt(MU_EARTH_KM3S2 / sma)

    # Object 1: circular orbit in x-y plane (inc=0 for simplicity,
    # then we rotate the whole scenario)
    # Start at (sma, 0, 0) with velocity (0, v_circ, 0)
    pos1_base = np.array([sma, 0.0, 0.0])
    vel1_base = np.array([0.0, v_circ, 0.0])

    # Object 2: starts offset and converges
    # At TCA (t_tca = t_end * t_tca_frac), objects should be at miss_distance_km apart
    t_tca = t_end * max(0.01, min(0.99, t_tca_frac))

    # Relative velocity at encounter: scale with t_tca to keep
    # initial offset reasonable (~10-50 km)
    # Target initial offset: ~50 km -> v_rel = 50 / t_tca
    target_initial_offset_km = 50.0
    v_rel = target_initial_offset_km / t_tca  # km/s cross-track

    # At t=0, object 2 is offset by:
    #   along-track: 0 (same phase)
    #   cross-track: miss_distance_km + v_rel * t_tca (will converge)
    #   radial: 0
    initial_cross_offset = miss_distance_km + v_rel * t_tca

    pos2_base = pos1_base.copy()
    pos2_base[2] += initial_cross_offset  # Z offset (cross-track for equatorial orbit)

    vel2_base = vel1_base.copy()
    vel2_base[2] -= v_rel  # approaching in Z

    # Rotate entire scenario to realistic inclination (51.6 deg)
    inc = np.deg2rad(51.6)
    raan = np.deg2rad(30.0)

    cos_i, sin_i = np.cos(inc), np.sin(inc)
    cos_O, sin_O = np.cos(raan), np.sin(raan)

    # Rotation: first by RAAN around Z, then by inc around node line
    R_O = np.array([
        [cos_O, -sin_O, 0],
        [sin_O,  cos_O, 0],
        [0,      0,     1],
    ])
    R_i = np.array([
        [1,     0,      0],
        [0, cos_i, -sin_i],
        [0, sin_i,  cos_i],
    ])
    R = R_O @ R_i

    state1 = np.concatenate([R @ pos1_base, R @ vel1_base])
    state2 = np.concatenate([R @ pos2_base, R @ vel2_base])

    # Initial covariances (position: 0.1 km, velocity: 0.0001 km/s)
    pos_sigma = 0.1  # km
    vel_sigma = 1e-4  # km/s
    cov1 = np.diag([pos_sigma**2]*3 + [vel_sigma**2]*3)
    cov2 = np.diag([pos_sigma**2]*3 + [vel_sigma**2]*3)

    # Measurement config with one outage window
    meas = MeasurementConfig(
        update_interval=3600.0,
        noise_sigma_pos=0.01,
        outage_windows=[
            {"start": t_end * 0.3, "end": t_end * 0.5},
        ],
    )

    config = SimulationConfig(
        state_obj1=state1,
        state_obj2=state2,
        cov_obj1=cov1,
        cov_obj2=cov2,
        dynamics_model=dynamics,
        process_noise=ProcessNoiseConfig(),
        measurement=meas,
        maneuver=ManeuverConfig(enabled=False),
        threshold_v1=ThresholdV1Config(),
        integrity_v1=IntegrityV1Config(),
        t_start=0.0,
        t_end=t_end,
        dt=dt,
        seed=seed,
    )

    return config


def export_scenario(config: SimulationConfig, output_dir: Path) -> Path:
    """Export scenario config to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = config.run_id()
    path = output_dir / f"scenario_{run_id}.json"

    data = {
        "run_id": run_id,
        "seed": config.seed,
        "dynamics_model": config.dynamics_model.value,
        "t_start": config.t_start,
        "t_end": config.t_end,
        "dt": config.dt,
        "combined_hard_body_radius": config.combined_hard_body_radius,
        "state_obj1": config.state_obj1.tolist(),
        "state_obj2": config.state_obj2.tolist(),
        "cov_obj1": config.cov_obj1.tolist(),
        "cov_obj2": config.cov_obj2.tolist(),
        "measurement": {
            "update_interval": config.measurement.update_interval,
            "noise_sigma_pos": config.measurement.noise_sigma_pos,
            "outage_windows": config.measurement.outage_windows,
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path
