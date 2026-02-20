"""Wiring tests for baseline vs stress divergence."""

from __future__ import annotations

import numpy as np

from stresslab.modules.risk_model import compute_risk
from stresslab.modules.scenario_generator import generate_default_scenario
from stresslab.modules.simulation_runner import run_simulation


def _short_config(seed: int = 42):
    cfg = generate_default_scenario(seed=seed, miss_distance_km=0.2)
    cfg.t_end = cfg.t_start + 14400.0
    cfg.dt = 60.0
    cfg.measurement.update_interval = 60.0
    return cfg


def test_outage_increases_staleness(tmp_path):
    cfg = _short_config(seed=7)
    cfg.measurement.outage_windows = [
        {"start": cfg.t_start + 3600.0, "end": cfg.t_start + 10800.0}
    ]

    result = run_simulation(cfg, output_dir=tmp_path / "outage")
    df = result["logger"].to_dataframe()

    assert float(df["staleness_obj1"].max()) >= 7200.0
    assert float(df["staleness_obj1_reference"].max()) <= cfg.measurement.update_interval


def test_process_noise_increases_covariance(tmp_path):
    baseline = _short_config(seed=11)
    baseline.measurement.outage_windows = []
    baseline.measurement.update_interval = 1e9
    baseline.process_noise.sigma_radial = 1e-5
    baseline.process_noise.sigma_tangential = 1e-5
    baseline.process_noise.sigma_normal = 1e-5
    baseline.process_noise.scale = 1.0

    stress = _short_config(seed=11)
    stress.measurement.outage_windows = []
    stress.measurement.update_interval = 1e9
    stress.process_noise.sigma_radial = 1e-5
    stress.process_noise.sigma_tangential = 1e-5
    stress.process_noise.sigma_normal = 1e-5
    stress.process_noise.scale = 10.0

    b = run_simulation(baseline, output_dir=tmp_path / "baseline")
    s = run_simulation(stress, output_dir=tmp_path / "stress")

    b_cov = float(b["logger"].to_dataframe()["cov_trace_obj1"].max())
    s_cov = float(s["logger"].to_dataframe()["cov_trace_obj1"].max())
    assert s_cov > b_cov * 1.2


def test_pc_uses_matching_covariance_streams():
    rel_position = np.array([0.05, 0.01, 0.0])
    eta = np.array([1.0, 0.0, 0.0])
    zeta = np.array([0.0, 1.0, 0.0])

    p_ref_obj1 = np.eye(6) * 1e-10
    p_ref_obj2 = np.eye(6) * 1e-10
    p_deg_obj1 = np.eye(6) * 1e-4
    p_deg_obj2 = np.eye(6) * 1e-4

    risk = compute_risk(
        rel_position,
        p_ref_obj1,
        p_ref_obj2,
        p_deg_obj1,
        p_deg_obj2,
        eta,
        zeta,
        combined_hard_body_radius=0.02,
    )

    assert risk.pc_degraded > risk.pc_reference
