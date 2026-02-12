"""Simulation Runner - top-level orchestrator.

Manages the time-stepping loop, calling modules in sequence:
  1. propagation_engine  - advance states + compute STMs
  2. covariance_engine   - propagate covariance (predict step)
  3. measurement_model   - determine if update is applied
  4. covariance_engine   - apply measurement update if applicable
  5. geometry_metrics    - compute encounter geometry
  6. risk_model          - compute collision probability
  7. decision_layer      - evaluate baseline + SHIRO
  8. logging_engine      - record timestep

Maintains two covariance streams per object:
  - baseline_cov: always receives measurement updates (no outages)
  - degraded_cov: subject to outage windows (actual tracking conditions)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from stresslab.types import (
    SimulationConfig,
    ShiroState,
    TimeStep,
    CovarianceResult,
    MeasurementResult,
    DecisionResult,
    AlertState,
)
from stresslab.modules.propagation_engine import propagate_step
from stresslab.modules.covariance_engine import (
    propagate_covariance,
    measurement_update,
    inject_maneuver,
    covariance_metrics,
)
from stresslab.modules.measurement_model import evaluate_measurement
from stresslab.modules.geometry_metrics import compute_geometry
from stresslab.modules.risk_model import compute_risk
from stresslab.modules.decision_layer import evaluate_decision
from stresslab.modules.logging_engine import LoggingEngine


def run_simulation(
    config: SimulationConfig,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> dict:
    """Execute a single simulation run.

    Args:
        config: full simulation configuration
        output_dir: directory for output files (if None, no files written)
        verbose: print progress

    Returns:
        dict with keys:
          - "logger": LoggingEngine instance
          - "summary": dict of summary metrics
          - "timeseries_path": Path (if output_dir given)
          - "summary_path": Path (if output_dir given)
    """
    rng = np.random.default_rng(config.seed)
    run_id = config.run_id()

    if verbose:
        print(f"[StressLAB] Run {run_id} | {config.dynamics_model.value} | "
              f"t=[{config.t_start}, {config.t_end}] dt={config.dt}")

    # Initialize states
    state1 = config.state_obj1.copy()
    state2 = config.state_obj2.copy()

    # Two covariance streams per object
    cov_baseline_1 = config.cov_obj1.copy()
    cov_baseline_2 = config.cov_obj2.copy()
    cov_degraded_1 = config.cov_obj1.copy()
    cov_degraded_2 = config.cov_obj2.copy()

    # Tracking
    prev_trace_1 = np.trace(cov_degraded_1)
    prev_trace_2 = np.trace(cov_degraded_2)
    last_update_time_1 = config.t_start
    last_update_time_2 = config.t_start

    # Decision state
    baseline_trigger: Optional[float] = None
    shiro_trigger: Optional[float] = None
    shiro_state = ShiroState.MONITOR

    logger = LoggingEngine()

    # Time loop
    times = np.arange(config.t_start, config.t_end, config.dt)

    for i, t in enumerate(times[:-1]):
        dt = config.dt

        # ---- 1. Propagation ----
        prop = propagate_step(
            state1, state2, t, dt, config.dynamics_model,
        )
        state1 = prop.state_obj1.as_array()
        state2 = prop.state_obj2.as_array()
        t_now = t + dt

        # ---- 2. Covariance prediction ----
        # Baseline (always updated)
        cov_baseline_1 = propagate_covariance(
            cov_baseline_1, prop.stm_obj1,
            state1[:3], state1[3:6],
            config.process_noise, dt,
        )
        cov_baseline_2 = propagate_covariance(
            cov_baseline_2, prop.stm_obj2,
            state2[:3], state2[3:6],
            config.process_noise, dt,
        )
        # Degraded (subject to outages)
        cov_degraded_1 = propagate_covariance(
            cov_degraded_1, prop.stm_obj1,
            state1[:3], state1[3:6],
            config.process_noise, dt,
        )
        cov_degraded_2 = propagate_covariance(
            cov_degraded_2, prop.stm_obj2,
            state2[:3], state2[3:6],
            config.process_noise, dt,
        )

        # ---- 3. Measurement model ----
        meas1 = evaluate_measurement(t_now, last_update_time_1, config.measurement)
        meas2 = evaluate_measurement(t_now, last_update_time_2, config.measurement)

        # ---- 4. Measurement updates ----
        # Baseline always gets updates when interval is met (ignore outages)
        time_since_1 = t_now - last_update_time_1
        interval_met_1 = time_since_1 >= config.measurement.update_interval
        time_since_2 = t_now - last_update_time_2
        interval_met_2 = time_since_2 >= config.measurement.update_interval

        if interval_met_1:
            cov_baseline_1 = measurement_update(
                cov_baseline_1, config.measurement.noise_sigma_pos,
            )
        if interval_met_2:
            cov_baseline_2 = measurement_update(
                cov_baseline_2, config.measurement.noise_sigma_pos,
            )

        # Degraded only gets updates if not in outage
        if meas1.applied:
            cov_degraded_1 = measurement_update(
                cov_degraded_1, config.measurement.noise_sigma_pos,
            )
            last_update_time_1 = t_now
        if meas2.applied:
            cov_degraded_2 = measurement_update(
                cov_degraded_2, config.measurement.noise_sigma_pos,
            )
            last_update_time_2 = t_now

        # ---- Maneuver injection ----
        if config.maneuver.enabled and abs(t_now - config.maneuver.execution_time) < dt:
            cov_degraded_1 = inject_maneuver(cov_degraded_1, config.maneuver.delta_v_sigma)
            cov_baseline_1 = inject_maneuver(cov_baseline_1, config.maneuver.delta_v_sigma)

        # ---- 5. Covariance metrics ----
        cov_result_1 = covariance_metrics(cov_degraded_1, prev_trace_1, dt)
        cov_result_2 = covariance_metrics(cov_degraded_2, prev_trace_2, dt)
        prev_trace_1 = cov_result_1.trace
        prev_trace_2 = cov_result_2.trace

        # ---- 6. Geometry ----
        geom = compute_geometry(prop.rel_position, prop.rel_velocity, prop.estimated_tca)

        # ---- 7. Risk ----
        risk = compute_risk(
            prop.rel_position,
            cov_baseline_1, cov_baseline_2,
            cov_degraded_1, cov_degraded_2,
            geom.b_plane_eta, geom.b_plane_zeta,
            config.combined_hard_body_radius,
        )

        # ---- 8. Decision ----
        # Use max staleness and combined covariance metrics
        max_staleness = max(meas1.time_since_last_update, meas2.time_since_last_update)
        combined_cov_norm = cov_result_1.trace + cov_result_2.trace
        combined_growth = cov_result_1.growth_rate + cov_result_2.growth_rate

        decision = evaluate_decision(
            pc=risk.pc_degraded,
            miss_distance=geom.miss_distance,
            time_to_tca=geom.time_to_tca,
            cov_norm=combined_cov_norm,
            growth_rate=combined_growth,
            staleness=max_staleness,
            current_time=t_now,
            baseline_config=config.baseline,
            shiro_config=config.shiro,
            prev_baseline_trigger=baseline_trigger,
            prev_shiro_trigger=shiro_trigger,
            prev_shiro_state=shiro_state,
        )
        baseline_trigger = decision.baseline_trigger_time
        shiro_trigger = decision.shiro_trigger_time
        shiro_state = decision.shiro_state

        # ---- 9. Logging ----
        step = TimeStep(
            t=t_now,
            propagation=prop,
            covariance_obj1=cov_result_1,
            covariance_obj2=cov_result_2,
            measurement_obj1=meas1,
            measurement_obj2=meas2,
            geometry=geom,
            risk=risk,
            decision=decision,
        )
        logger.record(step)

        if verbose and (i + 1) % 100 == 0:
            print(f"  t={t_now:.0f}s | miss={geom.miss_distance:.4f}km | "
                  f"Pc={risk.pc_degraded:.2e} | SHIRO={shiro_state.value}")

    # ---- Write outputs ----
    result = {
        "logger": logger,
        "run_id": run_id,
        "baseline_trigger_time": baseline_trigger,
        "shiro_trigger_time": shiro_trigger,
    }

    if output_dir is not None:
        ts_path = logger.write_timeseries(output_dir, run_id)
        sum_path = logger.write_summary(
            output_dir, run_id, config, baseline_trigger, shiro_trigger,
        )
        result["timeseries_path"] = ts_path
        result["summary_path"] = sum_path
        if verbose:
            print(f"[StressLAB] Outputs: {ts_path}, {sum_path}")

    if verbose:
        dcw = None
        if baseline_trigger and shiro_trigger:
            dcw = baseline_trigger - shiro_trigger
        print(f"[StressLAB] Done. Baseline trigger: {baseline_trigger}, "
              f"SHIRO trigger: {shiro_trigger}, DCW: {dcw}")

    return result
