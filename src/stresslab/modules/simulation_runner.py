"""Simulation Runner - top-level orchestrator.

Manages the time-stepping loop, calling modules in sequence:
  1. propagation_engine  - advance states + compute STMs
  2. covariance_engine   - propagate covariance (predict step)
  3. measurement_model   - determine if update is applied
  4. covariance_engine   - apply measurement update if applicable
  5. geometry_metrics    - compute encounter geometry
  6. risk_model          - compute collision probability
  7. decision_layer      - evaluate threshold-v1 + integrity-v1
  8. logging_engine      - record timestep

Maintains two KnowledgeStream instances per simulation:
  - reference_stream: always receives measurement updates (no outages)
  - degraded_stream: subject to outage windows (actual tracking conditions)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from stresslab.stresslab_types import (
    SimulationConfig,
    IntegrityV1State,
    KnowledgeStream,
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


def _propagate_stream(
    stream: KnowledgeStream,
    stm_obj1: np.ndarray,
    stm_obj2: np.ndarray,
    state1: np.ndarray,
    state2: np.ndarray,
    process_noise,
    dt: float,
) -> None:
    """Propagate both covariances in a KnowledgeStream (in-place)."""
    stream.cov_obj1 = propagate_covariance(
        stream.cov_obj1, stm_obj1,
        state1[:3], state1[3:6],
        process_noise, dt,
    )
    stream.cov_obj2 = propagate_covariance(
        stream.cov_obj2, stm_obj2,
        state2[:3], state2[3:6],
        process_noise, dt,
    )


def run_simulation(
    config: SimulationConfig,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
    progress: bool = False,
    step_callback=None,
    scenario_metadata: Optional[dict] = None,
) -> dict:
    """Execute a single simulation run.

    Args:
        config: full simulation configuration
        output_dir: directory for output files (if None, no files written)
        verbose: print progress
        progress: show tqdm progress bar
        step_callback: optional callable(dict) invoked after each timestep
            with current simulation values (for live display). Must not
            compute new physics; only reads from the in-memory values.
        scenario_metadata: optional presentation-only metadata from YAML config.
            Does not affect simulation results or run_id. Passed through to
            logging_engine.write_summary() for embedding in the summary JSON.

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

    # Two knowledge streams
    reference_stream = KnowledgeStream.from_initial(
        config.cov_obj1, config.cov_obj2, config.t_start,
    )
    degraded_stream = KnowledgeStream.from_initial(
        config.cov_obj1, config.cov_obj2, config.t_start,
    )

    # Decision state
    threshold_v1_trigger: Optional[float] = None
    integrity_v1_trigger: Optional[float] = None
    integrity_v1_state = IntegrityV1State.MONITOR

    logger = LoggingEngine()

    # Time loop
    times = np.arange(config.t_start, config.t_end, config.dt)

    loop_iter = enumerate(times[:-1])
    if progress:
        try:
            from tqdm import tqdm
            loop_iter = tqdm(
                loop_iter, total=len(times) - 1,
                desc="Simulation", unit="step", leave=True,
            )
        except ImportError:
            pass  # tqdm not installed; fall back silently

    for i, t in loop_iter:
        dt = config.dt

        # ---- 1. Propagation ----
        prop = propagate_step(
            state1, state2, t, dt, config.dynamics_model,
        )
        state1 = prop.state_obj1.as_array()
        state2 = prop.state_obj2.as_array()
        t_now = t + dt

        # ---- 2. Covariance prediction ----
        _propagate_stream(
            reference_stream, prop.stm_obj1, prop.stm_obj2,
            state1, state2, config.process_noise, dt,
        )
        _propagate_stream(
            degraded_stream, prop.stm_obj1, prop.stm_obj2,
            state1, state2, config.process_noise, dt,
        )

        # ---- 3. Measurement model ----
        meas1 = evaluate_measurement(
            t_now, degraded_stream.last_update_time_obj1, config.measurement,
        )
        meas2 = evaluate_measurement(
            t_now, degraded_stream.last_update_time_obj2, config.measurement,
        )

        # ---- 4. Measurement updates ----
        # Reference always gets updates when interval is met (ignore outages)
        time_since_1 = t_now - degraded_stream.last_update_time_obj1
        interval_met_1 = time_since_1 >= config.measurement.update_interval
        time_since_2 = t_now - degraded_stream.last_update_time_obj2
        interval_met_2 = time_since_2 >= config.measurement.update_interval

        if interval_met_1:
            reference_stream.cov_obj1 = measurement_update(
                reference_stream.cov_obj1, config.measurement.noise_sigma_pos,
            )
            reference_stream.last_update_time_obj1 = t_now
        if interval_met_2:
            reference_stream.cov_obj2 = measurement_update(
                reference_stream.cov_obj2, config.measurement.noise_sigma_pos,
            )
            reference_stream.last_update_time_obj2 = t_now

        # Degraded only gets updates if not in outage
        if meas1.applied:
            degraded_stream.cov_obj1 = measurement_update(
                degraded_stream.cov_obj1, config.measurement.noise_sigma_pos,
            )
            degraded_stream.last_update_time_obj1 = t_now
        if meas2.applied:
            degraded_stream.cov_obj2 = measurement_update(
                degraded_stream.cov_obj2, config.measurement.noise_sigma_pos,
            )
            degraded_stream.last_update_time_obj2 = t_now

        # ---- Maneuver injection ----
        if config.maneuver.enabled and abs(t_now - config.maneuver.execution_time) < dt:
            degraded_stream.cov_obj1 = inject_maneuver(
                degraded_stream.cov_obj1, config.maneuver.delta_v_sigma,
            )
            reference_stream.cov_obj1 = inject_maneuver(
                reference_stream.cov_obj1, config.maneuver.delta_v_sigma,
            )

        # ---- 5. Covariance metrics (from degraded stream) ----
        cov_result_1 = covariance_metrics(
            degraded_stream.cov_obj1, degraded_stream.prev_trace_obj1, dt,
        )
        cov_result_2 = covariance_metrics(
            degraded_stream.cov_obj2, degraded_stream.prev_trace_obj2, dt,
        )
        degraded_stream.prev_trace_obj1 = cov_result_1.trace
        degraded_stream.prev_trace_obj2 = cov_result_2.trace

        # ---- 6. Geometry ----
        geom = compute_geometry(prop.rel_position, prop.rel_velocity, prop.estimated_tca)

        # ---- 7. Risk ----
        risk = compute_risk(
            prop.rel_position,
            reference_stream.cov_obj1, reference_stream.cov_obj2,
            degraded_stream.cov_obj1, degraded_stream.cov_obj2,
            geom.b_plane_eta, geom.b_plane_zeta,
            config.combined_hard_body_radius,
        )

        # ---- 8. Decision ----
        max_staleness = max(meas1.time_since_last_update, meas2.time_since_last_update)
        reference_staleness_obj1 = t_now - reference_stream.last_update_time_obj1
        reference_staleness_obj2 = t_now - reference_stream.last_update_time_obj2
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
            threshold_v1_config=config.threshold_v1,
            integrity_v1_config=config.integrity_v1,
            prev_threshold_v1_trigger=threshold_v1_trigger,
            prev_integrity_v1_trigger=integrity_v1_trigger,
            prev_integrity_v1_state=integrity_v1_state,
            shiro_config=config.shiro,
        )
        threshold_v1_trigger = decision.threshold_v1_trigger_time
        integrity_v1_trigger = decision.integrity_v1_trigger_time
        integrity_v1_state = decision.integrity_v1_state

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
            reference_cov_trace_obj1=float(np.trace(reference_stream.cov_obj1)),
            reference_cov_trace_obj2=float(np.trace(reference_stream.cov_obj2)),
            reference_staleness_obj1=float(reference_staleness_obj1),
            reference_staleness_obj2=float(reference_staleness_obj2),
        )
        logger.record(step)

        # ---- Step callback (for live display) ----
        if step_callback is not None:
            step_callback({
                "t": t_now,
                "time_to_tca": geom.time_to_tca,
                "miss_distance": geom.miss_distance,
                "rel_velocity": geom.rel_velocity_at_tca,
                "staleness": max_staleness,
                "pc_reference": risk.pc_reference,
                "pc_degraded": risk.pc_degraded,
                "risk_ratio": risk.risk_ratio,
                "cov_norm": combined_cov_norm,
                "growth_rate": combined_growth,
                "threshold_v1_state": decision.threshold_v1_alert.value,
                "integrity_v1_state": decision.integrity_v1_state.value,
                "integrity_v1_score": decision.integrity_v1_score,
                "threshold_v1_trigger": threshold_v1_trigger,
                "integrity_v1_trigger": integrity_v1_trigger,
                "shiro_state": decision.shiro_state.value if decision.shiro_state else None,
                "shiro_trigger_path": decision.shiro_trigger_path,
            })

        if verbose and (i + 1) % 100 == 0:
            print(f"  t={t_now:.0f}s | miss={geom.miss_distance:.4f}km | "
                  f"Pc={risk.pc_degraded:.2e} | integrity_v1={integrity_v1_state.value}")

    # ---- Write outputs ----
    result = {
        "logger": logger,
        "run_id": run_id,
        "threshold_v1_trigger_time": threshold_v1_trigger,
        "integrity_v1_trigger_time": integrity_v1_trigger,
    }

    if output_dir is not None:
        ts_path = logger.write_timeseries(output_dir, run_id)
        sum_path = logger.write_summary(
            output_dir, run_id, config, threshold_v1_trigger, integrity_v1_trigger,
            scenario_metadata=scenario_metadata,
            timeseries_path=ts_path,
        )
        result["timeseries_path"] = ts_path
        result["summary_path"] = sum_path
        if verbose:
            print(f"[StressLAB] Outputs: {ts_path}, {sum_path}")

    if verbose:
        dcw = None
        if threshold_v1_trigger and integrity_v1_trigger:
            dcw = threshold_v1_trigger - integrity_v1_trigger
        print(f"[StressLAB] Done. Threshold-v1 trigger: {threshold_v1_trigger}, "
              f"Integrity-v1 trigger: {integrity_v1_trigger}, DCW: {dcw}")

    return result
