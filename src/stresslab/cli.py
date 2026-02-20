"""CLI entry point for StressLAB."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

from stresslab import __version__
from stresslab.modules.scenario_generator import generate_default_scenario, export_scenario
from stresslab.modules.simulation_runner import run_simulation
from stresslab.modules.synth_case import generate_synthetic_case
from stresslab.modules.monte_carlo import run_monte_carlo
from stresslab.stresslab_types import DynamicsModel


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

def _load_config_yaml(path: Path):
    """Load a YAML scenario file and return (SimulationConfig, scenario_metadata).

    The ``scenario_metadata`` dict is presentation-only and must NOT affect
    the SimulationConfig hash / run_id.  It is extracted from the optional
    ``scenario_metadata`` YAML section and contains:
      run_label, scenario_title, scenario_purpose, scenario_takeaway, tags
    If the section is absent, ``None`` is returned for scenario_metadata.
    """
    import yaml
    import numpy as np
    from stresslab.stresslab_types import (
        SimulationConfig, ProcessNoiseConfig, MeasurementConfig,
        ManeuverConfig, ThresholdV1Config, IntegrityV1Config, ShiroConfig,
    )

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Extract presentation-only scenario metadata (does NOT affect run_id)
    sm_raw = raw.get("scenario_metadata")
    scenario_metadata = None
    if sm_raw and isinstance(sm_raw, dict):
        scenario_metadata = {
            "run_label": sm_raw.get("run_label"),
            "scenario_title": sm_raw.get("scenario_title"),
            "scenario_purpose": sm_raw.get("scenario_purpose"),
            "scenario_takeaway": sm_raw.get("scenario_takeaway"),
            "tags": sm_raw.get("tags", []),
        }
    # SHIRO demo format: top-level purpose/name for display
    if scenario_metadata is None and (raw.get("purpose") or raw.get("name")):
        scenario_metadata = {
            "run_label": raw.get("name"),
            "scenario_purpose": raw.get("purpose"),
            "scenario_title": None,
            "scenario_takeaway": None,
            "tags": [],
        }

    # Build sub-configs from YAML sections
    pn_raw = raw.get("process_noise", {})
    # Support SHIRO demo process_noise.sigmas_mps2 (m/s^2 -> use as scale factors; keep sigma_* in km)
    sigma_r = pn_raw.get("sigma_radial")
    sigma_t = pn_raw.get("sigma_tangential")
    sigma_n = pn_raw.get("sigma_normal")
    if sigma_r is None and "sigmas_mps2" in pn_raw:
        s = pn_raw["sigmas_mps2"]
        # sigma in m/s^2 -> km/s^2 (ProcessNoiseConfig uses km)
        def mps2_to_kmps2(x):
            return float(x) * 1e-3
        sigma_r = mps2_to_kmps2(s.get("radial", 1e-6))
        sigma_t = mps2_to_kmps2(s.get("tangential", 1e-6))
        sigma_n = mps2_to_kmps2(s.get("normal", 1e-6))
    pn = ProcessNoiseConfig(
        sigma_radial=float(pn_raw.get("sigma_radial", sigma_r or 1e-9)),
        sigma_tangential=float(pn_raw.get("sigma_tangential", sigma_t or 1e-9)),
        sigma_normal=float(pn_raw.get("sigma_normal", sigma_n or 1e-9)),
        scale=float(pn_raw.get("global_scale", pn_raw.get("scale", 1.0))),
    )

    meas_raw = raw.get("measurement", {})
    # SHIRO demo format: tracking_model with measurement_cadence_s and outages in hours
    tr_raw = raw.get("tracking_model", {})
    if tr_raw:
        cadence = tr_raw.get("measurement_cadence_s", meas_raw.get("update_interval", 3600.0))
        noise_m = tr_raw.get("measurement_noise", {})
        noise_pos_km = (noise_m.get("position_sigma_m", 50) / 1000.0) if noise_m else meas_raw.get("noise_sigma_pos", 0.01)
        outage_windows = []
        for o in tr_raw.get("outages", []):
            sh = o.get("start_hours_from_start")
            eh = o.get("end_hours_from_start")
            if sh is not None and eh is not None:
                outage_windows.append({"start": sh * 3600.0, "end": eh * 3600.0})
        if outage_windows:
            meas_raw = {**meas_raw, "update_interval": cadence, "noise_sigma_pos": noise_pos_km, "outage_windows": outage_windows}
    meas = MeasurementConfig(
        update_interval=meas_raw.get("update_interval", 3600.0),
        noise_sigma_pos=meas_raw.get("noise_sigma_pos", 0.01),
        outage_windows=meas_raw.get("outage_windows", []),
    )

    man_raw = raw.get("maneuver", {})
    maneuver = ManeuverConfig(
        enabled=man_raw.get("enabled", False),
        delta_v_sigma=man_raw.get("delta_v_sigma", 0.001),
        execution_time=man_raw.get("execution_time", 0.0),
    )

    tv1_raw = raw.get("threshold_v1_decision", {})
    # SHIRO demo: decision_models.baseline.critical_trigger
    dm_baseline = (raw.get("decision_models") or {}).get("baseline", {})
    ct = dm_baseline.get("critical_trigger", tv1_raw)
    pc_crit = ct.get("pc_critical", tv1_raw.get("pc_threshold", 1e-4))
    d_act_km = None
    if "d_act_m" in ct:
        d_act_km = ct["d_act_m"] / 1000.0
    elif tv1_raw.get("miss_distance_threshold") is not None:
        d_act_km = tv1_raw["miss_distance_threshold"]
    tv1 = ThresholdV1Config(
        pc_threshold=pc_crit,
        miss_distance_threshold=d_act_km if d_act_km is not None else tv1_raw.get("miss_distance_threshold"),
        time_to_tca_gate=tv1_raw.get("time_to_tca_gate", 86400.0),
    )

    iv1_raw = raw.get("integrity_v1_decision", {})
    iv1 = IntegrityV1Config(
        weights=np.array(iv1_raw.get("weights", [0.4, 0.2, 0.2, 0.2])),
        threshold_monitor_to_watch=iv1_raw.get("threshold_monitor_to_watch", 0.25),
        threshold_watch_to_warning=iv1_raw.get("threshold_watch_to_warning", 0.50),
        threshold_warning_to_critical=iv1_raw.get("threshold_warning_to_critical", 0.75),
        pc_ref=iv1_raw.get("pc_ref", 1e-4),
        cov_norm_ref=iv1_raw.get("cov_norm_ref", 100.0),
        growth_rate_ref=iv1_raw.get("growth_rate_ref", 1.0),
        staleness_ref=iv1_raw.get("staleness_ref", 86400.0),
    )

    # SHIRO config from shiro_decision or decision_models.shiro
    shiro_config = None
    shiro_raw = raw.get("shiro_decision") or (raw.get("decision_models") or {}).get("shiro", {})
    if shiro_raw:
        gating = shiro_raw.get("gating", {})
        geom = gating.get("geometry_relevant_requires", {})
        d_watch_m = geom.get("d_watch_m", 8000)
        elev = shiro_raw.get("elevated_triggers", [])
        dt_max_s = 3600.0
        for e in elev:
            if e.get("type") == "freshness" and e.get("dt_since_last_update_gte_s") is not None:
                dt_max_s = e["dt_since_last_update_gte_s"]
                break
        crit = shiro_raw.get("critical_trigger", {})
        d_act_m = crit.get("d_act_m", 1000)
        shiro_config = ShiroConfig(
            d_watch_km=d_watch_m / 1000.0,
            d_act_km=d_act_m / 1000.0,
            pc_critical=crit.get("pc_critical", 1e-4),
            dt_max_s=dt_max_s,
        )

    tl = raw.get("timeline", {})
    risk = raw.get("risk", {})
    # SHIRO demo: scenario.duration_hours, scenario.tca_time_hours_from_start, encounter_geometry
    scenario_raw = raw.get("scenario", {})
    enc_raw = raw.get("encounter_geometry", {})
    seed = scenario_raw.get("seed", raw.get("seed", 42))
    t_end = tl.get("t_end")
    if t_end is None and scenario_raw.get("duration_hours") is not None:
        t_end = scenario_raw["duration_hours"] * 3600.0
    t_end = t_end or 259200.0
    dt = tl.get("dt", 60.0)
    t_tca_frac = tl.get("t_tca_frac")
    if t_tca_frac is None and scenario_raw.get("tca_time_hours_from_start") is not None and scenario_raw.get("duration_hours"):
        t_tca_frac = scenario_raw["tca_time_hours_from_start"] / scenario_raw["duration_hours"]
    t_tca_frac = t_tca_frac if t_tca_frac is not None else 0.6
    miss_km = raw.get("miss_distance_km")
    if miss_km is None and enc_raw.get("target_miss_distance_m") is not None:
        miss_km = enc_raw["target_miss_distance_m"] / 1000.0
    miss_km = miss_km if miss_km is not None else 0.5

    config = generate_default_scenario(
        seed=int(seed),
        miss_distance_km=float(miss_km),
        t_end=float(t_end),
        dt=float(dt),
        dynamics=DynamicsModel(raw.get("dynamics_model", scenario_raw.get("dynamics_model", "two_body_plus_J2"))),
        t_tca_frac=float(t_tca_frac),
    )

    config.process_noise = pn
    config.measurement = meas
    config.maneuver = maneuver
    config.threshold_v1 = tv1
    config.integrity_v1 = iv1
    config.shiro = shiro_config
    config.t_start = tl.get("t_start", 0.0)
    config.combined_hard_body_radius = risk.get("combined_hard_body_radius", 0.02)

    return config, scenario_metadata


# ---------------------------------------------------------------------------
# Helper: resolve parameter path on config
# ---------------------------------------------------------------------------

def _set_config_param(config, param_path: str, value):
    """Set a nested config parameter by dot-separated path.

    Supported paths:
        outage.duration   -> sets a single outage window of the given duration (seconds)
        outage.start      -> sets start of single outage window
        process_noise.scale -> config.process_noise.scale
        measurement.update_interval -> config.measurement.update_interval
        measurement.noise_sigma_pos -> config.measurement.noise_sigma_pos
        combined_hard_body_radius -> config.combined_hard_body_radius
        dt -> config.dt
        miss_distance_km -> re-generates scenario with that miss distance
    """
    import copy
    cfg = copy.deepcopy(config)

    parts = param_path.split(".")
    if parts[0] == "outage" and parts[1] == "duration":
        duration = float(value)
        # Place a single outage window starting at 15% of the timeline
        start = cfg.t_start + (cfg.t_end - cfg.t_start) * 0.15
        cfg.measurement.outage_windows = [
            {"start": start, "end": start + duration}
        ] if duration > 0 else []
    elif parts[0] == "outage" and parts[1] == "start":
        start = float(value)
        # Set the start of a single outage window, preserving existing duration
        # or using a default of 6 hours if no window exists yet
        existing = cfg.measurement.outage_windows
        if existing and len(existing) > 0:
            duration = existing[0]["end"] - existing[0]["start"]
        else:
            duration = 21600.0  # default 6h
        cfg.measurement.outage_windows = [
            {"start": start, "end": start + duration}
        ] if start >= 0 else []
    elif parts[0] == "process_noise" and len(parts) == 2:
        setattr(cfg.process_noise, parts[1], float(value))
    elif parts[0] == "measurement" and len(parts) == 2:
        setattr(cfg.measurement, parts[1], float(value))
    elif param_path == "combined_hard_body_radius":
        cfg.combined_hard_body_radius = float(value)
    elif param_path == "dt":
        cfg.dt = float(value)
    else:
        raise click.ClickException(
            f"Unknown sweep parameter: {param_path}. "
            "Supported: outage.duration, outage.start, process_noise.scale, "
            "measurement.update_interval, combined_hard_body_radius, dt"
        )
    return cfg


def _parse_sweep_value(v: str) -> float:
    """Parse a sweep value string, handling time units (h, m, s)."""
    v = v.strip()
    if v.endswith("h"):
        return float(v[:-1]) * 3600.0
    if v.endswith("m"):
        return float(v[:-1]) * 60.0
    if v.endswith("s"):
        return float(v[:-1])
    return float(v)


# ---------------------------------------------------------------------------
# Rich helpers (lazy imports)
# ---------------------------------------------------------------------------

def _get_console():
    """Get a Rich console, or None if Rich is unavailable."""
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def _rich_table_or_print(headers, rows, title=None):
    """Print a Rich table if available, otherwise a plain text table."""
    console = _get_console()
    if console is not None:
        from rich.table import Table
        table = Table(title=title, show_lines=False)
        for h in headers:
            table.add_column(h, justify="right" if h != headers[0] else "left")
        for row in rows:
            table.add_row(*[str(x) for x in row])
        console.print(table)
    else:
        # Plain text fallback
        if title:
            click.echo(f"\n{title}")
            click.echo("-" * len(title))
        widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
        hdr_line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
        click.echo(hdr_line)
        click.echo("-" * len(hdr_line))
        for row in rows:
            click.echo("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version=__version__)
def main():
    """StressLAB - Conjunction Decision Stress-Testing Platform."""
    pass


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--config", "config_path", default=None,
              type=click.Path(exists=True), help="YAML scenario config file")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--miss-distance", default=0.5, type=float, help="Target miss distance (km)")
@click.option("--t-end", default=259200.0, type=float, help="Simulation end time (s)")
@click.option("--dt", default=60.0, type=float, help="Timestep (s)")
@click.option("--dynamics", default="two_body_plus_J2",
              type=click.Choice(["two_body", "two_body_plus_J2"]))
@click.option("--output-dir", default="outputs", type=click.Path())
@click.option("--verbose", is_flag=True)
@click.option("--progress", is_flag=True, help="Show live progress bar")
@click.option("--live", is_flag=True, help="Show Rich live instrument panel")
def run(config_path, seed, miss_distance, t_end, dt, dynamics, output_dir,
        verbose, progress, live):
    """Run a single conjunction stress-test simulation."""
    output = Path(output_dir)

    scenario_metadata = None
    if config_path:
        config, scenario_metadata = _load_config_yaml(Path(config_path))
    else:
        config = generate_default_scenario(
            seed=seed,
            miss_distance_km=miss_distance,
            t_end=t_end,
            dt=dt,
            dynamics=DynamicsModel(dynamics),
        )

    # Export scenario
    scenario_path = export_scenario(config, output)
    if verbose:
        click.echo(f"Scenario config: {scenario_path}")

    # Live panel setup
    panel = None
    step_callback = None
    if live:
        try:
            from stresslab.live_panel import LivePanelCallback
            panel = LivePanelCallback(
                run_id=config.run_id(),
                seed=config.seed,
                dt=config.dt,
                t_end=config.t_end,
                dynamics=config.dynamics_model.value,
            )
            panel.start()
            step_callback = panel.on_step
        except ImportError:
            click.echo("[WARN] Rich not installed; --live disabled.", err=True)

    try:
        result = run_simulation(
            config, output_dir=output, verbose=verbose,
            progress=progress and not live,
            step_callback=step_callback,
            scenario_metadata=scenario_metadata,
        )
    finally:
        if panel is not None:
            panel.stop()

    click.echo(f"\nRun ID: {result['run_id']}")
    click.echo(f"Threshold-v1 trigger: {result['threshold_v1_trigger_time']}")
    click.echo(f"Integrity-v1 trigger: {result['integrity_v1_trigger_time']}")
    if result.get("timeseries_path"):
        click.echo(f"Time series:          {result['timeseries_path']}")
    if result.get("summary_path"):
        click.echo(f"Summary:              {result['summary_path']}")


# ---------------------------------------------------------------------------
# monte-carlo command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--config", "config_path", default=None,
              type=click.Path(exists=True), help="YAML scenario config file")
@click.option("--seed", default=42, type=int, help="Base random seed")
@click.option("--n-runs", default=100, type=int, help="Number of Monte Carlo runs")
@click.option("--miss-distance", default=0.5, type=float, help="Target miss distance (km)")
@click.option("--t-end", default=86400.0, type=float, help="Simulation end time (s)")
@click.option("--dt", default=120.0, type=float, help="Timestep (s)")
@click.option("--output-dir", default="outputs/monte_carlo", type=click.Path())
@click.option("--verbose", is_flag=True)
@click.option("--progress", is_flag=True, help="Show live progress bar")
def monte_carlo(config_path, seed, n_runs, miss_distance, t_end, dt,
                output_dir, verbose, progress):
    """Run Monte Carlo batch analysis."""
    output = Path(output_dir)

    if config_path:
        base_config, _metadata = _load_config_yaml(Path(config_path))
    else:
        base_config = generate_default_scenario(
            seed=seed,
            miss_distance_km=miss_distance,
            t_end=t_end,
            dt=dt,
        )

    df = run_monte_carlo(
        base_config,
        n_runs=n_runs,
        output_dir=output,
        verbose=verbose,
        progress=progress,
    )

    click.echo(f"\nMonte Carlo complete: {len(df)} runs")
    click.echo(f"Output directory: {output}")

    # Print summary statistics
    good = df.dropna(subset=["threshold_v1_trigger_time", "integrity_v1_trigger_time"], how="all")
    if "decision_compression_window" in good.columns:
        dcw = good["decision_compression_window"].dropna()
        if len(dcw) > 0:
            click.echo(f"\nDecision Compression Window:")
            click.echo(f"  Mean:   {dcw.mean():.1f}s")
            click.echo(f"  Std:    {dcw.std():.1f}s")
            click.echo(f"  Median: {dcw.median():.1f}s")


# ---------------------------------------------------------------------------
# sweep command
# ---------------------------------------------------------------------------

@main.command()
@click.argument("param_path")
@click.option("--values", required=True, type=str,
              help="Comma-separated sweep values (supports h/m/s suffixes, e.g. 0h,1h,3h,6h,12h)")
@click.option("--config", "config_path", default=None,
              type=click.Path(exists=True), help="YAML scenario config file")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--miss-distance", default=0.5, type=float, help="Target miss distance (km)")
@click.option("--t-end", default=86400.0, type=float, help="Simulation end time (s)")
@click.option("--dt", default=120.0, type=float, help="Timestep (s)")
@click.option("--output-dir", default="outputs/sweep", type=click.Path())
@click.option("--live", is_flag=True, help="Show Rich live display for each run")
@click.option("--progress", is_flag=True, help="Show progress")
def sweep(param_path, values, config_path, seed, miss_distance, t_end, dt, output_dir, live, progress):
    """Run a controlled parameter sweep.

    PARAM_PATH is the dot-separated config path to sweep, e.g. outage.duration

    Supported parameters:
      outage.duration, outage.start, process_noise.scale,
      measurement.update_interval, combined_hard_body_radius, dt

    Example:
      stresslab sweep outage.duration --values 0h,1h,3h,6h,12h --seed 42
    """
    import hashlib
    import copy
    import pandas as pd

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Parse values
    raw_values = [v.strip() for v in values.split(",") if v.strip()]
    if not raw_values:
        raise click.ClickException("--values must contain at least one non-empty value.")
    parsed_values = [_parse_sweep_value(v) for v in raw_values]

    # Load base config
    if config_path:
        base_config, _metadata = _load_config_yaml(Path(config_path))
    else:
        base_config = generate_default_scenario(
            seed=seed, miss_distance_km=miss_distance, t_end=t_end, dt=dt,
        )

    # Sweep metadata
    from stresslab.metrics_contract import METRICS_CONTRACT_VERSION
    from stresslab.modules.logging_engine import SCHEMA_VERSION

    sweep_results = []
    n_values = len(parsed_values)

    for idx, (raw_val, parsed_val) in enumerate(zip(raw_values, parsed_values)):
        if progress:
            click.echo(f"\n[Sweep {idx+1}/{n_values}] {param_path} = {raw_val} ({parsed_val})")

        cfg = _set_config_param(base_config, param_path, parsed_val)
        cfg.seed = seed  # Keep seed consistent

        run_output = output / f"run_{idx:03d}_{raw_val}"

        # Live panel setup for each sweep run
        panel = None
        step_callback = None
        if live:
            try:
                from stresslab.live_panel import LivePanelCallback
                panel = LivePanelCallback(
                    run_id=cfg.run_id(),
                    seed=cfg.seed,
                    dt=cfg.dt,
                    t_end=cfg.t_end,
                    dynamics=cfg.dynamics_model.value,
                )
                panel.start()
                step_callback = panel.on_step
            except ImportError:
                click.echo("[WARN] Rich not installed; --live disabled.", err=True)
                live = False  # Don't retry for subsequent runs

        try:
            result = run_simulation(
                cfg, output_dir=run_output, verbose=False,
                progress=progress and not live,
                step_callback=step_callback,
            )
        except Exception as e:
            if panel is not None:
                panel.stop()
            click.echo(f"  [WARN] Sweep run {idx+1} failed: {e}", err=True)
            sweep_results.append({
                "sweep_index": idx,
                "param_path": param_path,
                "param_value_raw": raw_val,
                "param_value": parsed_val,
                "run_id": None,
                "seed": seed,
                "error": str(e),
            })
            continue
        finally:
            if panel is not None:
                panel.stop()

        # Extract key metrics from logger
        logger = result["logger"]
        summary = logger.compute_summary(
            cfg, result["threshold_v1_trigger_time"], result["integrity_v1_trigger_time"],
        )

        row = {
            "sweep_index": idx,
            "param_path": param_path,
            "param_value_raw": raw_val,
            "param_value": parsed_val,
            "run_id": result["run_id"],
            "seed": seed,
            "threshold_v1_trigger_time": summary.threshold_v1_trigger_time,
            "integrity_v1_trigger_time": summary.integrity_v1_trigger_time,
            "decision_compression_window": summary.decision_compression_window,
            "false_safe_rate": summary.false_safe_rate,
            "false_alert_rate": summary.false_alert_rate,
            "max_pc_degraded": summary.max_pc_degraded,
            "max_cov_trace": summary.max_cov_trace,
            "max_staleness": summary.max_staleness,
            "decision_instability_index": summary.decision_instability_index,
            "mean_pc_drift": summary.mean_pc_drift,
            "staleness_pc_correlation": summary.staleness_pc_correlation,
            "outage_sensitivity_score": summary.outage_sensitivity_score,
        }
        sweep_results.append(row)

    sweep_df = pd.DataFrame(sweep_results)

    # Write sweep summary CSV
    csv_path = output / "sweep_summary.csv"
    sweep_df.to_csv(csv_path, index=False)

    # Write sweep summary JSON with schema versions
    sweep_json = {
        "schema_version": SCHEMA_VERSION,
        "metrics_contract_version": METRICS_CONTRACT_VERSION,
        "stresslab_version": __version__,
        "sweep_param": param_path,
        "sweep_values_raw": raw_values,
        "sweep_values": parsed_values,
        "seed": seed,
        "results": sweep_results,
    }
    json_path = output / "sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump(sweep_json, f, indent=2, default=_json_default)

    # Compute deterministic hash for reproducibility check
    content_hash = hashlib.sha256(
        json.dumps(sweep_results, sort_keys=True, default=_json_default).encode()
    ).hexdigest()[:16]
    click.echo(f"\nSweep hash: {content_hash}")

    # Print Rich table
    headers = [
        param_path, "T-v1 trigger", "I-v1 trigger", "DCW",
        "False-safe", "Max Pc_d", "Instability",
    ]
    rows = []
    for r in sweep_results:
        if "error" in r:
            rows.append([
                r["param_value_raw"],
                "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR",
            ])
            continue
        rows.append([
            r["param_value_raw"],
            f"{r['threshold_v1_trigger_time']:.0f}s" if r["threshold_v1_trigger_time"] else "---",
            f"{r['integrity_v1_trigger_time']:.0f}s" if r["integrity_v1_trigger_time"] else "---",
            f"{r['decision_compression_window']:.0f}s" if r["decision_compression_window"] else "---",
            f"{r['false_safe_rate']:.3f}",
            f"{r['max_pc_degraded']:.2e}",
            f"{r['decision_instability_index']:.4f}",
        ])

    _rich_table_or_print(headers, rows, title=f"Sweep: {param_path}")

    click.echo(f"\nSweep outputs: {csv_path}")
    click.echo(f"              {json_path}")


# ---------------------------------------------------------------------------
# report command (enhanced with plots)
# ---------------------------------------------------------------------------

@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output-dir", "--out", default=None, type=click.Path(),
              help="Output directory (default: inside input path)")
@click.option("--formats", default="png,csv,json",
              help="Comma-separated output formats: png,csv,json,pdf")
def report(input_path, output_dir, formats):
    """Generate reports and plots from a run, sweep, or MC folder.

    INPUT_PATH can be:
      - A run summary JSON file
      - A sweep output folder (containing sweep_summary.json)
      - A Monte Carlo output folder (containing monte_carlo_summary.csv)
    """
    from stresslab.reporting.compare import load_summary_json
    from stresslab.reporting.benchmark_report import generate_benchmark_report
    from stresslab.reporting.monte_carlo_report import generate_monte_carlo_report
    from stresslab.reporting.exporters import export_json, export_csv
    from stresslab.metrics_contract import MetricsSummary

    input_p = Path(input_path)
    fmt_set = set(f.strip().lower() for f in formats.split(","))

    # Determine output dir
    if output_dir:
        out = Path(output_dir)
    elif input_p.is_dir():
        out = input_p / "reports"
    else:
        out = input_p.parent / "reports"
    out.mkdir(parents=True, exist_ok=True)

    # Detect input type
    if input_p.is_file() and input_p.suffix == ".json":
        _report_single_run(input_p, out, fmt_set)
    elif input_p.is_dir() and (input_p / "sweep_summary.json").exists():
        _report_sweep(input_p, out, fmt_set)
    elif input_p.is_dir() and (input_p / "monte_carlo_summary.csv").exists():
        _report_monte_carlo(input_p, out, fmt_set)
    elif input_p.is_file() and input_p.suffix == ".csv":
        # MC summary CSV
        _report_monte_carlo(input_p.parent, out, fmt_set)
    else:
        # Try as single run summary
        _report_single_run(input_p, out, fmt_set)

    click.echo(f"\nReport outputs: {out}")


def _report_single_run(json_path: Path, out: Path, fmt_set: set):
    """Generate single-run benchmark report + plots."""
    from stresslab.reporting.compare import load_summary_json
    from stresslab.reporting.benchmark_report import generate_benchmark_report
    from stresslab.reporting.exporters import export_json, export_csv
    from stresslab.metrics_contract import MetricsSummary

    raw = load_summary_json(json_path)

    try:
        summary = MetricsSummary(**{
            k: raw[k] for k in MetricsSummary.__dataclass_fields__
        })
        bench_report = generate_benchmark_report(summary)
    except (KeyError, TypeError):
        bench_report = raw

    if "json" in fmt_set:
        export_json(bench_report, out / "report.json")
    if "csv" in fmt_set:
        export_csv(bench_report, out / "summary_table.csv")

    # Generate plots from companion timeseries if available
    ts_path = json_path.parent / json_path.name.replace("summary_", "timeseries_").replace(".json", ".parquet")
    if ts_path.exists() and "png" in fmt_set:
        try:
            import pandas as pd
            ts_df = pd.read_parquet(ts_path)
            from stresslab.plotting import (
                plot_posture_timeline, plot_pc_drift_vs_staleness,
                plot_decision_instability,
            )
            plot_posture_timeline(ts_df, out / "posture_timeline_overlay.png")
            plot_pc_drift_vs_staleness(ts_df, out / "pc_drift_vs_staleness.png")
            plot_decision_instability(ts_df, out / "decision_instability.png")
            click.echo(f"  Generated: posture_timeline_overlay.png, pc_drift_vs_staleness.png, decision_instability.png")
        except Exception as e:
            click.echo(f"  [WARN] Plot generation failed: {e}", err=True)

    click.echo(f"  Single-run report generated.")


def _report_sweep(sweep_dir: Path, out: Path, fmt_set: set):
    """Generate sweep report + plots."""
    import pandas as pd
    from stresslab.reporting.exporters import export_json, export_csv

    json_path = sweep_dir / "sweep_summary.json"
    with open(json_path) as f:
        sweep_data = json.load(f)

    if "json" in fmt_set:
        export_json(sweep_data, out / "sweep_report.json")

    if "csv" in fmt_set:
        csv_path = sweep_dir / "sweep_summary.csv"
        if csv_path.exists():
            import shutil
            shutil.copy2(csv_path, out / "summary_table.csv")

    if "png" in fmt_set:
        try:
            results = sweep_data.get("results", [])
            param_name = sweep_data.get("sweep_param", "parameter")
            from stresslab.plotting import (
                plot_outage_vs_trigger_shift,
                plot_outage_vs_false_safe_rate,
            )
            plot_outage_vs_trigger_shift(results, param_name, out / "outage_vs_trigger_shift.png")
            plot_outage_vs_false_safe_rate(results, param_name, out / "outage_vs_false_safe_rate.png")
            click.echo(f"  Generated: outage_vs_trigger_shift.png, outage_vs_false_safe_rate.png")
        except Exception as e:
            click.echo(f"  [WARN] Sweep plot generation failed: {e}", err=True)

    click.echo(f"  Sweep report generated.")


def _report_monte_carlo(mc_dir: Path, out: Path, fmt_set: set):
    """Generate Monte Carlo report + plots."""
    import pandas as pd
    from stresslab.reporting.monte_carlo_report import generate_monte_carlo_report
    from stresslab.reporting.exporters import export_json, export_csv

    csv_path = mc_dir / "monte_carlo_summary.csv"
    if not csv_path.exists():
        click.echo(f"  [WARN] No monte_carlo_summary.csv in {mc_dir}", err=True)
        return

    mc_df = pd.read_csv(csv_path)
    mc_report = generate_monte_carlo_report(mc_df)

    if "json" in fmt_set:
        export_json(mc_report, out / "mc_report.json")
    if "csv" in fmt_set:
        export_csv(mc_report, out / "model_comparison.csv")

    if "png" in fmt_set:
        try:
            from stresslab.plotting import plot_compression_window_distribution
            dcw_vals = mc_df["decision_compression_window"].dropna().tolist()
            if dcw_vals:
                plot_compression_window_distribution(dcw_vals, out / "compression_window_distribution.png")
                click.echo(f"  Generated: compression_window_distribution.png")
        except Exception as e:
            click.echo(f"  [WARN] MC plot generation failed: {e}", err=True)

    click.echo(f"  Monte Carlo report generated.")


# ---------------------------------------------------------------------------
# compare command (enhanced with Rich tables + auto-detect)
# ---------------------------------------------------------------------------

@main.command()
@click.argument("path_a", type=click.Path(exists=True))
@click.argument("path_b", type=click.Path(exists=True))
@click.option("--metrics", default=None, type=str,
              help="Comma-separated metrics to compare (default: all)")
@click.option("--output-dir", "--out", default=None, type=click.Path(),
              help="Output directory for comparison files")
def compare(path_a, path_b, metrics, output_dir):
    """Compare two simulation runs or sweep folders.

    PATH_A and PATH_B can be:
      - Run summary JSON files
      - Sweep output folders (auto-detected)
    """
    from stresslab.reporting.compare import (
        compare_from_files, comparison_to_dataframe, load_summary_json,
    )
    from stresslab.reporting.exporters import export_json, export_csv

    pa = Path(path_a)
    pb = Path(path_b)

    # Auto-detect: if directories, look for sweep_summary.json or a summary JSON
    if pa.is_dir():
        pa = _find_summary_in_dir(pa)
    if pb.is_dir():
        pb = _find_summary_in_dir(pb)

    comparison = compare_from_files(pa, pb)

    # Filter metrics if specified
    if metrics:
        wanted = set(m.strip() for m in metrics.split(","))
        comparison["metrics"] = [
            m for m in comparison.get("metrics", [])
            if m["metric"] in wanted
        ]

    # Print Rich table
    metric_list = comparison.get("metrics", [])
    if metric_list:
        label_a = comparison.get("label_a", "run_a")
        label_b = comparison.get("label_b", "run_b")
        headers = ["Metric", label_a, label_b, "Delta", "Rel %"]
        rows = []
        for m in metric_list:
            va = m.get(f"value_{label_a}", "---")
            vb = m.get(f"value_{label_b}", "---")
            delta = m.get("delta", 0)
            rel_pct = m.get("rel_delta_pct", 0)
            rows.append([
                m["metric"],
                _fmt_num(va),
                _fmt_num(vb),
                _fmt_num(delta),
                f"{rel_pct:+.2f}%",
            ])
        _rich_table_or_print(headers, rows, title="Run Comparison")

    # Write outputs
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        export_json(comparison, out / "comparison.json")
        comp_df = comparison_to_dataframe(comparison)
        export_csv(comp_df, out / "comparison.csv")
        click.echo(f"\nComparison written to: {out}")


def _find_summary_in_dir(d: Path) -> Path:
    """Find a summary JSON in a directory (sweep or run folder)."""
    # Sweep folder
    sweep_json = d / "sweep_summary.json"
    if sweep_json.exists():
        return sweep_json
    # MC stats
    mc_json = d / "monte_carlo_stats.json"
    if mc_json.exists():
        return mc_json
    # Find any summary_*.json
    summaries = sorted(d.glob("summary_*.json"))
    if summaries:
        return summaries[0]
    raise click.ClickException(f"No summary JSON found in {d}")


def _fmt_num(v) -> str:
    """Format a numeric value for table display."""
    if v is None or v == "---":
        return "---"
    if isinstance(v, float):
        if abs(v) < 0.001 or abs(v) > 1e6:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# doctor command (enhanced with Rich formatting + extra checks)
# ---------------------------------------------------------------------------

@main.command()
def doctor():
    """Check the StressLAB installation and environment."""
    console = _get_console()
    all_ok = True
    warnings = 0

    def _ok(msg):
        if console:
            console.print(f"  [green][OK][/green]   {msg}")
        else:
            click.echo(f"  [OK]   {msg}")

    def _warn(msg):
        nonlocal warnings
        warnings += 1
        if console:
            console.print(f"  [yellow][WARN][/yellow] {msg}")
        else:
            click.echo(f"  [WARN] {msg}")

    def _fail(msg):
        nonlocal all_ok
        all_ok = False
        if console:
            console.print(f"  [red][FAIL][/red] {msg}")
        else:
            click.echo(f"  [FAIL] {msg}")

    if console:
        console.print("[bold blue]StressLAB Doctor[/bold blue]")
        console.print("=" * 50)
    else:
        click.echo("StressLAB Doctor")
        click.echo("=" * 50)

    # 1. Version
    click.echo(f"  stresslab version:  {__version__}")

    # 2. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    click.echo(f"  Python version:     {py_ver}")
    if sys.version_info < (3, 10):
        _fail("Python >= 3.10 required")
    else:
        _ok("Python >= 3.10")

    # 3. Core dependencies
    deps = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("pyarrow", "pyarrow"),
        ("yaml", "pyyaml"),
        ("click", "click"),
        ("rich", "rich"),
    ]
    for import_name, pkg_name in deps:
        try:
            __import__(import_name)
            try:
                from importlib.metadata import version as _pkg_version
                ver = _pkg_version(pkg_name)
            except Exception:
                ver = "unknown"
            _ok(f"{pkg_name} ({ver})")
        except ImportError:
            _fail(f"{pkg_name} not found")

    # 4. Optional dependencies
    opt_deps = [
        ("tqdm", "tqdm", "progress bars"),
        ("matplotlib", "matplotlib", "plotting / report generation"),
    ]
    for import_name, pkg_name, purpose in opt_deps:
        try:
            __import__(import_name)
            try:
                from importlib.metadata import version as _pkg_version
                ver = _pkg_version(pkg_name)
            except Exception:
                ver = "unknown"
            _ok(f"{pkg_name} ({ver}) - {purpose}")
        except ImportError:
            _warn(f"{pkg_name} not installed - {purpose} unavailable")

    # 5. Metrics contract version
    from stresslab.metrics_contract import METRICS_CONTRACT_VERSION
    click.echo(f"  Contract version:   {METRICS_CONTRACT_VERSION}")

    # 6. Schema version
    from stresslab.modules.logging_engine import SCHEMA_VERSION
    click.echo(f"  Schema version:     {SCHEMA_VERSION}")

    # 7. Config validation check
    click.echo("")
    try:
        from stresslab.stresslab_types import SimulationConfig
        cfg = generate_default_scenario()
        _ = cfg.run_id()
        _ok("Default scenario generation")
    except Exception as e:
        _fail(f"Default scenario generation: {e}")

    # 8. Default config file
    default_cfg = Path("configs/default_scenario.yaml")
    if default_cfg.exists():
        _ok(f"Default config file: {default_cfg}")
    else:
        _warn(f"Default config file not found: {default_cfg}")

    # 9. Determinism hints
    click.echo("")
    blas_threads = os.environ.get("OPENBLAS_NUM_THREADS", "unset")
    mkl_threads = os.environ.get("MKL_NUM_THREADS", "unset")
    click.echo(f"  OPENBLAS_NUM_THREADS: {blas_threads}")
    click.echo(f"  MKL_NUM_THREADS:      {mkl_threads}")
    if blas_threads == "unset" or mkl_threads == "unset":
        _warn("Set OPENBLAS_NUM_THREADS=1 and MKL_NUM_THREADS=1 for deterministic results")

    # 10. Write permissions
    try:
        test_dir = Path("outputs")
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / ".doctor_check"
        test_file.write_text("ok")
        test_file.unlink()
        _ok("Write permissions in outputs/")
    except Exception as e:
        _fail(f"Cannot write to outputs/: {e}")

    # 11. Case workspace permissions
    try:
        case_dir = Path("outputs") / "cases"
        case_dir.mkdir(parents=True, exist_ok=True)
        case_test = case_dir / ".doctor_case_check"
        case_test.write_text("ok")
        case_test.unlink()
        _ok("Write permissions in outputs/cases/")
    except Exception as e:
        _fail(f"Cannot write to outputs/cases/: {e}")

    # Summary
    click.echo("")
    if all_ok and warnings == 0:
        msg = "All checks passed."
        if console:
            console.print(f"[bold green]{msg}[/bold green]")
        else:
            click.echo(msg)
    elif all_ok:
        msg = f"All checks passed with {warnings} warning(s)."
        if console:
            console.print(f"[bold yellow]{msg}[/bold yellow]")
        else:
            click.echo(msg)
    else:
        msg = "Some checks failed. See above for details."
        if console:
            console.print(f"[bold red]{msg}[/bold red]")
        else:
            click.echo(msg)
        sys.exit(1)


# ---------------------------------------------------------------------------
# synth-case command
# ---------------------------------------------------------------------------

@main.command("synth-case")
@click.option("--mode", type=click.Choice(["A", "B"], case_sensitive=False), default="A")
@click.option("--seed", default=42, type=int)
@click.option("--case-id", required=True, type=str)
@click.option("--target-miss-m", default=200.0, type=float)
@click.option("--target-tca-hours", default=4.0, type=float)
@click.option("--primary-pos-sigma-m", default=20.0, type=float)
@click.option("--secondary-pos-sigma-m", default=100.0, type=float)
@click.option("--primary-vel-sigma-mps", default=0.02, type=float)
@click.option("--secondary-vel-sigma-mps", default=0.10, type=float)
@click.option("--hbr-m", default=5.0, type=float)
@click.option("--horizon-seconds", default=172800.0, type=float)
@click.option("--output", required=True, type=click.Path())
def synth_case(
    mode,
    seed,
    case_id,
    target_miss_m,
    target_tca_hours,
    primary_pos_sigma_m,
    secondary_pos_sigma_m,
    primary_vel_sigma_mps,
    secondary_vel_sigma_mps,
    hbr_m,
    horizon_seconds,
    output,
):
    """Generate synthetic conjunction CaseSnapshot for Stress Tester."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = generate_synthetic_case(
        case_id=case_id,
        mode=mode,
        seed=seed,
        target_miss_m=target_miss_m,
        target_tca_hours=target_tca_hours,
        primary_pos_sigma_m=primary_pos_sigma_m,
        secondary_pos_sigma_m=secondary_pos_sigma_m,
        primary_vel_sigma_mps=primary_vel_sigma_mps,
        secondary_vel_sigma_mps=secondary_vel_sigma_mps,
        hard_body_radius_m=hbr_m,
        horizon_seconds=horizon_seconds,
    )

    if out_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            raise click.ClickException("PyYAML not installed. Install with: pip install pyyaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(result["case_snapshot"], f, sort_keys=False)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result["case_snapshot"], f, indent=2)

    click.echo(f"Wrote CaseSnapshot: {out_path}")
    click.echo(json.dumps(result["preview"], indent=2))


# ---------------------------------------------------------------------------
# serve command (local web dashboard)
# ---------------------------------------------------------------------------

@main.command()
@click.option("--workspace", default="outputs", type=click.Path(exists=True),
              help="Root directory containing simulation artifacts")
@click.option("--port", default=5179, type=int, help="Port to listen on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--open", "open_browser", is_flag=True, help="Open browser on start")
@click.option("--dev", is_flag=True, help="Enable CORS for Vite dev server")
@click.option("--enable-launch", is_flag=True, help="Allow API-triggered baseline/stress run launches")
def serve(workspace, port, host, open_browser, dev, enable_launch):
    """Start the local web dashboard for browsing simulation results.

    Launches a FastAPI server that serves the StressLAB dashboard UI
    and provides a REST API for browsing runs, sweeps, and MC batches.

    \b
    Examples:
      stresslab serve                       # Browse outputs/ on port 5179
      stresslab serve --workspace ./my_runs # Browse a custom directory
      stresslab serve --open                # Auto-open browser
      stresslab serve --dev                 # Enable CORS for Vite dev server
    """
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "Web dependencies not installed. Run: pip install stresslab[web]"
        )

    from stresslab.server.app import create_app

    workspace_path = Path(workspace).resolve()
    app = create_app(workspace=workspace_path, dev_mode=dev, enable_launch=enable_launch)

    if open_browser:
        import webbrowser
        import threading

        def _open():
            import time
            time.sleep(1.0)  # Give server a moment to start
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=_open, daemon=True).start()

    click.echo(f"StressLAB Dashboard")
    click.echo(f"  Version:   {__version__}")
    click.echo(f"  Workspace: {workspace_path}")
    click.echo(f"  URL:       http://{host}:{port}")
    click.echo(f"  Launch API:{' enabled' if enable_launch else ' disabled'}")
    if dev:
        click.echo(f"  Dev mode:  CORS enabled for localhost:5173/5174")
    click.echo("")

    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON serialization fallback for numpy types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    main()
