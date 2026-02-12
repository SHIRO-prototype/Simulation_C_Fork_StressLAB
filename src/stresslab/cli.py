"""CLI entry point for StressLAB."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from stresslab import __version__
from stresslab.modules.scenario_generator import generate_default_scenario, export_scenario
from stresslab.modules.simulation_runner import run_simulation
from stresslab.modules.monte_carlo import run_monte_carlo
from stresslab.types import DynamicsModel


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

def _load_config_yaml(path: Path):
    """Load a YAML scenario file and return a SimulationConfig."""
    import yaml
    import numpy as np
    from stresslab.types import (
        SimulationConfig, ProcessNoiseConfig, MeasurementConfig,
        ManeuverConfig, ThresholdV1Config, IntegrityV1Config,
    )

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Build sub-configs from YAML sections
    pn_raw = raw.get("process_noise", {})
    pn = ProcessNoiseConfig(
        sigma_radial=pn_raw.get("sigma_radial", 1e-9),
        sigma_tangential=pn_raw.get("sigma_tangential", 1e-9),
        sigma_normal=pn_raw.get("sigma_normal", 1e-9),
        scale=pn_raw.get("scale", 1.0),
    )

    meas_raw = raw.get("measurement", {})
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
    tv1 = ThresholdV1Config(
        pc_threshold=tv1_raw.get("pc_threshold", 1e-4),
        miss_distance_threshold=tv1_raw.get("miss_distance_threshold"),
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

    tl = raw.get("timeline", {})
    risk = raw.get("risk", {})

    # Generate the scenario geometry (orbital states), then overlay YAML params
    config = generate_default_scenario(
        seed=raw.get("seed", 42),
        miss_distance_km=raw.get("miss_distance_km", 0.5),
        t_end=tl.get("t_end", 259200.0),
        dt=tl.get("dt", 60.0),
        dynamics=DynamicsModel(raw.get("dynamics_model", "two_body_plus_J2")),
    )

    # Overlay sub-configs
    config.process_noise = pn
    config.measurement = meas
    config.maneuver = maneuver
    config.threshold_v1 = tv1
    config.integrity_v1 = iv1
    config.t_start = tl.get("t_start", 0.0)
    config.combined_hard_body_radius = risk.get("combined_hard_body_radius", 0.02)

    return config


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
def run(config_path, seed, miss_distance, t_end, dt, dynamics, output_dir, verbose, progress):
    """Run a single conjunction stress-test simulation."""
    output = Path(output_dir)

    if config_path:
        config = _load_config_yaml(Path(config_path))
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

    result = run_simulation(config, output_dir=output, verbose=verbose, progress=progress)

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
        base_config = _load_config_yaml(Path(config_path))
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
# report command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--input", "input_path", required=True,
              type=click.Path(exists=True), help="Path to run summary JSON")
@click.option("--format", "fmt", default="json",
              type=click.Choice(["json", "csv", "latex"]),
              help="Output format")
@click.option("--output-dir", default="outputs/reports",
              type=click.Path(), help="Output directory")
def report(input_path, fmt, output_dir):
    """Generate a formatted report from a run summary."""
    from stresslab.reporting.compare import load_summary_json
    from stresslab.reporting.benchmark_report import generate_benchmark_report
    from stresslab.reporting.exporters import (
        export_json, export_csv, export_latex_table, report_to_latex_rows,
    )
    from stresslab.metrics_contract import MetricsSummary

    output = Path(output_dir)

    raw = load_summary_json(Path(input_path))

    # Try to reconstruct a MetricsSummary if all fields are present
    try:
        summary = MetricsSummary(**{
            k: raw[k] for k in MetricsSummary.__dataclass_fields__
        })
        bench_report = generate_benchmark_report(summary)
    except (KeyError, TypeError):
        # Fall back to treating the raw dict as the report
        bench_report = raw

    if fmt == "json":
        out = export_json(bench_report, output / "report.json")
    elif fmt == "csv":
        out = export_csv(bench_report, output / "report.csv")
    elif fmt == "latex":
        rows = report_to_latex_rows(bench_report)
        out = export_latex_table(rows, output / "report.tex")
    else:
        click.echo(f"Unknown format: {fmt}", err=True)
        return

    click.echo(f"Report written to: {out}")


# ---------------------------------------------------------------------------
# compare command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--run-a", required=True, type=click.Path(exists=True),
              help="Path to first run summary JSON")
@click.option("--run-b", required=True, type=click.Path(exists=True),
              help="Path to second run summary JSON")
@click.option("--format", "fmt", default="json",
              type=click.Choice(["json", "csv"]),
              help="Output format")
@click.option("--output-dir", default="outputs/reports",
              type=click.Path(), help="Output directory")
def compare(run_a, run_b, fmt, output_dir):
    """Compare two simulation run summaries."""
    from stresslab.reporting.compare import (
        compare_from_files, comparison_to_dataframe,
    )
    from stresslab.reporting.exporters import export_json, export_csv

    output = Path(output_dir)
    comparison = compare_from_files(Path(run_a), Path(run_b))

    if fmt == "json":
        out = export_json(comparison, output / "comparison.json")
    elif fmt == "csv":
        df = comparison_to_dataframe(comparison)
        out = export_csv(df, output / "comparison.csv")
    else:
        click.echo(f"Unknown format: {fmt}", err=True)
        return

    click.echo(f"Comparison written to: {out}")


# ---------------------------------------------------------------------------
# doctor command
# ---------------------------------------------------------------------------

@main.command()
def doctor():
    """Check the StressLAB installation and environment."""
    all_ok = True

    click.echo("StressLAB Doctor")
    click.echo("=" * 40)

    # 1. Version
    click.echo(f"  stresslab version:  {__version__}")

    # 2. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    click.echo(f"  Python version:     {py_ver}")
    if sys.version_info < (3, 10):
        click.echo("  [FAIL] Python >= 3.10 required")
        all_ok = False
    else:
        click.echo("  [OK]   Python >= 3.10")

    # 3. Core dependencies
    deps = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("pyarrow", "pyarrow"),
        ("yaml", "pyyaml"),
        ("click", "click"),
    ]
    for import_name, pkg_name in deps:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            click.echo(f"  [OK]   {pkg_name} ({ver})")
        except ImportError:
            click.echo(f"  [FAIL] {pkg_name} not found")
            all_ok = False

    # 4. Optional dependencies
    opt_deps = [
        ("tqdm", "tqdm", "progress bars"),
        ("matplotlib", "matplotlib", "plotting"),
    ]
    for import_name, pkg_name, purpose in opt_deps:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            click.echo(f"  [OK]   {pkg_name} ({ver}) - {purpose}")
        except ImportError:
            click.echo(f"  [WARN] {pkg_name} not installed - {purpose} unavailable")

    # 5. Metrics contract version
    from stresslab.metrics_contract import METRICS_CONTRACT_VERSION
    click.echo(f"  Contract version:   {METRICS_CONTRACT_VERSION}")

    # 6. Schema version
    from stresslab.modules.logging_engine import SCHEMA_VERSION
    click.echo(f"  Schema version:     {SCHEMA_VERSION}")

    # 7. Config validation check
    click.echo("")
    try:
        from stresslab.types import SimulationConfig
        cfg = generate_default_scenario()
        _ = cfg.run_id()
        click.echo("  [OK]   Default scenario generation")
    except Exception as e:
        click.echo(f"  [FAIL] Default scenario generation: {e}")
        all_ok = False

    # 8. Default config file
    default_cfg = Path("configs/default_scenario.yaml")
    if default_cfg.exists():
        click.echo(f"  [OK]   Default config file: {default_cfg}")
    else:
        click.echo(f"  [WARN] Default config file not found: {default_cfg}")

    click.echo("")
    if all_ok:
        click.echo("All checks passed.")
    else:
        click.echo("Some checks failed. See above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
