"""CLI entry point for StressLAB."""

from __future__ import annotations

from pathlib import Path

import click

from stresslab.modules.scenario_generator import generate_default_scenario, export_scenario
from stresslab.modules.simulation_runner import run_simulation
from stresslab.modules.monte_carlo import run_monte_carlo
from stresslab.types import DynamicsModel


@click.group()
@click.version_option(version="1.0.0")
def main():
    """StressLAB - Conjunction Decision Stress-Testing Platform."""
    pass


@main.command()
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--miss-distance", default=0.5, type=float, help="Target miss distance (km)")
@click.option("--t-end", default=259200.0, type=float, help="Simulation end time (s)")
@click.option("--dt", default=60.0, type=float, help="Timestep (s)")
@click.option("--dynamics", default="two_body_plus_J2",
              type=click.Choice(["two_body", "two_body_plus_J2"]))
@click.option("--output-dir", default="outputs", type=click.Path())
@click.option("--verbose", is_flag=True)
def run(seed, miss_distance, t_end, dt, dynamics, output_dir, verbose):
    """Run a single conjunction stress-test simulation."""
    output = Path(output_dir)

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

    result = run_simulation(config, output_dir=output, verbose=verbose)

    click.echo(f"\nRun ID: {result['run_id']}")
    click.echo(f"Threshold-v1 trigger: {result['threshold_v1_trigger_time']}")
    click.echo(f"Integrity-v1 trigger: {result['integrity_v1_trigger_time']}")
    if result.get("timeseries_path"):
        click.echo(f"Time series:          {result['timeseries_path']}")
    if result.get("summary_path"):
        click.echo(f"Summary:              {result['summary_path']}")


@main.command()
@click.option("--seed", default=42, type=int, help="Base random seed")
@click.option("--n-runs", default=100, type=int, help="Number of Monte Carlo runs")
@click.option("--miss-distance", default=0.5, type=float, help="Target miss distance (km)")
@click.option("--t-end", default=86400.0, type=float, help="Simulation end time (s)")
@click.option("--dt", default=120.0, type=float, help="Timestep (s)")
@click.option("--output-dir", default="outputs/monte_carlo", type=click.Path())
@click.option("--verbose", is_flag=True)
def monte_carlo(seed, n_runs, miss_distance, t_end, dt, output_dir, verbose):
    """Run Monte Carlo batch analysis."""
    output = Path(output_dir)

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
    import json as _json

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


if __name__ == "__main__":
    main()
