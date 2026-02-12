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
    click.echo(f"Baseline trigger: {result['baseline_trigger_time']}")
    click.echo(f"SHIRO trigger:    {result['shiro_trigger_time']}")
    if result.get("timeseries_path"):
        click.echo(f"Time series:      {result['timeseries_path']}")
    if result.get("summary_path"):
        click.echo(f"Summary:          {result['summary_path']}")


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
    good = df.dropna(subset=["baseline_trigger_time", "shiro_trigger_time"], how="all")
    if "decision_compression_window" in good.columns:
        dcw = good["decision_compression_window"].dropna()
        if len(dcw) > 0:
            click.echo(f"\nDecision Compression Window:")
            click.echo(f"  Mean:   {dcw.mean():.1f}s")
            click.echo(f"  Std:    {dcw.std():.1f}s")
            click.echo(f"  Median: {dcw.median():.1f}s")


if __name__ == "__main__":
    main()
