"""Monte Carlo Batch Runner - statistical analysis across multiple runs.

Executes N simulation runs with randomized parameters (initial conditions,
outage patterns, process noise scales) and aggregates results for
comparison between threshold-v1 and integrity-v1 decision models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from stresslab.types import SimulationConfig, MeasurementConfig
from stresslab.modules.simulation_runner import run_simulation


def _perturb_initial_conditions(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> SimulationConfig:
    """Perturb initial conditions by sampling from initial covariance."""
    import copy
    cfg = copy.deepcopy(config)

    # Sample perturbation from initial covariance
    try:
        L1 = np.linalg.cholesky(cfg.cov_obj1)
        cfg.state_obj1 = cfg.state_obj1 + L1 @ rng.standard_normal(6)
    except np.linalg.LinAlgError:
        pass  # Skip if covariance not PD

    try:
        L2 = np.linalg.cholesky(cfg.cov_obj2)
        cfg.state_obj2 = cfg.state_obj2 + L2 @ rng.standard_normal(6)
    except np.linalg.LinAlgError:
        pass

    return cfg


def _randomize_outages(
    config: SimulationConfig,
    rng: np.random.Generator,
    max_outages: int = 3,
    max_duration: float = 43200.0,
) -> SimulationConfig:
    """Generate random outage windows."""
    import copy
    cfg = copy.deepcopy(config)

    n_outages = rng.integers(0, max_outages + 1)
    windows = []
    for _ in range(n_outages):
        start = rng.uniform(cfg.t_start, cfg.t_end * 0.8)
        duration = rng.uniform(3600.0, max_duration)
        windows.append({"start": float(start), "end": float(start + duration)})

    cfg.measurement.outage_windows = windows
    return cfg


def _randomize_process_noise(
    config: SimulationConfig,
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.1, 10.0),
) -> SimulationConfig:
    """Randomize process noise scale factor."""
    import copy
    cfg = copy.deepcopy(config)
    cfg.process_noise.scale = float(rng.uniform(*scale_range))
    return cfg


def run_monte_carlo(
    base_config: SimulationConfig,
    n_runs: int = 100,
    randomize_ic: bool = True,
    randomize_outages: bool = True,
    randomize_pn: bool = True,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
    progress: bool = False,
) -> pd.DataFrame:
    """Run Monte Carlo batch analysis.

    Args:
        base_config: base scenario configuration
        n_runs: number of Monte Carlo runs
        randomize_ic: perturb initial conditions
        randomize_outages: randomize outage windows
        randomize_pn: randomize process noise scale
        output_dir: directory for aggregate outputs
        verbose: print progress
        progress: show tqdm progress bar

    Returns:
        DataFrame with one row per run containing summary metrics.
    """
    rng = np.random.default_rng(base_config.seed)
    results = []

    # Choose progress display: Rich > tqdm > plain
    loop_iter = range(n_runs)
    rich_progress = None
    rich_task_id = None
    if progress:
        try:
            from rich.progress import (
                Progress, SpinnerColumn, BarColumn,
                TextColumn, TimeRemainingColumn, MofNCompleteColumn,
            )
            rich_progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[status]}"),
            )
            rich_progress.start()
            rich_task_id = rich_progress.add_task(
                "Monte Carlo", total=n_runs, status="starting..."
            )
        except ImportError:
            try:
                from tqdm import tqdm
                loop_iter = tqdm(loop_iter, desc="Monte Carlo", unit="run", leave=True)
            except ImportError:
                pass  # No progress display available

    # Interval for writing partial stats
    partial_interval = max(1, n_runs // 10)  # every ~10% of runs

    for i in loop_iter:
        if verbose:
            print(f"\n[Monte Carlo] Run {i+1}/{n_runs}")

        # Create per-run seed
        run_seed = int(rng.integers(0, 2**31))

        import copy
        cfg = copy.deepcopy(base_config)
        cfg.seed = run_seed

        if randomize_ic:
            cfg = _perturb_initial_conditions(cfg, rng)
        if randomize_outages:
            cfg = _randomize_outages(cfg, rng)
        if randomize_pn:
            cfg = _randomize_process_noise(cfg, rng)

        try:
            run_result = run_simulation(cfg, verbose=False)
            logger = run_result["logger"]
            df = logger.to_dataframe()

            # Compute per-run summary
            total_outage_time = sum(
                w["end"] - w["start"] for w in cfg.measurement.outage_windows
            )

            row = {
                "run_index": i,
                "seed": run_seed,
                "process_noise_scale": cfg.process_noise.scale,
                "total_outage_duration": total_outage_time,
                "n_outage_windows": len(cfg.measurement.outage_windows),
                "threshold_v1_trigger_time": run_result["threshold_v1_trigger_time"],
                "integrity_v1_trigger_time": run_result["integrity_v1_trigger_time"],
                "max_pc_degraded": float(df["pc_degraded"].max()),
                "max_cov_trace": float(df["cov_trace_obj1"].max()),
                "max_staleness": float(df["staleness_obj1"].max()),
                "max_miss_distance": float(df["miss_distance"].max()),
                "min_miss_distance": float(df["miss_distance"].min()),
            }

            # Decision compression window
            bt = run_result["threshold_v1_trigger_time"]
            st = run_result["integrity_v1_trigger_time"]
            row["decision_compression_window"] = (
                bt - st if bt is not None and st is not None else None
            )

            results.append(row)

        except Exception as e:
            if verbose:
                print(f"  [WARN] Run {i+1} failed: {e}")
            results.append({
                "run_index": i,
                "seed": run_seed,
                "error": str(e),
            })

        # Update Rich progress if active
        if rich_progress is not None and rich_task_id is not None:
            status = f"seed={run_seed}"
            rich_progress.update(rich_task_id, advance=1, status=status)

        # Write partial aggregate stats periodically
        if output_dir is not None and len(results) > 0 and (i + 1) % partial_interval == 0:
            partial_df = pd.DataFrame(results)
            partial_stats = _compute_aggregate_stats(partial_df)
            partial_stats["runs_completed"] = len(results)
            partial_stats["runs_total"] = n_runs
            partial_path = output_dir / "monte_carlo_partial.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(partial_path, "w") as f:
                json.dump(partial_stats, f, indent=2)

    # Stop Rich progress
    if rich_progress is not None:
        rich_progress.stop()

    mc_df = pd.DataFrame(results)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "monte_carlo_summary.csv"
        mc_df.to_csv(csv_path, index=False)
        if verbose:
            print(f"\n[Monte Carlo] Summary written to {csv_path}")

        # Write aggregate statistics
        stats = _compute_aggregate_stats(mc_df)
        stats_path = output_dir / "monte_carlo_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        if verbose:
            print(f"[Monte Carlo] Stats written to {stats_path}")

    return mc_df


def _compute_aggregate_stats(df: pd.DataFrame) -> dict:
    """Compute aggregate statistics from Monte Carlo results."""
    # Filter successful runs
    if "error" in df.columns:
        good = df[df["error"].isna()]
    else:
        good = df

    stats: dict = {
        "total_runs": len(df),
        "successful_runs": len(good),
    }

    if len(good) == 0:
        return stats

    # Trigger timing
    tv1_triggers = good["threshold_v1_trigger_time"].dropna()
    iv1_triggers = good["integrity_v1_trigger_time"].dropna()

    stats["threshold_v1_trigger"] = {
        "triggered_count": int(len(tv1_triggers)),
        "mean": float(tv1_triggers.mean()) if len(tv1_triggers) > 0 else None,
        "std": float(tv1_triggers.std()) if len(tv1_triggers) > 0 else None,
    }
    stats["integrity_v1_trigger"] = {
        "triggered_count": int(len(iv1_triggers)),
        "mean": float(iv1_triggers.mean()) if len(iv1_triggers) > 0 else None,
        "std": float(iv1_triggers.std()) if len(iv1_triggers) > 0 else None,
    }

    # DCW
    dcw = good["decision_compression_window"].dropna()
    stats["decision_compression_window"] = {
        "mean": float(dcw.mean()) if len(dcw) > 0 else None,
        "std": float(dcw.std()) if len(dcw) > 0 else None,
        "median": float(dcw.median()) if len(dcw) > 0 else None,
    }

    # Max Pc
    stats["max_pc_degraded"] = {
        "mean": float(good["max_pc_degraded"].mean()),
        "std": float(good["max_pc_degraded"].std()),
        "max": float(good["max_pc_degraded"].max()),
    }

    return stats
