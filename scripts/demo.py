#!/usr/bin/env python3
"""StressLAB Demo Script

Runs the canonical demo workflow:
  1. Single run with 12-hour outage (flagship scenario)
  2. Single run with no outage (baseline)
  3. Compare the two runs
  4. Outage duration sweep (0h to 12h)
  5. Generate reports + flagship plots for sweep

Outputs are written to outputs/demo/. This script exercises the full
CLI instrumentation pipeline and produces publication-ready figures.

Usage:
    python scripts/demo.py
    python scripts/demo.py --quick     # Faster: shorter sim, fewer sweep points
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(args: list[str], label: str) -> None:
    """Run a CLI command and print status."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(args)}\n")
    result = subprocess.run(args, capture_output=False)
    if result.returncode != 0:
        print(f"\n  [FAILED] {label} (exit code {result.returncode})")
        sys.exit(1)
    print(f"  [OK] {label}")


def main():
    parser = argparse.ArgumentParser(description="StressLAB Demo")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: shorter simulations, fewer sweep points")
    args = parser.parse_args()

    demo_dir = Path("outputs/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        t_end = "3600"     # 1 hour
        dt = "120"         # 2 minutes
        sweep_values = "0h,1h"
    else:
        t_end = "86400"    # 1 day
        dt = "60"          # 1 minute
        sweep_values = "0h,1h,3h,6h,12h"

    # ---- 1. Single run: with outage ----
    run_cmd([
        sys.executable, "-m", "stresslab.cli", "run",
        "--config", "configs/demos/leo_3day_with_outage.yaml",
        "--output-dir", str(demo_dir / "run_with_outage"),
        "--progress",
    ], "Single Run: LEO 3-day with 12h outage")

    # ---- 2. Single run: no outage ----
    run_cmd([
        sys.executable, "-m", "stresslab.cli", "run",
        "--config", "configs/demos/leo_1day_no_outage.yaml",
        "--output-dir", str(demo_dir / "run_no_outage"),
        "--progress",
    ], "Single Run: LEO 1-day no outage")

    # ---- 3. Compare the two runs ----
    run_cmd([
        sys.executable, "-m", "stresslab.cli", "compare",
        str(demo_dir / "run_with_outage"),
        str(demo_dir / "run_no_outage"),
        "--out", str(demo_dir / "comparison"),
    ], "Compare: outage vs no-outage")

    # ---- 4. Outage duration sweep ----
    run_cmd([
        sys.executable, "-m", "stresslab.cli", "sweep",
        "outage.duration",
        "--values", sweep_values,
        "--seed", "42",
        "--t-end", t_end,
        "--dt", dt,
        "--output-dir", str(demo_dir / "sweep_outage_duration"),
        "--progress",
    ], f"Sweep: outage.duration = [{sweep_values}]")

    # ---- 5. Generate reports + plots for sweep ----
    run_cmd([
        sys.executable, "-m", "stresslab.cli", "report",
        str(demo_dir / "sweep_outage_duration"),
        "--out", str(demo_dir / "sweep_report"),
        "--formats", "png,json,csv",
    ], "Report: sweep with flagship plots")

    # ---- 6. Generate report for single outage run ----
    run_out = demo_dir / "run_with_outage"
    summary_files = list(run_out.glob("summary_*.json"))
    if summary_files:
        run_cmd([
            sys.executable, "-m", "stresslab.cli", "report",
            str(summary_files[0]),
            "--out", str(demo_dir / "single_run_report"),
            "--formats", "png,json,csv",
        ], "Report: single run with posture timeline + Pc drift plots")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  DEMO COMPLETE")
    print(f"{'='*60}")
    print(f"  All outputs in: {demo_dir.resolve()}")
    print(f"\n  Key artifacts:")
    print(f"    - Run comparison:      {demo_dir / 'comparison'}")
    print(f"    - Sweep summary:       {demo_dir / 'sweep_outage_duration'}")
    print(f"    - Sweep report/plots:  {demo_dir / 'sweep_report'}")
    print(f"    - Single run report:   {demo_dir / 'single_run_report'}")
    print()


if __name__ == "__main__":
    main()
