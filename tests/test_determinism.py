"""Determinism guard test - ensures identical config+seed produces identical output.

This end-to-end test runs the full simulation pipeline twice with the same
configuration and seed, then asserts that all time-series values and summary
metrics are bitwise identical. It catches:
  - Random state leaks (unseeded RNG usage)
  - Floating-point ordering issues (non-deterministic reductions)
  - Uninitialized memory reads
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stresslab.modules.scenario_generator import generate_default_scenario
from stresslab.modules.simulation_runner import run_simulation


def _make_short_config(seed: int = 42):
    """Create a config with short time window for fast e2e test."""
    config = generate_default_scenario(seed=seed)
    # Run for just 1 hour with 60s steps = 60 timesteps (fast)
    config.t_end = config.t_start + 3600.0
    config.dt = 60.0
    # Add an outage window to exercise the degraded path
    config.measurement.outage_windows = [
        {"start": config.t_start + 600.0, "end": config.t_start + 1800.0}
    ]
    return config


class TestDeterminismGuard:
    """Run simulation twice, assert bitwise identical outputs."""

    def test_identical_timeseries(self, tmp_path: Path):
        """Two runs with same config+seed produce identical time series."""
        config = _make_short_config(seed=42)

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"

        result1 = run_simulation(config, output_dir=dir1)
        result2 = run_simulation(config, output_dir=dir2)

        df1 = result1["logger"].to_dataframe()
        df2 = result2["logger"].to_dataframe()

        # Same shape
        assert df1.shape == df2.shape

        # Bitwise identical values for every column
        for col in df1.columns:
            np.testing.assert_array_equal(
                df1[col].values, df2[col].values,
                err_msg=f"Column '{col}' differs between runs",
            )

    def test_identical_run_id(self):
        """Same config produces same run_id."""
        config1 = _make_short_config(seed=42)
        config2 = _make_short_config(seed=42)
        assert config1.run_id() == config2.run_id()

    def test_identical_summary(self, tmp_path: Path):
        """Two runs produce identical JSON summaries (key metric values)."""
        config = _make_short_config(seed=42)

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"

        result1 = run_simulation(config, output_dir=dir1)
        result2 = run_simulation(config, output_dir=dir2)

        # Read summaries
        sum_path1 = result1["summary_path"]
        sum_path2 = result2["summary_path"]

        with open(sum_path1) as f:
            s1 = json.load(f)
        with open(sum_path2) as f:
            s2 = json.load(f)

        # Every key must match exactly
        assert s1.keys() == s2.keys()
        for key in s1:
            assert s1[key] == s2[key], f"Summary key '{key}' differs: {s1[key]} vs {s2[key]}"

    def test_identical_trigger_times(self, tmp_path: Path):
        """Decision trigger times are identical across runs."""
        config = _make_short_config(seed=42)

        result1 = run_simulation(config)
        result2 = run_simulation(config)

        assert result1["threshold_v1_trigger_time"] == result2["threshold_v1_trigger_time"]
        assert result1["integrity_v1_trigger_time"] == result2["integrity_v1_trigger_time"]

    def test_different_seed_different_output(self):
        """Sanity check: different seeds produce different run_ids.

        Note: The scenario generator currently produces the same flyby
        geometry regardless of seed (seed only affects the config hash).
        This test verifies that at minimum the run_ids are distinct,
        confirming the hashing lock differentiates configs by seed.
        """
        config1 = _make_short_config(seed=42)
        config2 = _make_short_config(seed=99)

        assert config1.run_id() != config2.run_id()

    def test_parquet_files_identical(self, tmp_path: Path):
        """Parquet data (excluding metadata timestamps) is identical."""
        config = _make_short_config(seed=42)

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"

        result1 = run_simulation(config, output_dir=dir1)
        result2 = run_simulation(config, output_dir=dir2)

        df1 = pd.read_parquet(result1["timeseries_path"])
        df2 = pd.read_parquet(result2["timeseries_path"])

        pd.testing.assert_frame_equal(df1, df2)
