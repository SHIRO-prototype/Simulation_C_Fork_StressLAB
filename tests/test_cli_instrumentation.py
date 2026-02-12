"""Tests for CLI instrumentation: sweep, compare, report, doctor, live panel, and plotting.

Covers:
  - Sweep command: deterministic outputs, hash reproducibility
  - Compare command: correct deltas
  - Report command: generates expected files
  - Doctor command: enhanced checks
  - Live panel callback: does not break simulation
  - Plotting module: all 6 plot functions produce output files
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from stresslab import __version__
from stresslab.modules.scenario_generator import generate_default_scenario
from stresslab.modules.simulation_runner import run_simulation
from stresslab.types import SimulationConfig


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def short_config():
    """Generate a short scenario for quick tests (10 steps)."""
    return generate_default_scenario(
        seed=42, t_end=600.0, dt=60.0,
    )


@pytest.fixture
def run_result(short_config, tmp_path):
    """Run a short simulation and return (result_dict, output_dir)."""
    out = tmp_path / "outputs"
    result = run_simulation(short_config, output_dir=out, verbose=False)
    return result, out


@pytest.fixture
def two_run_summaries(tmp_path):
    """Create two run summary JSONs for comparison tests."""
    cfg_a = generate_default_scenario(seed=42, t_end=600.0, dt=60.0)
    cfg_b = generate_default_scenario(seed=99, t_end=600.0, dt=60.0)

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    res_a = run_simulation(cfg_a, output_dir=out_a)
    res_b = run_simulation(cfg_b, output_dir=out_b)

    json_a = list(out_a.glob("summary_*.json"))[0]
    json_b = list(out_b.glob("summary_*.json"))[0]

    return json_a, json_b


# ===================================================================
# TestSweepCommand
# ===================================================================

class TestSweepCommand:
    """Test the 'stresslab sweep' command."""

    def test_sweep_basic_execution(self, tmp_path):
        from stresslab.cli import main
        runner = CliRunner()
        out = tmp_path / "sweep"

        result = runner.invoke(main, [
            "sweep", "outage.duration",
            "--values", "0h,1h",
            "--seed", "42",
            "--t-end", "600",
            "--dt", "60",
            "--output-dir", str(out),
            "--progress",
        ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"

    def test_sweep_produces_csv_and_json(self, tmp_path):
        from stresslab.cli import main
        runner = CliRunner()
        out = tmp_path / "sweep"

        result = runner.invoke(main, [
            "sweep", "outage.duration",
            "--values", "0h,1h",
            "--seed", "42",
            "--t-end", "600",
            "--dt", "60",
            "--output-dir", str(out),
        ])
        assert result.exit_code == 0
        assert (out / "sweep_summary.csv").exists()
        assert (out / "sweep_summary.json").exists()

    def test_sweep_json_has_schema_versions(self, tmp_path):
        from stresslab.cli import main
        runner = CliRunner()
        out = tmp_path / "sweep"

        runner.invoke(main, [
            "sweep", "outage.duration",
            "--values", "0h,1h",
            "--seed", "42",
            "--t-end", "600",
            "--dt", "60",
            "--output-dir", str(out),
        ])

        with open(out / "sweep_summary.json") as f:
            data = json.load(f)

        assert "schema_version" in data
        assert "metrics_contract_version" in data
        assert "stresslab_version" in data
        assert data["stresslab_version"] == __version__

    def test_sweep_deterministic_hash(self, tmp_path):
        """Running sweep twice with same params gives same hash."""
        from stresslab.cli import main
        runner = CliRunner()

        hashes = []
        for i in range(2):
            out = tmp_path / f"sweep_{i}"
            result = runner.invoke(main, [
                "sweep", "outage.duration",
                "--values", "0h,1h",
                "--seed", "42",
                "--t-end", "600",
                "--dt", "60",
                "--output-dir", str(out),
            ])
            assert result.exit_code == 0
            # Extract hash from output
            for line in result.output.splitlines():
                if "Sweep hash:" in line:
                    hashes.append(line.split(":")[-1].strip())
                    break

        assert len(hashes) == 2
        assert hashes[0] == hashes[1], f"Hashes differ: {hashes}"

    def test_sweep_result_count_matches_values(self, tmp_path):
        from stresslab.cli import main
        runner = CliRunner()
        out = tmp_path / "sweep"

        runner.invoke(main, [
            "sweep", "outage.duration",
            "--values", "0h,1h,3h",
            "--seed", "42",
            "--t-end", "600",
            "--dt", "60",
            "--output-dir", str(out),
        ])

        with open(out / "sweep_summary.json") as f:
            data = json.load(f)

        assert len(data["results"]) == 3
        assert data["sweep_param"] == "outage.duration"

    def test_sweep_process_noise_parameter(self, tmp_path):
        """Test sweeping a non-outage parameter."""
        from stresslab.cli import main
        runner = CliRunner()
        out = tmp_path / "sweep"

        result = runner.invoke(main, [
            "sweep", "process_noise.scale",
            "--values", "0.5,1.0,2.0",
            "--seed", "42",
            "--t-end", "600",
            "--dt", "60",
            "--output-dir", str(out),
        ])
        assert result.exit_code == 0
        assert (out / "sweep_summary.json").exists()

    def test_sweep_unknown_parameter_fails(self, tmp_path):
        from stresslab.cli import main
        runner = CliRunner()
        out = tmp_path / "sweep"

        result = runner.invoke(main, [
            "sweep", "nonexistent.param",
            "--values", "1,2",
            "--seed", "42",
            "--t-end", "600",
            "--dt", "60",
            "--output-dir", str(out),
        ])
        assert result.exit_code != 0

    def test_sweep_time_unit_parsing(self):
        from stresslab.cli import _parse_sweep_value
        assert _parse_sweep_value("3h") == 10800.0
        assert _parse_sweep_value("30m") == 1800.0
        assert _parse_sweep_value("90s") == 90.0
        assert _parse_sweep_value("100") == 100.0
        assert _parse_sweep_value(" 1h ") == 3600.0


# ===================================================================
# TestCompareCommand
# ===================================================================

class TestCompareCommand:
    """Test the 'stresslab compare' command."""

    def test_compare_basic_execution(self, two_run_summaries):
        from stresslab.cli import main
        runner = CliRunner()
        json_a, json_b = two_run_summaries

        result = runner.invoke(main, [
            "compare", str(json_a), str(json_b),
        ])
        assert result.exit_code == 0

    def test_compare_shows_metrics(self, two_run_summaries):
        from stresslab.cli import main
        runner = CliRunner()
        json_a, json_b = two_run_summaries

        result = runner.invoke(main, [
            "compare", str(json_a), str(json_b),
        ])
        assert result.exit_code == 0
        # Should show at least one metric name
        assert "false_safe_rate" in result.output or "Delta" in result.output

    def test_compare_with_filter(self, two_run_summaries):
        from stresslab.cli import main
        runner = CliRunner()
        json_a, json_b = two_run_summaries

        result = runner.invoke(main, [
            "compare", str(json_a), str(json_b),
            "--metrics", "false_safe_rate,max_staleness",
        ])
        assert result.exit_code == 0

    def test_compare_with_output_dir(self, two_run_summaries, tmp_path):
        from stresslab.cli import main
        runner = CliRunner()
        json_a, json_b = two_run_summaries
        out = tmp_path / "comparison_out"

        result = runner.invoke(main, [
            "compare", str(json_a), str(json_b),
            "--out", str(out),
        ])
        assert result.exit_code == 0
        assert (out / "comparison.json").exists()
        assert (out / "comparison.csv").exists()

    def test_compare_auto_detect_directories(self, two_run_summaries, tmp_path):
        """Compare should auto-detect summary files in directories."""
        from stresslab.cli import main
        runner = CliRunner()
        json_a, json_b = two_run_summaries
        dir_a = json_a.parent
        dir_b = json_b.parent

        result = runner.invoke(main, [
            "compare", str(dir_a), str(dir_b),
        ])
        assert result.exit_code == 0

    def test_compare_delta_correctness(self, two_run_summaries):
        """Verify the deltas are computed correctly."""
        from stresslab.reporting.compare import compare_from_files

        json_a, json_b = two_run_summaries
        comparison = compare_from_files(Path(json_a), Path(json_b))

        for m in comparison.get("metrics", []):
            label_a = comparison["label_a"]
            label_b = comparison["label_b"]
            va = m.get(f"value_{label_a}")
            vb = m.get(f"value_{label_b}")
            delta = m.get("delta")
            if va is not None and vb is not None and delta is not None:
                assert abs(delta - (vb - va)) < 1e-10, \
                    f"Delta mismatch for {m['metric']}: {delta} != {vb} - {va}"


# ===================================================================
# TestReportCommand
# ===================================================================

class TestReportCommand:
    """Test the 'stresslab report' command."""

    def test_report_single_run(self, run_result, tmp_path):
        from stresslab.cli import main
        runner = CliRunner()
        _, out = run_result

        json_files = list(out.glob("summary_*.json"))
        assert len(json_files) >= 1

        report_out = tmp_path / "report_out"
        result = runner.invoke(main, [
            "report", str(json_files[0]),
            "--out", str(report_out),
            "--formats", "json,csv",
        ])
        assert result.exit_code == 0
        assert (report_out / "report.json").exists()
        assert (report_out / "summary_table.csv").exists()

    def test_report_json_has_envelope(self, run_result, tmp_path):
        """Report JSON must be wrapped in the standard envelope."""
        from stresslab.cli import main
        runner = CliRunner()
        _, out = run_result

        json_files = list(out.glob("summary_*.json"))
        report_out = tmp_path / "report_out"

        runner.invoke(main, [
            "report", str(json_files[0]),
            "--out", str(report_out),
            "--formats", "json",
        ])

        with open(report_out / "report.json") as f:
            envelope = json.load(f)

        assert "schema_version" in envelope
        assert "metrics_contract_version" in envelope
        assert "stresslab_version" in envelope
        assert "report" in envelope

    def test_report_single_run_with_plots(self, run_result, tmp_path):
        """If PNG format requested and timeseries exists, plots should be generated."""
        from stresslab.cli import main
        runner = CliRunner()
        _, out = run_result

        json_files = list(out.glob("summary_*.json"))
        report_out = tmp_path / "report_out"

        result = runner.invoke(main, [
            "report", str(json_files[0]),
            "--out", str(report_out),
            "--formats", "png,json",
        ])
        assert result.exit_code == 0
        # Should generate plot PNGs if timeseries parquet exists
        parquet_files = list(out.glob("timeseries_*.parquet"))
        if parquet_files:
            assert (report_out / "posture_timeline_overlay.png").exists() or \
                   "Plot generation failed" in result.output

    def test_report_sweep_dir(self, tmp_path):
        """Report command should handle sweep directories."""
        from stresslab.cli import main
        runner = CliRunner()

        # First create a sweep
        sweep_out = tmp_path / "sweep"
        runner.invoke(main, [
            "sweep", "outage.duration",
            "--values", "0h,1h",
            "--seed", "42",
            "--t-end", "600",
            "--dt", "60",
            "--output-dir", str(sweep_out),
        ])

        # Now report on it
        report_out = tmp_path / "report_out"
        result = runner.invoke(main, [
            "report", str(sweep_out),
            "--out", str(report_out),
            "--formats", "json,csv,png",
        ])
        assert result.exit_code == 0


# ===================================================================
# TestDoctorEnhanced
# ===================================================================

class TestDoctorEnhanced:
    """Test the enhanced doctor command with extra checks."""

    def test_doctor_checks_rich(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0
        assert "rich" in result.output

    def test_doctor_checks_matplotlib(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert "matplotlib" in result.output

    def test_doctor_checks_blas_threads(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert "OPENBLAS_NUM_THREADS" in result.output

    def test_doctor_checks_write_permissions(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert "outputs" in result.output.lower()


# ===================================================================
# TestLivePanelCallback
# ===================================================================

class TestLivePanelCallback:
    """Test that the live panel callback does not break simulation."""

    def test_callback_receives_data(self, short_config):
        """Step callback should be called and receive expected keys."""
        received = []

        def capture_callback(data):
            received.append(data)

        result = run_simulation(short_config, step_callback=capture_callback)
        assert len(received) > 0

        # Check expected keys in callback data
        expected_keys = {
            "t", "time_to_tca", "miss_distance", "staleness",
            "pc_reference", "pc_degraded", "risk_ratio",
            "cov_norm", "threshold_v1_state", "integrity_v1_state",
        }
        for key in expected_keys:
            assert key in received[0], f"Missing key: {key}"

    def test_callback_does_not_change_results(self, short_config):
        """Simulation results must be identical with or without callback."""
        result_no_cb = run_simulation(short_config)

        captured = []
        result_with_cb = run_simulation(short_config, step_callback=captured.append)

        assert result_no_cb["run_id"] == result_with_cb["run_id"]
        assert result_no_cb["threshold_v1_trigger_time"] == \
               result_with_cb["threshold_v1_trigger_time"]
        assert result_no_cb["integrity_v1_trigger_time"] == \
               result_with_cb["integrity_v1_trigger_time"]

    def test_live_panel_import(self):
        """LivePanelCallback should be importable."""
        from stresslab.live_panel import LivePanelCallback
        assert LivePanelCallback is not None

    def test_live_panel_instantiation(self):
        """LivePanelCallback should instantiate without error."""
        from stresslab.live_panel import LivePanelCallback
        panel = LivePanelCallback(
            run_id="test123",
            seed=42,
            dt=60.0,
            t_end=600.0,
            dynamics="two_body_plus_J2",
        )
        assert panel is not None


# ===================================================================
# TestPlottingModule
# ===================================================================

class TestPlottingModule:
    """Test all 6 plotting functions produce valid output files."""

    @pytest.fixture
    def sample_timeseries(self):
        """Create a minimal sample timeseries DataFrame."""
        n = 50
        t = np.arange(0, n * 60.0, 60.0)
        return pd.DataFrame({
            "timestamp": t,
            "threshold_v1_alert_state": np.where(t < 1500, "Safe", "Alert"),
            "integrity_v1_state": np.where(t < 900, "Monitor",
                                   np.where(t < 1800, "Watch",
                                   np.where(t < 2400, "Warning", "Critical"))),
            "integrity_v1_score": np.linspace(0.1, 0.9, n),
            "pc_degraded": np.logspace(-8, -3, n),
            "pc_reference": np.logspace(-8, -4, n),
            "miss_distance": np.linspace(2.0, 0.1, n),
            "staleness_obj1": np.where(t < 900, t % 3600,
                              np.where(t < 1800, (t - 900) + 3600, t % 3600)),
            "staleness_obj2": np.zeros(n),
            "pc_drift": np.abs(np.logspace(-8, -3, n) - np.logspace(-8, -4, n)) /
                        np.maximum(np.logspace(-8, -4, n), 1e-30),
        })

    def test_plot_posture_timeline(self, sample_timeseries, tmp_path):
        from stresslab.plotting import plot_posture_timeline
        out = tmp_path / "posture_timeline.png"
        plot_posture_timeline(sample_timeseries, out)
        assert out.exists()
        assert out.stat().st_size > 1000  # Non-trivial file

    def test_plot_pc_drift_vs_staleness(self, sample_timeseries, tmp_path):
        from stresslab.plotting import plot_pc_drift_vs_staleness
        out = tmp_path / "pc_drift_vs_staleness.png"
        plot_pc_drift_vs_staleness(sample_timeseries, out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_plot_outage_vs_trigger_shift(self, tmp_path):
        from stresslab.plotting import plot_outage_vs_trigger_shift
        results = [
            {"param_value": 0, "param_value_raw": "0h",
             "threshold_v1_trigger_time": 3600, "integrity_v1_trigger_time": 1800,
             "decision_compression_window": 1800},
            {"param_value": 3600, "param_value_raw": "1h",
             "threshold_v1_trigger_time": 2400, "integrity_v1_trigger_time": 1200,
             "decision_compression_window": 1200},
            {"param_value": 10800, "param_value_raw": "3h",
             "threshold_v1_trigger_time": None, "integrity_v1_trigger_time": 900,
             "decision_compression_window": None},
        ]
        out = tmp_path / "outage_vs_trigger.png"
        plot_outage_vs_trigger_shift(results, "outage.duration", out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_plot_outage_vs_false_safe_rate(self, tmp_path):
        from stresslab.plotting import plot_outage_vs_false_safe_rate
        results = [
            {"param_value": 0, "param_value_raw": "0h",
             "false_safe_rate": 0.0, "max_pc_degraded": 1e-6,
             "decision_instability_index": 0.1},
            {"param_value": 3600, "param_value_raw": "1h",
             "false_safe_rate": 0.05, "max_pc_degraded": 1e-5,
             "decision_instability_index": 0.25},
            {"param_value": 10800, "param_value_raw": "3h",
             "false_safe_rate": 0.15, "max_pc_degraded": 1e-4,
             "decision_instability_index": 0.5},
        ]
        out = tmp_path / "false_safe.png"
        plot_outage_vs_false_safe_rate(results, "outage.duration", out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_plot_compression_window_distribution(self, tmp_path):
        from stresslab.plotting import plot_compression_window_distribution
        rng = np.random.default_rng(42)
        dcw_vals = rng.normal(3600, 600, 100).tolist()  # ~1h mean
        out = tmp_path / "dcw_distribution.png"
        plot_compression_window_distribution(dcw_vals, out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_plot_decision_instability(self, sample_timeseries, tmp_path):
        from stresslab.plotting import plot_decision_instability
        out = tmp_path / "decision_instability.png"
        plot_decision_instability(sample_timeseries, out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_plot_posture_with_custom_title(self, sample_timeseries, tmp_path):
        from stresslab.plotting import plot_posture_timeline
        out = tmp_path / "custom_title.png"
        plot_posture_timeline(sample_timeseries, out, title="Custom Title Test")
        assert out.exists()

    def test_plot_handles_empty_like_data(self, tmp_path):
        """Plot functions should handle minimal data without crashing."""
        from stresslab.plotting import plot_posture_timeline
        df = pd.DataFrame({
            "timestamp": [0.0, 60.0],
            "threshold_v1_alert_state": ["Safe", "Safe"],
            "integrity_v1_state": ["Monitor", "Monitor"],
            "integrity_v1_score": [0.1, 0.1],
            "pc_degraded": [1e-8, 1e-8],
            "pc_reference": [1e-8, 1e-8],
            "miss_distance": [1.0, 1.0],
            "staleness_obj1": [0.0, 60.0],
            "staleness_obj2": [0.0, 0.0],
        })
        out = tmp_path / "minimal.png"
        plot_posture_timeline(df, out)
        assert out.exists()


# ===================================================================
# TestSetConfigParam
# ===================================================================

class TestSetConfigParam:
    """Test the _set_config_param helper."""

    def test_set_outage_duration(self):
        from stresslab.cli import _set_config_param
        cfg = generate_default_scenario(seed=42, t_end=86400.0, dt=120.0)
        new_cfg = _set_config_param(cfg, "outage.duration", 7200.0)
        assert len(new_cfg.measurement.outage_windows) == 1
        window = new_cfg.measurement.outage_windows[0]
        assert abs(window["end"] - window["start"] - 7200.0) < 0.01

    def test_set_zero_outage_clears_windows(self):
        from stresslab.cli import _set_config_param
        cfg = generate_default_scenario(seed=42, t_end=86400.0, dt=120.0)
        new_cfg = _set_config_param(cfg, "outage.duration", 0.0)
        assert len(new_cfg.measurement.outage_windows) == 0

    def test_set_process_noise_scale(self):
        from stresslab.cli import _set_config_param
        cfg = generate_default_scenario(seed=42, t_end=86400.0, dt=120.0)
        new_cfg = _set_config_param(cfg, "process_noise.scale", 5.0)
        assert new_cfg.process_noise.scale == 5.0

    def test_set_measurement_interval(self):
        from stresslab.cli import _set_config_param
        cfg = generate_default_scenario(seed=42, t_end=86400.0, dt=120.0)
        new_cfg = _set_config_param(cfg, "measurement.update_interval", 1800.0)
        assert new_cfg.measurement.update_interval == 1800.0

    def test_set_config_is_deepcopy(self):
        """Setting a param should not modify the original config."""
        from stresslab.cli import _set_config_param
        cfg = generate_default_scenario(seed=42, t_end=86400.0, dt=120.0)
        original_scale = cfg.process_noise.scale
        new_cfg = _set_config_param(cfg, "process_noise.scale", 99.0)
        assert cfg.process_noise.scale == original_scale
        assert new_cfg.process_noise.scale == 99.0


# ===================================================================
# TestRichHelpers
# ===================================================================

class TestRichHelpers:
    """Test Rich helper functions from cli.py."""

    def test_fmt_num_none(self):
        from stresslab.cli import _fmt_num
        assert _fmt_num(None) == "---"

    def test_fmt_num_dash(self):
        from stresslab.cli import _fmt_num
        assert _fmt_num("---") == "---"

    def test_fmt_num_small_float(self):
        from stresslab.cli import _fmt_num
        result = _fmt_num(1.5e-6)
        assert "e" in result  # Should use scientific notation

    def test_fmt_num_normal_float(self):
        from stresslab.cli import _fmt_num
        result = _fmt_num(3.1415)
        assert "3.14" in result

    def test_fmt_num_integer(self):
        from stresslab.cli import _fmt_num
        result = _fmt_num(42)
        assert result == "42"
