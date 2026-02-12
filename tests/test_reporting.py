"""Tests for the reporting layer (Phase 3).

Covers:
  - benchmark_report: single-run report generation
  - monte_carlo_report: batch/MC report generation
  - exporters: multi-format output (JSON, CSV, LaTeX)
  - compare: run-pair comparison analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stresslab.metrics_contract import MetricsSummary


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_summary(**overrides) -> MetricsSummary:
    """Factory for MetricsSummary with sensible defaults."""
    defaults = dict(
        contract_version="2.0.0",
        threshold_v1_trigger_time=50000.0,
        integrity_v1_trigger_time=48000.0,
        decision_compression_window=2000.0,
        false_safe_rate=0.05,
        false_alert_rate=0.10,
        outage_sensitivity_score=7200.0,
        max_staleness=7200.0,
        max_pc_degraded=1.5e-4,
        max_pc_reference=1.2e-4,
        max_cov_trace=0.005,
        decision_instability_index=0.12,
        decision_transitions_per_hour=2.5,
        decision_entropy=0.85,
        mean_pc_drift=0.25,
        max_pc_drift=1.8,
        staleness_pc_correlation=0.65,
        mean_freshness=0.80,
        min_freshness=0.40,
        total_timesteps=4320,
    )
    defaults.update(overrides)
    return MetricsSummary(**defaults)


def _make_mc_df(n_runs: int = 20) -> pd.DataFrame:
    """Build a mock Monte Carlo DataFrame with known values."""
    rng = np.random.RandomState(42)
    data = {
        "run_index": list(range(n_runs)),
        "seed": [42 + i for i in range(n_runs)],
        "process_noise_scale": rng.uniform(0.5, 2.0, n_runs).tolist(),
        "total_outage_duration": np.linspace(0, 14400, n_runs).tolist(),
        "n_outage_windows": rng.randint(0, 5, n_runs).tolist(),
        "threshold_v1_trigger_time": (40000 + rng.normal(0, 2000, n_runs)).tolist(),
        "integrity_v1_trigger_time": (38000 + rng.normal(0, 2000, n_runs)).tolist(),
        "max_pc_degraded": (1e-4 + rng.exponential(1e-5, n_runs)).tolist(),
        "max_cov_trace": (0.003 + rng.exponential(0.001, n_runs)).tolist(),
        "max_staleness": (3600 + rng.exponential(1800, n_runs)).tolist(),
        "max_miss_distance": rng.uniform(0.5, 2.0, n_runs).tolist(),
        "min_miss_distance": rng.uniform(0.01, 0.3, n_runs).tolist(),
        "decision_compression_window": (2000 + rng.normal(0, 500, n_runs)).tolist(),
    }
    return pd.DataFrame(data)


def _make_timeseries_df(n: int = 100) -> pd.DataFrame:
    """Build a mock time-series DataFrame."""
    t = np.linspace(0, 86400, n)
    return pd.DataFrame({
        "timestamp": t,
        "pc_reference": np.exp(-((t - 43200) ** 2) / (2 * 10000 ** 2)) * 1e-4,
        "pc_degraded": np.exp(-((t - 43200) ** 2) / (2 * 10000 ** 2)) * 1.5e-4,
        "pc_drift": np.abs(np.random.RandomState(0).normal(0, 0.1, n)),
        "cov_trace_obj1": 0.001 + np.linspace(0, 0.004, n),
        "staleness_obj1": np.clip(np.linspace(0, 7200, n), 0, None),
        "freshness_score": np.linspace(1.0, 0.4, n),
        "integrity_v1_score": np.linspace(0.1, 0.8, n),
        "miss_distance": 0.5 + np.sin(t / 10000) * 0.2,
    })


# ===========================================================================
# TestBenchmarkReport
# ===========================================================================

class TestBenchmarkReport:
    """Tests for stresslab.reporting.benchmark_report."""

    def test_timing_shift_integrity_first(self):
        from stresslab.reporting.benchmark_report import timing_shift_table
        s = _make_summary()  # integrity triggers first (dcw = +2000)
        table = timing_shift_table(s)
        assert table["threshold_v1_trigger_s"] == 50000.0
        assert table["integrity_v1_trigger_s"] == 48000.0
        assert table["dcw_seconds"] == 2000.0
        assert table["early_warning_model"] == "integrity_v1"
        assert "integrity-v1 triggered" in table["dcw_description"]

    def test_timing_shift_threshold_first(self):
        from stresslab.reporting.benchmark_report import timing_shift_table
        s = _make_summary(
            threshold_v1_trigger_time=45000.0,
            integrity_v1_trigger_time=48000.0,
            decision_compression_window=-3000.0,
        )
        table = timing_shift_table(s)
        assert table["early_warning_model"] == "threshold_v1"
        assert table["dcw_seconds"] == -3000.0

    def test_timing_shift_simultaneous(self):
        from stresslab.reporting.benchmark_report import timing_shift_table
        s = _make_summary(
            threshold_v1_trigger_time=48000.0,
            integrity_v1_trigger_time=48000.0,
            decision_compression_window=0.0,
        )
        table = timing_shift_table(s)
        assert table["early_warning_model"] == "simultaneous"

    def test_timing_shift_incomplete(self):
        from stresslab.reporting.benchmark_report import timing_shift_table
        s = _make_summary(
            threshold_v1_trigger_time=None,
            integrity_v1_trigger_time=48000.0,
            decision_compression_window=None,
        )
        table = timing_shift_table(s)
        assert table["early_warning_model"] == "incomplete"
        assert table["threshold_v1_trigger_s"] is None

    def test_decision_performance_table(self):
        from stresslab.reporting.benchmark_report import decision_performance_table
        s = _make_summary()
        table = decision_performance_table(s)
        assert table["false_safe_rate"] == pytest.approx(0.05)
        assert table["false_alert_rate"] == pytest.approx(0.10)
        assert table["decision_instability_index"] == pytest.approx(0.12)
        assert table["decision_transitions_per_hour"] == pytest.approx(2.5)
        assert table["decision_entropy_nats"] == pytest.approx(0.85)

    def test_uncertainty_table(self):
        from stresslab.reporting.benchmark_report import uncertainty_table
        s = _make_summary()
        table = uncertainty_table(s)
        assert table["max_cov_trace_km2"] == pytest.approx(0.005)
        assert table["max_staleness_s"] == pytest.approx(7200.0)
        assert table["mean_pc_drift"] == pytest.approx(0.25)
        assert table["max_pc_drift"] == pytest.approx(1.8)
        assert table["staleness_pc_correlation"] == pytest.approx(0.65)
        assert table["mean_freshness"] == pytest.approx(0.80)
        assert table["min_freshness"] == pytest.approx(0.40)
        assert table["max_pc_degraded"] == pytest.approx(1.5e-4)
        assert table["max_pc_reference"] == pytest.approx(1.2e-4)
        assert table["outage_sensitivity_score_s"] == pytest.approx(7200.0)

    def test_peak_values_table(self):
        from stresslab.reporting.benchmark_report import peak_values_table
        s = _make_summary()
        table = peak_values_table(s)
        assert table["max_pc_degraded"] == pytest.approx(1.5e-4)
        assert table["max_pc_reference"] == pytest.approx(1.2e-4)
        assert table["max_cov_trace_km2"] == pytest.approx(0.005)
        assert table["max_staleness_s"] == pytest.approx(7200.0)
        assert table["total_timesteps"] == 4320

    def test_timeseries_curves(self):
        from stresslab.reporting.benchmark_report import timeseries_curves
        df = _make_timeseries_df(50)
        curves = timeseries_curves(df)
        assert "timestamp" in curves
        assert "pc_reference" in curves
        assert "pc_degraded" in curves
        assert len(curves["timestamp"]) == 50

    def test_timeseries_curves_missing_columns(self):
        from stresslab.reporting.benchmark_report import timeseries_curves
        df = pd.DataFrame({"timestamp": [1, 2, 3], "some_other": [4, 5, 6]})
        curves = timeseries_curves(df)
        assert "timestamp" in curves
        assert "pc_reference" not in curves

    def test_generate_benchmark_report_structure(self):
        from stresslab.reporting.benchmark_report import generate_benchmark_report
        s = _make_summary()
        report = generate_benchmark_report(s, run_id="abc123")
        assert report["report_type"] == "single_run_benchmark"
        assert report["contract_version"] == "2.0.0"
        assert report["run_id"] == "abc123"
        assert "timing_shift" in report
        assert "decision_performance" in report
        assert "uncertainty_degradation" in report
        assert "peak_values" in report
        assert "timeseries_curves" not in report  # no df provided

    def test_generate_benchmark_report_with_timeseries(self):
        from stresslab.reporting.benchmark_report import generate_benchmark_report
        s = _make_summary()
        df = _make_timeseries_df()
        report = generate_benchmark_report(s, df=df, run_id="xyz789")
        assert "timeseries_curves" in report
        assert "timestamp" in report["timeseries_curves"]

    def test_generate_benchmark_report_no_run_id(self):
        from stresslab.reporting.benchmark_report import generate_benchmark_report
        s = _make_summary()
        report = generate_benchmark_report(s)
        assert "run_id" not in report


# ===========================================================================
# TestMonteCarloReport
# ===========================================================================

class TestMonteCarloReport:
    """Tests for stresslab.reporting.monte_carlo_report."""

    def test_trigger_timing_stats_structure(self):
        from stresslab.reporting.monte_carlo_report import trigger_timing_stats
        mc_df = _make_mc_df()
        stats = trigger_timing_stats(mc_df)
        assert stats["total_runs"] == 20
        assert stats["successful_runs"] == 20
        assert "threshold_v1" in stats
        assert "integrity_v1" in stats
        assert stats["threshold_v1"]["count"] == 20
        assert "mean" in stats["threshold_v1"]
        assert "std" in stats["threshold_v1"]
        assert "median" in stats["threshold_v1"]

    def test_trigger_timing_stats_empty(self):
        from stresslab.reporting.monte_carlo_report import trigger_timing_stats
        mc_df = pd.DataFrame({
            "threshold_v1_trigger_time": [],
            "integrity_v1_trigger_time": [],
            "error": [],
        })
        stats = trigger_timing_stats(mc_df)
        assert stats["successful_runs"] == 0

    def test_trigger_timing_stats_with_errors(self):
        from stresslab.reporting.monte_carlo_report import trigger_timing_stats
        mc_df = _make_mc_df(5)
        mc_df["error"] = [None, None, "crashed", None, "timeout"]
        stats = trigger_timing_stats(mc_df)
        assert stats["total_runs"] == 5
        assert stats["successful_runs"] == 3

    def test_dcw_distribution_report(self):
        from stresslab.reporting.monte_carlo_report import dcw_distribution_report
        mc_df = _make_mc_df()
        dist = dcw_distribution_report(mc_df)
        assert "mean" in dist
        assert "std" in dist
        assert "p50" in dist
        assert "count" in dist
        assert dist["count"] == 20

    def test_dcw_distribution_empty(self):
        from stresslab.reporting.monte_carlo_report import dcw_distribution_report
        mc_df = pd.DataFrame({
            "decision_compression_window": [],
            "error": [],
        })
        dist = dcw_distribution_report(mc_df)
        assert dist == {}

    def test_outage_gradient_report(self):
        from stresslab.reporting.monte_carlo_report import outage_gradient_report
        mc_df = _make_mc_df()
        grad = outage_gradient_report(mc_df)
        assert "threshold_v1_gradient" in grad
        assert "integrity_v1_gradient" in grad
        # Gradients should be float or None
        assert isinstance(grad["threshold_v1_gradient"], (float, type(None)))

    def test_outage_gradient_report_empty(self):
        from stresslab.reporting.monte_carlo_report import outage_gradient_report
        mc_df = pd.DataFrame({
            "threshold_v1_trigger_time": [],
            "integrity_v1_trigger_time": [],
            "total_outage_duration": [],
            "error": [],
        })
        grad = outage_gradient_report(mc_df)
        assert grad == {}

    def test_pc_degradation_stats(self):
        from stresslab.reporting.monte_carlo_report import pc_degradation_stats
        mc_df = _make_mc_df()
        stats = pc_degradation_stats(mc_df)
        assert "mean" in stats
        assert "max" in stats
        assert "min" in stats
        assert stats["count"] == 20

    def test_pc_degradation_stats_empty(self):
        from stresslab.reporting.monte_carlo_report import pc_degradation_stats
        mc_df = pd.DataFrame({"max_pc_degraded": [], "error": []})
        stats = pc_degradation_stats(mc_df)
        assert stats == {}

    def test_model_comparison_table(self):
        from stresslab.reporting.monte_carlo_report import model_comparison_table
        mc_df = _make_mc_df()
        table = model_comparison_table(mc_df)
        assert "threshold_v1" in table
        assert "integrity_v1" in table
        assert "early_warning" in table
        assert table["threshold_v1"]["trigger_rate"] == pytest.approx(1.0)
        ew = table["early_warning"]
        assert ew["both_triggered_runs"] == 20
        total_ew = ew["integrity_v1_first"] + ew["threshold_v1_first"] + ew["simultaneous"]
        assert total_ew == 20

    def test_model_comparison_partial_triggers(self):
        from stresslab.reporting.monte_carlo_report import model_comparison_table
        mc_df = _make_mc_df(10)
        # Make some runs not trigger
        mc_df.loc[0:2, "threshold_v1_trigger_time"] = np.nan
        mc_df.loc[3:5, "integrity_v1_trigger_time"] = np.nan
        table = model_comparison_table(mc_df)
        assert table["threshold_v1"]["trigger_rate"] < 1.0
        assert table["integrity_v1"]["trigger_rate"] < 1.0

    def test_generate_monte_carlo_report_structure(self):
        from stresslab.reporting.monte_carlo_report import generate_monte_carlo_report
        mc_df = _make_mc_df()
        report = generate_monte_carlo_report(mc_df)
        assert report["report_type"] == "monte_carlo_batch"
        assert "trigger_timing" in report
        assert "dcw_distribution" in report
        assert "outage_gradient" in report
        assert "pc_degradation" in report
        assert "model_comparison" in report


# ===========================================================================
# TestExporters
# ===========================================================================

class TestExporters:
    """Tests for stresslab.reporting.exporters."""

    def test_export_json_envelope(self, tmp_path):
        from stresslab.reporting.exporters import export_json
        from stresslab.modules.logging_engine import SCHEMA_VERSION
        from stresslab.metrics_contract import METRICS_CONTRACT_VERSION
        from stresslab import __version__ as STRESSLAB_VERSION

        report = {"report_type": "test", "value": 42}
        out = export_json(report, tmp_path / "report.json")
        assert out.exists()

        with open(out) as f:
            data = json.load(f)

        assert data["schema_version"] == SCHEMA_VERSION
        assert data["metrics_contract_version"] == METRICS_CONTRACT_VERSION
        assert data["stresslab_version"] == STRESSLAB_VERSION
        assert data["report"]["report_type"] == "test"
        assert data["report"]["value"] == 42

    def test_export_json_creates_parent_dirs(self, tmp_path):
        from stresslab.reporting.exporters import export_json
        out = export_json({"a": 1}, tmp_path / "deep" / "nested" / "report.json")
        assert out.exists()

    def test_export_json_numpy_serialization(self, tmp_path):
        from stresslab.reporting.exporters import export_json
        report = {
            "int_val": np.int64(42),
            "float_val": np.float64(3.14),
            "array_val": np.array([1, 2, 3]),
        }
        out = export_json(report, tmp_path / "np.json")
        with open(out) as f:
            data = json.load(f)
        assert data["report"]["int_val"] == 42
        assert data["report"]["float_val"] == pytest.approx(3.14)
        assert data["report"]["array_val"] == [1, 2, 3]

    def test_export_csv_from_dict(self, tmp_path):
        from stresslab.reporting.exporters import export_csv
        data = {"metric_a": 1.0, "metric_b": 2.0, "metric_c": "hello"}
        out = export_csv(data, tmp_path / "flat.csv")
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 1
        assert "metric_a" in df.columns
        assert df["metric_a"].iloc[0] == pytest.approx(1.0)

    def test_export_csv_from_dataframe(self, tmp_path):
        from stresslab.reporting.exporters import export_csv
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        out = export_csv(df, tmp_path / "df.csv")
        loaded = pd.read_csv(out)
        assert len(loaded) == 3
        assert list(loaded.columns) == ["x", "y"]

    def test_export_csv_nested_dict_flattened(self, tmp_path):
        from stresslab.reporting.exporters import export_csv
        data = {"section": {"a": 1, "b": 2}, "top_level": 3}
        out = export_csv(data, tmp_path / "nested.csv")
        df = pd.read_csv(out)
        assert "section.a" in df.columns
        assert "section.b" in df.columns
        assert "top_level" in df.columns

    def test_export_latex_table_structure(self, tmp_path):
        from stresslab.reporting.exporters import export_latex_table
        rows = [("Metric A", "1.23"), ("Metric B", "4.56")]
        out = export_latex_table(
            rows, tmp_path / "table.tex",
            caption="Test Table", label="tab:test",
        )
        assert out.exists()
        content = out.read_text()
        assert r"\begin{table}" in content
        assert r"\end{table}" in content
        assert r"\toprule" in content
        assert r"\midrule" in content
        assert r"\bottomrule" in content
        assert r"\caption{Test Table}" in content
        assert r"\label{tab:test}" in content
        assert "Metric A" in content
        assert "1.23" in content

    def test_export_latex_escapes_special_chars(self, tmp_path):
        from stresslab.reporting.exporters import export_latex_table
        rows = [("Rate_100%", "50%")]
        out = export_latex_table(rows, tmp_path / "escaped.tex")
        content = out.read_text()
        assert r"Rate\_100\%" in content
        assert r"50\%" in content

    def test_report_to_latex_rows(self):
        from stresslab.reporting.benchmark_report import generate_benchmark_report
        from stresslab.reporting.exporters import report_to_latex_rows
        s = _make_summary()
        report = generate_benchmark_report(s)
        rows = report_to_latex_rows(report)
        assert len(rows) > 0
        # All rows should be (str, str) tuples
        for metric, value in rows:
            assert isinstance(metric, str)
            assert isinstance(value, str)
        # Check some expected metrics
        metric_names = [r[0] for r in rows]
        assert "Threshold-v1 trigger" in metric_names
        assert "Integrity-v1 trigger" in metric_names
        assert "DCW" in metric_names
        assert "False-safe rate" in metric_names
        assert "Total timesteps" in metric_names

    def test_export_timeseries_csv_all_columns(self, tmp_path):
        from stresslab.reporting.exporters import export_timeseries_csv
        df = _make_timeseries_df(30)
        out = export_timeseries_csv(df, tmp_path / "ts.csv")
        loaded = pd.read_csv(out)
        assert len(loaded) == 30
        assert set(df.columns) == set(loaded.columns)

    def test_export_timeseries_csv_subset_columns(self, tmp_path):
        from stresslab.reporting.exporters import export_timeseries_csv
        df = _make_timeseries_df(20)
        out = export_timeseries_csv(
            df, tmp_path / "ts_sub.csv",
            columns=["timestamp", "pc_reference", "nonexistent"],
        )
        loaded = pd.read_csv(out)
        assert len(loaded) == 20
        assert "timestamp" in loaded.columns
        assert "pc_reference" in loaded.columns
        assert "nonexistent" not in loaded.columns

    def test_export_timeseries_csv_creates_dirs(self, tmp_path):
        from stresslab.reporting.exporters import export_timeseries_csv
        df = _make_timeseries_df(5)
        out = export_timeseries_csv(df, tmp_path / "a" / "b" / "ts.csv")
        assert out.exists()


# ===========================================================================
# TestCompare
# ===========================================================================

class TestCompare:
    """Tests for stresslab.reporting.compare."""

    def test_compare_summaries_numeric_deltas(self):
        from stresslab.reporting.compare import compare_summaries
        a = _make_summary(max_pc_degraded=1.0e-4, max_staleness=3600.0)
        b = _make_summary(max_pc_degraded=2.0e-4, max_staleness=7200.0)
        result = compare_summaries(a, b, label_a="run_1", label_b="run_2")
        assert result["comparison_type"] == "single_run_pair"
        assert result["label_a"] == "run_1"
        assert result["label_b"] == "run_2"
        metrics = result["metrics"]
        assert len(metrics) > 0
        # Find max_pc_degraded metric
        pc_metric = next(m for m in metrics if m["metric"] == "max_pc_degraded")
        assert pc_metric["value_run_1"] == pytest.approx(1.0e-4)
        assert pc_metric["value_run_2"] == pytest.approx(2.0e-4)
        assert pc_metric["delta"] == pytest.approx(1.0e-4)
        assert pc_metric["rel_delta_pct"] == pytest.approx(100.0)

    def test_compare_summaries_from_dicts(self):
        from stresslab.reporting.compare import compare_summaries
        a = {"max_pc_degraded": 1e-4, "max_staleness": 3600.0, "label": "A"}
        b = {"max_pc_degraded": 2e-4, "max_staleness": 7200.0, "label": "B"}
        result = compare_summaries(a, b)
        metrics = result["metrics"]
        # Only numeric fields shared should appear
        metric_names = {m["metric"] for m in metrics}
        assert "max_pc_degraded" in metric_names
        assert "max_staleness" in metric_names
        assert "label" not in metric_names  # non-numeric

    def test_compare_summaries_zero_base(self):
        from stresslab.reporting.compare import compare_summaries
        a = _make_summary(false_safe_rate=0.0)
        b = _make_summary(false_safe_rate=0.10)
        result = compare_summaries(a, b)
        # With va=0 the denom becomes 1.0, so rel_delta_pct = 0.10*100 = 10
        fsr = next(m for m in result["metrics"] if m["metric"] == "false_safe_rate")
        assert fsr["delta"] == pytest.approx(0.10)
        # denom = max(abs(0), 1e-30) = 1.0 (since 0 < 1e-30 is False, so 1.0)
        # Actually abs(0.0) > 1e-30 is False so denom = 1.0
        assert fsr["rel_delta_pct"] == pytest.approx(10.0)

    def test_load_summary_json_raw(self, tmp_path):
        from stresslab.reporting.compare import load_summary_json
        raw = {"max_pc_degraded": 1e-4, "total_timesteps": 100}
        path = tmp_path / "raw.json"
        path.write_text(json.dumps(raw))
        loaded = load_summary_json(path)
        assert loaded["max_pc_degraded"] == pytest.approx(1e-4)

    def test_load_summary_json_enveloped(self, tmp_path):
        from stresslab.reporting.compare import load_summary_json
        enveloped = {
            "schema_version": "2.0.0",
            "report": {"max_pc_degraded": 2e-4, "total_timesteps": 200},
        }
        path = tmp_path / "enveloped.json"
        path.write_text(json.dumps(enveloped))
        loaded = load_summary_json(path)
        assert loaded["max_pc_degraded"] == pytest.approx(2e-4)
        assert loaded["total_timesteps"] == 200

    def test_compare_from_files(self, tmp_path):
        from stresslab.reporting.compare import compare_from_files
        a = {"max_pc_degraded": 1e-4, "max_staleness": 3600.0}
        b = {"max_pc_degraded": 3e-4, "max_staleness": 5400.0}
        path_a = tmp_path / "run_a.json"
        path_b = tmp_path / "run_b.json"
        path_a.write_text(json.dumps(a))
        path_b.write_text(json.dumps(b))
        result = compare_from_files(path_a, path_b)
        assert result["label_a"] == "run_a"
        assert result["label_b"] == "run_b"
        assert len(result["metrics"]) == 2

    def test_compare_from_files_custom_labels(self, tmp_path):
        from stresslab.reporting.compare import compare_from_files
        a = {"x": 1.0}
        b = {"x": 2.0}
        pa = tmp_path / "a.json"
        pb = tmp_path / "b.json"
        pa.write_text(json.dumps(a))
        pb.write_text(json.dumps(b))
        result = compare_from_files(pa, pb, label_a="baseline", label_b="experiment")
        assert result["label_a"] == "baseline"
        assert result["label_b"] == "experiment"

    def test_comparison_to_dataframe(self):
        from stresslab.reporting.compare import compare_summaries, comparison_to_dataframe
        a = _make_summary()
        b = _make_summary(max_pc_degraded=3e-4)
        comp = compare_summaries(a, b)
        df = comparison_to_dataframe(comp)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "metric" in df.columns
        assert "delta" in df.columns

    def test_comparison_to_dataframe_empty(self):
        from stresslab.reporting.compare import comparison_to_dataframe
        comp = {"metrics": []}
        df = comparison_to_dataframe(comp)
        assert len(df) == 0


# ===========================================================================
# Integration: round-trip benchmark report -> export -> load -> compare
# ===========================================================================

class TestReportingIntegration:
    """End-to-end integration tests across reporting modules."""

    def test_benchmark_to_json_roundtrip(self, tmp_path):
        from stresslab.reporting.benchmark_report import generate_benchmark_report
        from stresslab.reporting.exporters import export_json
        from stresslab.reporting.compare import load_summary_json

        s = _make_summary()
        report = generate_benchmark_report(s, run_id="integration_test")
        out_path = export_json(report, tmp_path / "bench.json")

        loaded = load_summary_json(out_path)
        assert loaded["report_type"] == "single_run_benchmark"
        assert loaded["run_id"] == "integration_test"

    def test_benchmark_to_latex_roundtrip(self, tmp_path):
        from stresslab.reporting.benchmark_report import generate_benchmark_report
        from stresslab.reporting.exporters import report_to_latex_rows, export_latex_table

        s = _make_summary()
        report = generate_benchmark_report(s)
        rows = report_to_latex_rows(report)
        out_path = export_latex_table(
            rows, tmp_path / "bench.tex",
            caption="Integration Test",
        )
        content = out_path.read_text()
        assert r"\begin{table}" in content
        assert "Integration Test" in content

    def test_benchmark_to_csv_roundtrip(self, tmp_path):
        from stresslab.reporting.benchmark_report import generate_benchmark_report
        from stresslab.reporting.exporters import export_csv

        s = _make_summary()
        report = generate_benchmark_report(s)
        out_path = export_csv(report, tmp_path / "bench.csv")
        df = pd.read_csv(out_path)
        assert len(df) == 1
        assert "timing_shift.dcw_seconds" in df.columns

    def test_mc_report_to_json_roundtrip(self, tmp_path):
        from stresslab.reporting.monte_carlo_report import generate_monte_carlo_report
        from stresslab.reporting.exporters import export_json

        mc_df = _make_mc_df()
        report = generate_monte_carlo_report(mc_df)
        out_path = export_json(report, tmp_path / "mc.json")

        with open(out_path) as f:
            data = json.load(f)
        assert data["report"]["report_type"] == "monte_carlo_batch"
        assert "trigger_timing" in data["report"]
