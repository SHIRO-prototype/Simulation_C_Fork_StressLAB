"""Tests for Phase 4 product hardening.

Covers:
  - Config dataclass validation (__post_init__ guards)
  - YAML config loading (_load_config_yaml)
  - Progress bar integration (tqdm wrapping)
  - DRY versioning (single source of truth)
  - Doctor command diagnostics
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from stresslab import __version__
from stresslab.types import (
    ConfigValidationError,
    ProcessNoiseConfig,
    MeasurementConfig,
    ManeuverConfig,
    ThresholdV1Config,
    IntegrityV1Config,
    SimulationConfig,
)
from stresslab.modules.scenario_generator import generate_default_scenario


# ===================================================================
# TestConfigValidation — ProcessNoiseConfig
# ===================================================================

class TestProcessNoiseConfigValidation:
    """Validate ProcessNoiseConfig __post_init__ guards."""

    def test_valid_defaults(self):
        cfg = ProcessNoiseConfig()
        assert cfg.sigma_radial == 1e-9
        assert cfg.scale == 1.0

    def test_valid_zero_sigma(self):
        cfg = ProcessNoiseConfig(sigma_radial=0.0, sigma_tangential=0.0, sigma_normal=0.0)
        assert cfg.sigma_radial == 0.0

    def test_negative_sigma_radial(self):
        with pytest.raises(ConfigValidationError, match="sigma_radial"):
            ProcessNoiseConfig(sigma_radial=-1e-9)

    def test_negative_sigma_tangential(self):
        with pytest.raises(ConfigValidationError, match="sigma_tangential"):
            ProcessNoiseConfig(sigma_tangential=-0.001)

    def test_negative_sigma_normal(self):
        with pytest.raises(ConfigValidationError, match="sigma_normal"):
            ProcessNoiseConfig(sigma_normal=-1.0)

    def test_zero_scale(self):
        with pytest.raises(ConfigValidationError, match="scale"):
            ProcessNoiseConfig(scale=0.0)

    def test_negative_scale(self):
        with pytest.raises(ConfigValidationError, match="scale"):
            ProcessNoiseConfig(scale=-1.0)

    def test_large_valid_scale(self):
        cfg = ProcessNoiseConfig(scale=1e6)
        assert cfg.scale == 1e6


# ===================================================================
# TestConfigValidation — MeasurementConfig
# ===================================================================

class TestMeasurementConfigValidation:
    """Validate MeasurementConfig __post_init__ guards."""

    def test_valid_defaults(self):
        cfg = MeasurementConfig()
        assert cfg.update_interval == 3600.0
        assert cfg.outage_windows == []

    def test_valid_with_outage_windows(self):
        cfg = MeasurementConfig(outage_windows=[
            {"start": 100.0, "end": 200.0},
            {"start": 500.0, "end": 1000.0},
        ])
        assert len(cfg.outage_windows) == 2

    def test_zero_update_interval(self):
        with pytest.raises(ConfigValidationError, match="update_interval"):
            MeasurementConfig(update_interval=0.0)

    def test_negative_update_interval(self):
        with pytest.raises(ConfigValidationError, match="update_interval"):
            MeasurementConfig(update_interval=-60.0)

    def test_zero_noise_sigma(self):
        with pytest.raises(ConfigValidationError, match="noise_sigma_pos"):
            MeasurementConfig(noise_sigma_pos=0.0)

    def test_negative_noise_sigma(self):
        with pytest.raises(ConfigValidationError, match="noise_sigma_pos"):
            MeasurementConfig(noise_sigma_pos=-0.01)

    def test_outage_missing_start(self):
        with pytest.raises(ConfigValidationError, match="outage_windows\\[0\\]"):
            MeasurementConfig(outage_windows=[{"end": 200.0}])

    def test_outage_missing_end(self):
        with pytest.raises(ConfigValidationError, match="outage_windows\\[0\\]"):
            MeasurementConfig(outage_windows=[{"start": 100.0}])

    def test_outage_not_dict(self):
        with pytest.raises(ConfigValidationError, match="outage_windows\\[0\\]"):
            MeasurementConfig(outage_windows=[(100.0, 200.0)])

    def test_outage_end_equals_start(self):
        with pytest.raises(ConfigValidationError, match="end.*must be > start"):
            MeasurementConfig(outage_windows=[{"start": 100.0, "end": 100.0}])

    def test_outage_end_before_start(self):
        with pytest.raises(ConfigValidationError, match="end.*must be > start"):
            MeasurementConfig(outage_windows=[{"start": 200.0, "end": 100.0}])

    def test_second_outage_invalid(self):
        with pytest.raises(ConfigValidationError, match="outage_windows\\[1\\]"):
            MeasurementConfig(outage_windows=[
                {"start": 100.0, "end": 200.0},
                {"start": 500.0, "end": 400.0},
            ])


# ===================================================================
# TestConfigValidation — ManeuverConfig
# ===================================================================

class TestManeuverConfigValidation:
    """Validate ManeuverConfig __post_init__ guards."""

    def test_valid_defaults(self):
        cfg = ManeuverConfig()
        assert cfg.enabled is False
        assert cfg.delta_v_sigma == 0.001

    def test_valid_zero_delta_v(self):
        cfg = ManeuverConfig(delta_v_sigma=0.0)
        assert cfg.delta_v_sigma == 0.0

    def test_negative_delta_v(self):
        with pytest.raises(ConfigValidationError, match="delta_v_sigma"):
            ManeuverConfig(delta_v_sigma=-0.001)

    def test_negative_execution_time(self):
        with pytest.raises(ConfigValidationError, match="execution_time"):
            ManeuverConfig(execution_time=-10.0)

    def test_valid_zero_execution_time(self):
        cfg = ManeuverConfig(execution_time=0.0)
        assert cfg.execution_time == 0.0


# ===================================================================
# TestConfigValidation — ThresholdV1Config
# ===================================================================

class TestThresholdV1ConfigValidation:
    """Validate ThresholdV1Config __post_init__ guards."""

    def test_valid_defaults(self):
        cfg = ThresholdV1Config()
        assert cfg.pc_threshold == 1e-4

    def test_zero_pc_threshold(self):
        with pytest.raises(ConfigValidationError, match="pc_threshold"):
            ThresholdV1Config(pc_threshold=0.0)

    def test_negative_pc_threshold(self):
        with pytest.raises(ConfigValidationError, match="pc_threshold"):
            ThresholdV1Config(pc_threshold=-1e-4)

    def test_valid_miss_distance_none(self):
        cfg = ThresholdV1Config(miss_distance_threshold=None)
        assert cfg.miss_distance_threshold is None

    def test_zero_miss_distance(self):
        with pytest.raises(ConfigValidationError, match="miss_distance_threshold"):
            ThresholdV1Config(miss_distance_threshold=0.0)

    def test_negative_miss_distance(self):
        with pytest.raises(ConfigValidationError, match="miss_distance_threshold"):
            ThresholdV1Config(miss_distance_threshold=-1.0)

    def test_valid_miss_distance(self):
        cfg = ThresholdV1Config(miss_distance_threshold=0.5)
        assert cfg.miss_distance_threshold == 0.5

    def test_zero_tca_gate(self):
        with pytest.raises(ConfigValidationError, match="time_to_tca_gate"):
            ThresholdV1Config(time_to_tca_gate=0.0)

    def test_negative_tca_gate(self):
        with pytest.raises(ConfigValidationError, match="time_to_tca_gate"):
            ThresholdV1Config(time_to_tca_gate=-86400.0)


# ===================================================================
# TestConfigValidation — IntegrityV1Config
# ===================================================================

class TestIntegrityV1ConfigValidation:
    """Validate IntegrityV1Config __post_init__ guards."""

    def test_valid_defaults(self):
        cfg = IntegrityV1Config()
        assert cfg.weights.shape == (4,)

    def test_wrong_weight_count(self):
        with pytest.raises(ConfigValidationError, match="4 elements"):
            IntegrityV1Config(weights=np.array([0.5, 0.5]))

    def test_negative_weight(self):
        with pytest.raises(ConfigValidationError, match="weights must be >= 0"):
            IntegrityV1Config(weights=np.array([0.4, -0.2, 0.2, 0.2]))

    def test_zero_weights_valid(self):
        cfg = IntegrityV1Config(weights=np.array([0.0, 0.0, 0.0, 0.0]))
        assert np.all(cfg.weights == 0)

    def test_thresholds_not_increasing(self):
        with pytest.raises(ConfigValidationError, match="strictly increasing"):
            IntegrityV1Config(
                threshold_monitor_to_watch=0.5,
                threshold_watch_to_warning=0.5,
                threshold_warning_to_critical=0.75,
            )

    def test_thresholds_reversed(self):
        with pytest.raises(ConfigValidationError, match="strictly increasing"):
            IntegrityV1Config(
                threshold_monitor_to_watch=0.75,
                threshold_watch_to_warning=0.50,
                threshold_warning_to_critical=0.25,
            )

    def test_zero_pc_ref(self):
        with pytest.raises(ConfigValidationError, match="pc_ref"):
            IntegrityV1Config(pc_ref=0.0)

    def test_zero_cov_norm_ref(self):
        with pytest.raises(ConfigValidationError, match="cov_norm_ref"):
            IntegrityV1Config(cov_norm_ref=0.0)

    def test_zero_growth_rate_ref(self):
        with pytest.raises(ConfigValidationError, match="growth_rate_ref"):
            IntegrityV1Config(growth_rate_ref=0.0)

    def test_zero_staleness_ref(self):
        with pytest.raises(ConfigValidationError, match="staleness_ref"):
            IntegrityV1Config(staleness_ref=0.0)

    def test_negative_pc_ref(self):
        with pytest.raises(ConfigValidationError, match="pc_ref"):
            IntegrityV1Config(pc_ref=-1e-4)


# ===================================================================
# TestConfigValidation — SimulationConfig
# ===================================================================

class TestSimulationConfigValidation:
    """Validate SimulationConfig __post_init__ guards."""

    def test_valid_defaults(self):
        cfg = SimulationConfig()
        assert cfg.dt == 60.0
        assert cfg.t_end == 259200.0

    def test_valid_scenario(self):
        cfg = generate_default_scenario()
        assert cfg.state_obj1.shape == (6,)
        assert cfg.cov_obj1.shape == (6, 6)

    def test_zero_dt(self):
        with pytest.raises(ConfigValidationError, match="dt must be > 0"):
            SimulationConfig(dt=0.0)

    def test_negative_dt(self):
        with pytest.raises(ConfigValidationError, match="dt must be > 0"):
            SimulationConfig(dt=-60.0)

    def test_t_end_equals_t_start(self):
        with pytest.raises(ConfigValidationError, match="t_end.*must be > t_start"):
            SimulationConfig(t_start=100.0, t_end=100.0)

    def test_t_end_before_t_start(self):
        with pytest.raises(ConfigValidationError, match="t_end.*must be > t_start"):
            SimulationConfig(t_start=200.0, t_end=100.0)

    def test_zero_hard_body_radius(self):
        with pytest.raises(ConfigValidationError, match="combined_hard_body_radius"):
            SimulationConfig(combined_hard_body_radius=0.0)

    def test_negative_hard_body_radius(self):
        with pytest.raises(ConfigValidationError, match="combined_hard_body_radius"):
            SimulationConfig(combined_hard_body_radius=-0.02)

    def test_wrong_state_shape(self):
        with pytest.raises(ConfigValidationError, match="state_obj1"):
            SimulationConfig(state_obj1=np.zeros(3))

    def test_wrong_cov_shape(self):
        with pytest.raises(ConfigValidationError, match="cov_obj1"):
            SimulationConfig(cov_obj1=np.eye(3))

    def test_wrong_state_obj2_shape(self):
        with pytest.raises(ConfigValidationError, match="state_obj2"):
            SimulationConfig(state_obj2=np.zeros(4))

    def test_wrong_cov_obj2_shape(self):
        with pytest.raises(ConfigValidationError, match="cov_obj2"):
            SimulationConfig(cov_obj2=np.eye(4))


# ===================================================================
# TestConfigValidationError — exception type
# ===================================================================

class TestConfigValidationError:
    """Verify ConfigValidationError is a ValueError subclass."""

    def test_is_value_error(self):
        assert issubclass(ConfigValidationError, ValueError)

    def test_catchable_as_value_error(self):
        with pytest.raises(ValueError):
            raise ConfigValidationError("test")

    def test_message_preserved(self):
        err = ConfigValidationError("test message")
        assert str(err) == "test message"


# ===================================================================
# TestYAMLConfigLoading
# ===================================================================

class TestYAMLConfigLoading:
    """Test _load_config_yaml helper from cli.py."""

    def test_load_default_scenario(self):
        from stresslab.cli import _load_config_yaml
        cfg_path = Path(__file__).parent.parent / "configs" / "default_scenario.yaml"
        if not cfg_path.exists():
            pytest.skip("default_scenario.yaml not found")
        cfg = _load_config_yaml(cfg_path)
        assert isinstance(cfg, SimulationConfig)
        assert cfg.state_obj1.shape == (6,)
        assert cfg.cov_obj1.shape == (6, 6)
        assert cfg.dt == 60.0
        assert cfg.t_end == 259200.0

    def test_load_default_scenario_outage_windows(self):
        from stresslab.cli import _load_config_yaml
        cfg_path = Path(__file__).parent.parent / "configs" / "default_scenario.yaml"
        if not cfg_path.exists():
            pytest.skip("default_scenario.yaml not found")
        cfg = _load_config_yaml(cfg_path)
        assert len(cfg.measurement.outage_windows) == 1
        assert cfg.measurement.outage_windows[0]["start"] == 77760.0

    def test_load_custom_yaml(self, tmp_path):
        from stresslab.cli import _load_config_yaml
        yaml_content = textwrap.dedent("""\
            seed: 123
            miss_distance_km: 1.0
            dynamics_model: two_body

            timeline:
              t_start: 0.0
              t_end: 86400.0
              dt: 120.0

            measurement:
              update_interval: 1800.0
              noise_sigma_pos: 0.005
              outage_windows: []

            process_noise:
              sigma_radial: 2.0e-9
              sigma_tangential: 2.0e-9
              sigma_normal: 2.0e-9
              scale: 2.0

            risk:
              combined_hard_body_radius: 0.03
        """)
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(yaml_content)

        cfg = _load_config_yaml(yaml_file)
        assert cfg.dt == 120.0
        assert cfg.t_end == 86400.0
        assert cfg.measurement.update_interval == 1800.0
        assert cfg.process_noise.scale == 2.0
        assert cfg.combined_hard_body_radius == 0.03

    def test_load_minimal_yaml_uses_defaults(self, tmp_path):
        """A YAML with only a seed should work, using defaults for everything."""
        from stresslab.cli import _load_config_yaml
        yaml_file = tmp_path / "minimal.yaml"
        yaml_file.write_text("seed: 99\n")

        cfg = _load_config_yaml(yaml_file)
        assert isinstance(cfg, SimulationConfig)
        assert cfg.seed == 99
        assert cfg.dt == 60.0  # default
        assert cfg.measurement.update_interval == 3600.0  # default

    def test_load_yaml_invalid_process_noise(self, tmp_path):
        """YAML that produces invalid config should raise ConfigValidationError."""
        from stresslab.cli import _load_config_yaml
        yaml_content = textwrap.dedent("""\
            seed: 42
            process_noise:
              sigma_radial: -1.0
              scale: 1.0
        """)
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(ConfigValidationError, match="sigma_radial"):
            _load_config_yaml(yaml_file)


# ===================================================================
# TestProgressBarIntegration
# ===================================================================

class TestProgressBarIntegration:
    """Test that progress=True/False does not crash simulation."""

    @pytest.fixture
    def short_config(self):
        """Generate a short scenario for quick tests."""
        return generate_default_scenario(
            seed=42, t_end=600.0, dt=60.0,  # 10 steps
        )

    def test_simulation_no_progress(self, short_config):
        from stresslab.modules.simulation_runner import run_simulation
        result = run_simulation(short_config, progress=False)
        assert "run_id" in result
        assert result["logger"] is not None

    def test_simulation_with_progress(self, short_config):
        from stresslab.modules.simulation_runner import run_simulation
        result = run_simulation(short_config, progress=True)
        assert "run_id" in result
        assert result["logger"] is not None

    def test_monte_carlo_no_progress(self, short_config):
        from stresslab.modules.monte_carlo import run_monte_carlo
        df = run_monte_carlo(
            short_config, n_runs=2, progress=False,
            randomize_ic=False, randomize_outages=False, randomize_pn=False,
        )
        assert len(df) == 2

    def test_monte_carlo_with_progress(self, short_config):
        from stresslab.modules.monte_carlo import run_monte_carlo
        df = run_monte_carlo(
            short_config, n_runs=2, progress=True,
            randomize_ic=False, randomize_outages=False, randomize_pn=False,
        )
        assert len(df) == 2

    def test_progress_does_not_change_results(self, short_config):
        """Results should be identical with or without progress bars."""
        from stresslab.modules.simulation_runner import run_simulation
        r1 = run_simulation(short_config, progress=False)
        r2 = run_simulation(short_config, progress=True)
        assert r1["run_id"] == r2["run_id"]
        assert (
            r1["threshold_v1_trigger_time"] == r2["threshold_v1_trigger_time"]
        )
        assert (
            r1["integrity_v1_trigger_time"] == r2["integrity_v1_trigger_time"]
        )


# ===================================================================
# TestDRYVersion
# ===================================================================

class TestDRYVersion:
    """Verify single-source-of-truth versioning."""

    def test_version_is_string(self):
        assert isinstance(__version__, str)

    def test_version_is_semver_like(self):
        parts = __version__.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts[:2])

    def test_cli_version_matches_package(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_accessible_from_package(self):
        import stresslab
        assert hasattr(stresslab, "__version__")
        assert stresslab.__version__ == __version__


# ===================================================================
# TestDoctorCommand
# ===================================================================

class TestDoctorCommand:
    """Test the 'stresslab doctor' diagnostic command."""

    def test_doctor_exits_cleanly(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0

    def test_doctor_shows_version(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert __version__ in result.output

    def test_doctor_checks_python_version(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert "Python" in result.output
        assert "[OK]" in result.output

    def test_doctor_checks_core_deps(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        for dep in ["numpy", "scipy", "pandas", "pyarrow", "pyyaml", "click"]:
            assert dep in result.output

    def test_doctor_checks_optional_deps(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert "tqdm" in result.output

    def test_doctor_checks_scenario_generation(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert "scenario generation" in result.output.lower()

    def test_doctor_shows_contract_version(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert "Contract version" in result.output

    def test_doctor_shows_schema_version(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert "Schema version" in result.output

    def test_doctor_all_checks_passed(self):
        from stresslab.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert "All checks passed" in result.output


# ===================================================================
# TestCLIRunCommand
# ===================================================================

class TestCLIRunCommand:
    """Test the CLI run command with --config option."""

    def test_run_with_config_flag(self, tmp_path):
        from stresslab.cli import main
        yaml_content = textwrap.dedent("""\
            seed: 42
            timeline:
              t_end: 600.0
              dt: 60.0
            measurement:
              update_interval: 300.0
              noise_sigma_pos: 0.01
              outage_windows: []
        """)
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)
        output_dir = tmp_path / "outputs"

        runner = CliRunner()
        result = runner.invoke(main, [
            "run",
            "--config", str(yaml_file),
            "--output-dir", str(output_dir),
        ])
        assert result.exit_code == 0
        assert "Run ID" in result.output

    def test_run_without_config(self, tmp_path):
        from stresslab.cli import main
        output_dir = tmp_path / "outputs"

        runner = CliRunner()
        result = runner.invoke(main, [
            "run",
            "--t-end", "600",
            "--dt", "60",
            "--output-dir", str(output_dir),
        ])
        assert result.exit_code == 0
        assert "Run ID" in result.output
