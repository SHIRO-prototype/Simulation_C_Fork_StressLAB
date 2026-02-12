"""Tests for scenario generator and scenario hashing lock."""

import copy
import numpy as np
import pytest
from pathlib import Path
import tempfile

from stresslab.types import (
    DynamicsModel,
    MU_EARTH_KM3S2,
    RE_EARTH_KM,
    SimulationConfig,
    ProcessNoiseConfig,
    MeasurementConfig,
    ManeuverConfig,
    ThresholdV1Config,
    IntegrityV1Config,
)
from stresslab.modules.scenario_generator import (
    generate_default_scenario,
    export_scenario,
)


class TestScenarioGenerator:
    def test_deterministic(self):
        """Same seed should produce identical scenarios."""
        c1 = generate_default_scenario(seed=42)
        c2 = generate_default_scenario(seed=42)
        np.testing.assert_array_equal(c1.state_obj1, c2.state_obj1)
        np.testing.assert_array_equal(c1.state_obj2, c2.state_obj2)

    def test_different_seeds(self):
        """Different seeds should produce different scenarios."""
        c1 = generate_default_scenario(seed=42)
        c2 = generate_default_scenario(seed=99)
        assert c1.run_id() != c2.run_id()

    def test_valid_orbit(self):
        """Generated states should be valid orbits."""
        config = generate_default_scenario()
        r = np.linalg.norm(config.state_obj1[:3])
        assert r > RE_EARTH_KM
        assert r < RE_EARTH_KM + 1000.0

    def test_export(self):
        """Should export valid JSON."""
        config = generate_default_scenario()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_scenario(config, Path(tmpdir))
            assert path.exists()
            import json
            with open(path) as f:
                data = json.load(f)
            assert "run_id" in data
            assert "state_obj1" in data

    def test_run_id_hash(self):
        """Run ID should be a hex string."""
        config = generate_default_scenario()
        rid = config.run_id()
        assert len(rid) == 16
        int(rid, 16)  # should not raise


class TestScenarioHashingLock:
    """Verify that the run_id changes when any output-affecting parameter changes.

    The hashing lock ensures that two configs with different parameters
    always produce different run_ids, preventing cache collisions.
    """

    def _base_config(self) -> SimulationConfig:
        return generate_default_scenario(seed=42)

    def test_identical_configs_same_hash(self):
        c1 = self._base_config()
        c2 = self._base_config()
        assert c1.run_id() == c2.run_id()

    def test_seed_changes_hash(self):
        c1 = self._base_config()
        c2 = self._base_config()
        c2.seed = 99
        assert c1.run_id() != c2.run_id()

    def test_dt_changes_hash(self):
        c1 = self._base_config()
        c2 = self._base_config()
        c2.dt = 30.0
        assert c1.run_id() != c2.run_id()

    def test_process_noise_changes_hash(self):
        c1 = self._base_config()
        c2 = self._base_config()
        c2.process_noise.sigma_radial = 1e-6  # different from default
        assert c1.run_id() != c2.run_id()

    def test_measurement_interval_changes_hash(self):
        c1 = self._base_config()
        c2 = self._base_config()
        c2.measurement.update_interval = 1800.0
        assert c1.run_id() != c2.run_id()

    def test_outage_windows_change_hash(self):
        c1 = self._base_config()
        c2 = self._base_config()
        c2.measurement.outage_windows = [{"start": 1000.0, "end": 2000.0}]
        assert c1.run_id() != c2.run_id()

    def test_threshold_v1_changes_hash(self):
        c1 = self._base_config()
        c2 = self._base_config()
        c2.threshold_v1.pc_threshold = 1e-3
        assert c1.run_id() != c2.run_id()

    def test_integrity_v1_changes_hash(self):
        c1 = self._base_config()
        c2 = self._base_config()
        c2.integrity_v1.threshold_warning_to_critical = 0.9
        assert c1.run_id() != c2.run_id()

    def test_maneuver_changes_hash(self):
        c1 = self._base_config()
        c2 = self._base_config()
        c2.maneuver.enabled = True
        assert c1.run_id() != c2.run_id()

    def test_dynamics_model_changes_hash(self):
        c1 = self._base_config()
        c2 = self._base_config()
        c2.dynamics_model = DynamicsModel.TWO_BODY
        assert c1.run_id() != c2.run_id()

    def test_hard_body_radius_changes_hash(self):
        c1 = self._base_config()
        c2 = self._base_config()
        c2.combined_hard_body_radius = 0.05
        assert c1.run_id() != c2.run_id()

    def test_covariance_changes_hash(self):
        c1 = self._base_config()
        c2 = self._base_config()
        c2.cov_obj1 = np.eye(6) * 1e-2  # much larger initial uncertainty
        assert c1.run_id() != c2.run_id()

    def test_hash_includes_code_version(self):
        """The run_id incorporates the code version tag."""
        import stresslab
        config = self._base_config()
        # Get the serializable dict and verify code_version is present
        blob = config._serializable(stresslab.__version__)
        assert "code_version" in blob
        assert blob["code_version"] == stresslab.__version__

    def test_serializable_includes_all_config_sections(self):
        """Verify the serializable dict covers all major config sections."""
        config = self._base_config()
        blob = config._serializable("1.0.0")
        required_sections = {
            "code_version", "state_obj1", "state_obj2", "cov_obj1", "cov_obj2",
            "dynamics_model", "process_noise", "measurement", "maneuver",
            "threshold_v1", "integrity_v1",
            "t_start", "t_end", "dt",
            "combined_hard_body_radius", "seed",
        }
        assert required_sections.issubset(set(blob.keys()))
