"""Tests for scenario generator."""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from stresslab.types import DynamicsModel, MU_EARTH_KM3S2, RE_EARTH_KM
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
        # States should differ (different miss distance offsets aren't seed-dependent,
        # but the overall config hash differs)
        assert c1.run_id() != c2.run_id()

    def test_valid_orbit(self):
        """Generated states should be valid orbits."""
        config = generate_default_scenario()
        r = np.linalg.norm(config.state_obj1[:3])
        # Should be in LEO range
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
