"""Tests for the measurement model."""

import pytest

from stresslab.types import MeasurementConfig
from stresslab.modules.measurement_model import evaluate_measurement


class TestMeasurementModel:
    def test_first_update(self):
        """First measurement should apply after interval."""
        config = MeasurementConfig(update_interval=3600.0, outage_windows=[])
        result = evaluate_measurement(t=3600.0, last_update_time=0.0, config=config)
        assert result.applied is True
        assert result.time_since_last_update == 3600.0

    def test_too_early(self):
        """Measurement should not apply before interval."""
        config = MeasurementConfig(update_interval=3600.0, outage_windows=[])
        result = evaluate_measurement(t=1800.0, last_update_time=0.0, config=config)
        assert result.applied is False

    def test_outage_blocks_update(self):
        """Measurement should not apply during outage."""
        config = MeasurementConfig(
            update_interval=3600.0,
            outage_windows=[{"start": 3000.0, "end": 7000.0}],
        )
        result = evaluate_measurement(t=3600.0, last_update_time=0.0, config=config)
        assert result.applied is False

    def test_after_outage(self):
        """Measurement should apply after outage ends."""
        config = MeasurementConfig(
            update_interval=3600.0,
            outage_windows=[{"start": 3000.0, "end": 7000.0}],
        )
        result = evaluate_measurement(t=7200.0, last_update_time=0.0, config=config)
        assert result.applied is True

    def test_staleness_accumulates(self):
        """Staleness should accumulate during outage."""
        config = MeasurementConfig(
            update_interval=3600.0,
            outage_windows=[{"start": 3000.0, "end": 50000.0}],
        )
        result = evaluate_measurement(t=40000.0, last_update_time=0.0, config=config)
        assert result.applied is False
        assert result.time_since_last_update == 40000.0
