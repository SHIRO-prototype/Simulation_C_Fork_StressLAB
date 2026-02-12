"""Measurement Model - controls observation cadence and staleness.

Determines at each timestep whether a measurement update is applied,
accounting for scheduled update intervals and outage windows.
"""

from __future__ import annotations

from stresslab.types import MeasurementConfig, MeasurementResult


def evaluate_measurement(
    t: float,
    last_update_time: float,
    config: MeasurementConfig,
) -> MeasurementResult:
    """Determine if a measurement is applied at time t.

    A measurement is applied if:
      1. Enough time has passed since the last update (>= update_interval)
      2. The current time is NOT within any outage window

    Args:
        t: current time (seconds from epoch)
        last_update_time: time of last measurement update
        config: measurement configuration

    Returns:
        MeasurementResult with applied flag and staleness.
    """
    time_since = t - last_update_time

    # Check if we are in an outage window
    in_outage = False
    for window in config.outage_windows:
        if window["start"] <= t <= window["end"]:
            in_outage = True
            break

    # Measurement applied if interval elapsed and not in outage
    interval_met = time_since >= config.update_interval
    applied = interval_met and not in_outage

    return MeasurementResult(
        applied=applied,
        time_since_last_update=time_since,
    )
