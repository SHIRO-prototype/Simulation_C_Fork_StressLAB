import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { fetchRunSummary, fetchRunTimeseries, runDownloadUrl } from "../lib/api";
import type { TimeseriesResponse } from "../lib/types";
import MetricCard from "../components/MetricCard";
import PostureTimeline from "../components/PostureTimeline";
import PcDriftPlot from "../components/PcDriftPlot";
import InstabilityPlot from "../components/InstabilityPlot";

/* ------------------------------------------------------------------ */
/*  Formatting helpers                                                 */
/* ------------------------------------------------------------------ */

/** Format a trigger-time value: number -> "Xs", null/undefined -> fallback. */
function fmtTrigger(v: unknown, fallback = "Not triggered"): string {
  if (v == null) return fallback;
  if (typeof v === "number") return `${v}s`;
  return String(v);
}

/** Fixed-decimal string, or "---" when the value is absent. */
function fmtFixed(v: unknown, digits: number): string {
  if (v == null) return "---";
  if (typeof v === "number") return v.toFixed(digits);
  return String(v);
}

/** Scientific-notation string, or "---". */
function fmtSci(v: unknown): string {
  if (v == null) return "---";
  if (typeof v === "number") return v.toExponential();
  return String(v);
}

/** Format seconds: number -> "Xs", null -> "---". */
function fmtSeconds(v: unknown): string {
  if (v == null) return "---";
  if (typeof v === "number") return `${v}s`;
  return String(v);
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();

  const [summary, setSummary] = useState<Record<string, unknown> | null>(null);
  const [timeseries, setTimeseries] = useState<TimeseriesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    Promise.all([fetchRunSummary(runId), fetchRunTimeseries(runId)])
      .then(([sumRes, tsRes]) => {
        if (cancelled) return;
        setSummary(sumRes.summary);
        setTimeseries(tsRes);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [runId]);

  /* ---- Loading state ---- */
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <p className="text-gray-500 text-lg">Loading run {runId}...</p>
      </div>
    );
  }

  /* ---- Error state ---- */
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <p className="text-red-600 text-lg font-medium">
          Failed to load run data
        </p>
        <p className="text-gray-500 text-sm">{error}</p>
        <button
          onClick={() => navigate("/")}
          className="mt-2 rounded bg-gray-200 px-4 py-2 text-sm hover:bg-gray-300"
        >
          Back to runs
        </button>
      </div>
    );
  }

  /* ---- Shorthand access to summary fields ---- */
  const s = summary ?? {};

  return (
    <div className="space-y-8">
      {/* ---- Back button + title row ---- */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate("/")}
          className="rounded bg-gray-200 px-3 py-1.5 text-sm font-medium hover:bg-gray-300"
        >
          &larr; Back
        </button>
        <h1 className="text-xl font-bold text-gray-900 truncate">
          Run: {runId}
        </h1>
      </div>

      {/* ---- Metric cards ---- */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <MetricCard
          label="Threshold V1 Trigger"
          value={fmtTrigger(s.threshold_v1_trigger_time)}
          highlight={s.threshold_v1_trigger_time != null}
        />
        <MetricCard
          label="Integrity V1 Trigger"
          value={fmtTrigger(s.integrity_v1_trigger_time)}
          highlight={s.integrity_v1_trigger_time != null}
        />
        <MetricCard
          label="Decision Compression"
          value={fmtTrigger(s.decision_compression_window, "---")}
        />
        <MetricCard
          label="False Safe Rate"
          value={fmtFixed(s.false_safe_rate, 3)}
        />
        <MetricCard
          label="False Alert Rate"
          value={fmtFixed(s.false_alert_rate, 3)}
        />
        <MetricCard
          label="Decision Instability"
          value={fmtFixed(s.decision_instability_index, 4)}
        />
        <MetricCard
          label="Max Pc Degraded"
          value={fmtSci(s.max_pc_degraded)}
        />
        <MetricCard
          label="Max Pc Reference"
          value={fmtSci(s.max_pc_reference)}
        />
        <MetricCard
          label="Max Cov Trace"
          value={fmtSci(s.max_cov_trace)}
        />
        <MetricCard
          label="Max Staleness"
          value={fmtSeconds(s.max_staleness)}
        />
        <MetricCard
          label="Transitions/hr"
          value={fmtFixed(s.decision_transitions_per_hour, 2)}
        />
        <MetricCard
          label="Decision Entropy"
          value={fmtFixed(s.decision_entropy, 4)}
        />
        <MetricCard
          label="Mean Pc Drift"
          value={fmtSci(s.mean_pc_drift)}
        />
        <MetricCard
          label="Max Pc Drift"
          value={fmtSci(s.max_pc_drift)}
        />
        <MetricCard
          label="Staleness-Pc Corr."
          value={fmtFixed(s.staleness_pc_correlation, 3)}
        />
        <MetricCard
          label="Outage Sensitivity"
          value={fmtFixed(s.outage_sensitivity_score, 3)}
        />
        <MetricCard
          label="Mean Freshness"
          value={fmtFixed(s.mean_freshness, 3)}
        />
        <MetricCard
          label="Min Freshness"
          value={fmtFixed(s.min_freshness, 3)}
        />
        <MetricCard
          label="Total Timesteps"
          value={s.total_timesteps != null ? String(s.total_timesteps) : "---"}
        />
        <MetricCard
          label="Dynamics Model"
          value={typeof s.dynamics_model === "string" ? s.dynamics_model : "---"}
        />
        <MetricCard
          label="Seed"
          value={s.seed != null ? String(s.seed) : "---"}
        />
      </div>

      {/* ---- Download buttons ---- */}
      {runId && (
        <div className="flex gap-3">
          <a
            href={runDownloadUrl(runId, `summary_${runId}.json`)}
            download
            className="inline-flex items-center rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
          >
            Download Summary JSON
          </a>
          <a
            href={runDownloadUrl(runId, `timeseries_${runId}.parquet`)}
            download
            className="inline-flex items-center rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
          >
            Download Timeseries Parquet
          </a>
        </div>
      )}

      {/* ---- Charts ---- */}
      {timeseries && (
        <div className="space-y-6">
          <PostureTimeline
            columns={timeseries.columns}
            rows={timeseries.rows}
            summary={s}
          />
          <PcDriftPlot
            columns={timeseries.columns}
            rows={timeseries.rows}
            summary={s}
          />
          <InstabilityPlot
            columns={timeseries.columns}
            rows={timeseries.rows}
            summary={s}
          />
        </div>
      )}
    </div>
  );
}
