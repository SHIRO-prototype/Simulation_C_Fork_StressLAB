import { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { fetchRunSummary, fetchRunTimeseries, runDownloadUrl } from "../lib/api";
import type { TimeseriesResponse } from "../lib/types";
import MetricCard from "../components/MetricCard";
import PcPanel from "../components/PcPanel";
import StalenessPanel from "../components/StalenessPanel";
import UncertaintyPanel from "../components/UncertaintyPanel";
import StateBands from "../components/StateBands";
import PcDriftPlot from "../components/PcDriftPlot";
import InstabilityPlot from "../components/InstabilityPlot";
import StoryPanel from "../components/StoryPanel";
import CurrentPostureWidget from "../components/CurrentPostureWidget";
import ConfidenceGauge from "../components/ConfidenceGauge";
import CompareToControlPanel from "../components/CompareToControlPanel";

/* ------------------------------------------------------------------ */
/*  Formatting helpers                                                 */
/* ------------------------------------------------------------------ */

/** Format seconds to hh:mm string. */
function secToHM(s: number): string {
  const h = Math.floor(Math.abs(s) / 3600);
  const m = Math.floor((Math.abs(s) % 3600) / 60);
  return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}`;
}

/** Format a trigger-time value as T-hh:mm relative to TCA. */
function fmtTrigger(v: unknown, fallback = "Not triggered"): string {
  if (v == null) return fallback;
  if (typeof v === "number") {
    return `T-${secToHM(v)} (${v.toFixed(0)}s)`;
  }
  return String(v);
}

/** Fixed-decimal string, or "---" when the value is absent. */
function fmtFixed(v: unknown, digits: number): string {
  if (v == null) return "---";
  if (typeof v === "number") return v.toFixed(digits);
  return String(v);
}

/** Scientific-notation string with 3 significant figures, or "---". */
function fmtSci(v: unknown): string {
  if (v == null) return "---";
  if (typeof v === "number") return v.toPrecision(3).replace(/e\+?/, "e");
  return String(v);
}

/** Format seconds as dual display: Xs (hh:mm), or "---". */
function fmtSeconds(v: unknown): string {
  if (v == null) return "---";
  if (typeof v === "number") return `${v.toFixed(0)}s (${secToHM(v)})`;
  return String(v);
}

/** Format a compression window value. */
function fmtCompression(v: unknown): string {
  if (v == null) return "---";
  if (typeof v === "number") {
    const sign = v >= 0 ? "+" : "";
    return `${sign}${v.toFixed(0)}s (${secToHM(v)})`;
  }
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
      <div className="section-card section-pad flex min-h-[60vh] items-center justify-center">
        <p className="text-lg text-slate-500">Loading run {runId}...</p>
      </div>
    );
  }

  /* ---- Error state ---- */
  if (error) {
    return (
      <div className="section-card section-pad flex min-h-[60vh] flex-col items-center justify-center gap-4">
        <p className="text-lg font-medium text-red-600">
          Failed to load run data
        </p>
        <p className="text-sm text-slate-500">{error}</p>
        <button
          onClick={() => navigate("/")}
          className="action-button action-button-secondary mt-2"
        >
          Back to runs
        </button>
      </div>
    );
  }

  /* ---- Shorthand access to summary fields ---- */
  const s = summary ?? {};
  const sanity = s.sanity as Record<string, boolean> | undefined;

  return (
    <div className="space-y-8">
      {/* ---- Back button + title row ---- */}
      <div className="section-card section-pad flex flex-wrap items-center gap-4">
        <button
          onClick={() => navigate("/")}
          className="action-button action-button-secondary"
        >
          &larr; Back
        </button>
        <h1 className="section-heading truncate" style={{ fontSize: "2rem" }}>
          Run: {runId}
        </h1>
      </div>

      {/* ---- Sanity badges ---- */}
      {sanity && (
        <div className="flex flex-wrap gap-3">
          {[
            { key: "degradation_active", label: "Degradation Active", ok: sanity.degradation_active },
            { key: "pc_diverged", label: "Pc Diverged", ok: sanity.pc_diverged },
            { key: "staleness_ramped", label: "Staleness Ramped", ok: sanity.staleness_ramped },
          ].map((f) => (
            <span
              key={f.key}
              className={f.ok ? "status-pill status-pill-ok" : "status-pill status-pill-warn"}
            >
              {f.label}
            </span>
          ))}
        </div>
      )}

      {/* ---- Story narrative ---- */}
      <StoryPanel story={s.story as { bullets: string[] } | null | undefined} />

      {/* ---- Compare to control (D0_nominal) ---- */}
      {runId && (
        <CompareToControlPanel summary={s} runId={runId} />
      )}

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
          value={fmtCompression(s.decision_compression_window)}
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
          unit="km\u00B2"
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
          unit="nats"
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
          value={fmtSeconds(s.outage_sensitivity_score)}
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

      {/* ---- Download & export buttons ---- */}
      {runId && (
        <div className="flex flex-wrap gap-3">
          <Link
            to={`/runs/${runId}/onepager`}
            target="_blank"
            className="action-button action-button-dark"
          >
            Open One-Pager
          </Link>
          <button
            onClick={() => {
              const w = window.open(`/runs/${runId}/onepager`, "_blank");
              if (w) {
                w.addEventListener("afterprint", () => {});
                w.onload = () => setTimeout(() => w.print(), 800);
              }
            }}
            className="action-button action-button-secondary"
          >
            Print to PDF
          </button>
          <a
            href={runDownloadUrl(runId, `summary_${runId}.json`)}
            download
            className="action-button action-button-accent"
          >
            Download Summary JSON
          </a>
          <a
            href={runDownloadUrl(runId, `timeseries_${runId}.parquet`)}
            download
            className="action-button action-button-accent"
          >
            Download Timeseries Parquet
          </a>
        </div>
      )}

      {/* ---- Operator Decision Panel ---- */}
      {timeseries && (
        <div className="space-y-4">
          <CurrentPostureWidget
            columns={timeseries.columns}
            rows={timeseries.rows}
            summary={s}
          />
          <ConfidenceGauge
            columns={timeseries.columns}
            rows={timeseries.rows}
          />
        </div>
      )}

      {/* ---- Charts ---- */}
      {timeseries && (
        <div className="space-y-6">
          <PcPanel
            columns={timeseries.columns}
            rows={timeseries.rows}
            summary={s}
          />
          <StalenessPanel
            columns={timeseries.columns}
            rows={timeseries.rows}
            summary={s}
          />
          <UncertaintyPanel
            columns={timeseries.columns}
            rows={timeseries.rows}
            summary={s}
          />
          <StateBands
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
