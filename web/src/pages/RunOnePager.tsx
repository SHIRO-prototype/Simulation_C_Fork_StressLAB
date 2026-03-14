/**
 * RunOnePager — print-ready single-page summary of a simulation run.
 *
 * Route: /runs/:runId/onepager
 * Renders OUTSIDE the Layout shell for clean printing / PDF export.
 *
 * Sections:
 *   1. Header: run_label, run_id, created_at, version strings
 *   2. Operator snapshot: CurrentPostureWidget + ConfidenceGauge
 *   3. Four chart panels: Pc, Staleness, Uncertainty, StateBands
 *   4. Story narrative
 *   5. Model KPI table (all summary metrics)
 *   6. Footer: reproduction CLI command
 */

import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { fetchRunSummary, fetchRunTimeseries } from "../lib/api";
import type { TimeseriesResponse } from "../lib/types";
import CurrentPostureWidget from "../components/CurrentPostureWidget";
import ConfidenceGauge from "../components/ConfidenceGauge";
import PcPanel from "../components/PcPanel";
import StalenessPanel from "../components/StalenessPanel";
import UncertaintyPanel from "../components/UncertaintyPanel";
import StateBands from "../components/StateBands";
import StoryPanel from "../components/StoryPanel";

/* ------------------------------------------------------------------ */
/*  Formatting helpers                                                 */
/* ------------------------------------------------------------------ */

function fmtTrigger(v: unknown): string {
  if (v == null) return "Not triggered";
  if (typeof v === "number") {
    const h = Math.floor(Math.abs(v) / 3600);
    const m = Math.floor((Math.abs(v) % 3600) / 60);
    return `T-${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")} (${v.toFixed(0)}s)`;
  }
  return String(v);
}

function fmtFixed(v: unknown, d: number): string {
  if (v == null) return "---";
  if (typeof v === "number") return v.toFixed(d);
  return String(v);
}

function fmtSci(v: unknown): string {
  if (v == null) return "---";
  if (typeof v === "number") return v.toPrecision(3).replace(/e\+?/, "e");
  return String(v);
}

function fmtSeconds(v: unknown): string {
  if (v == null) return "---";
  if (typeof v === "number") {
    const h = Math.floor(Math.abs(v) / 3600);
    const m = Math.floor((Math.abs(v) % 3600) / 60);
    return `${v.toFixed(0)}s (${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")})`;
  }
  return String(v);
}

function fmtCompression(v: unknown): string {
  if (v == null) return "---";
  if (typeof v === "number") {
    const sign = v >= 0 ? "+" : "";
    const h = Math.floor(Math.abs(v) / 3600);
    const m = Math.floor((Math.abs(v) % 3600) / 60);
    return `${sign}${v.toFixed(0)}s (${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")})`;
  }
  return String(v);
}

/* ------------------------------------------------------------------ */
/*  KPI table definition                                               */
/* ------------------------------------------------------------------ */

interface KpiRow {
  label: string;
  format: (v: unknown) => string;
  key: string;
}

const KPI_ROWS: KpiRow[] = [
  { label: "Threshold V1 Trigger", key: "threshold_v1_trigger_time", format: fmtTrigger },
  { label: "Integrity V1 Trigger", key: "integrity_v1_trigger_time", format: fmtTrigger },
  { label: "Decision Compression", key: "decision_compression_window", format: fmtCompression },
  { label: "False Safe Rate", key: "false_safe_rate", format: (v) => fmtFixed(v, 3) },
  { label: "False Alert Rate", key: "false_alert_rate", format: (v) => fmtFixed(v, 3) },
  { label: "Decision Instability", key: "decision_instability_index", format: (v) => fmtFixed(v, 4) },
  { label: "Transitions/hr", key: "decision_transitions_per_hour", format: (v) => fmtFixed(v, 2) },
  { label: "Decision Entropy", key: "decision_entropy", format: (v) => `${fmtFixed(v, 4)} nats` },
  { label: "Max Pc Degraded", key: "max_pc_degraded", format: fmtSci },
  { label: "Max Pc Reference", key: "max_pc_reference", format: fmtSci },
  { label: "Mean Pc Drift", key: "mean_pc_drift", format: fmtSci },
  { label: "Max Pc Drift", key: "max_pc_drift", format: fmtSci },
  { label: "Staleness-Pc Corr.", key: "staleness_pc_correlation", format: (v) => fmtFixed(v, 3) },
  { label: "Max Cov Trace", key: "max_cov_trace", format: (v) => `${fmtSci(v)} km\u00B2` },
  { label: "Max Staleness", key: "max_staleness", format: fmtSeconds },
  { label: "Mean Freshness", key: "mean_freshness", format: (v) => fmtFixed(v, 3) },
  { label: "Min Freshness", key: "min_freshness", format: (v) => fmtFixed(v, 3) },
  { label: "Outage Sensitivity", key: "outage_sensitivity_score", format: fmtSeconds },
  { label: "Total Timesteps", key: "total_timesteps", format: (v) => (v != null ? String(v) : "---") },
  { label: "Dynamics Model", key: "dynamics_model", format: (v) => (typeof v === "string" ? v : "---") },
];

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function RunOnePager() {
  const { runId } = useParams<{ runId: string }>();

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

  /* ---- Loading ---- */
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-white">
        <p className="text-gray-500 text-lg">Loading one-pager for {runId}...</p>
      </div>
    );
  }

  /* ---- Error ---- */
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-white gap-4">
        <p className="text-red-600 text-lg font-medium">Failed to load run data</p>
        <p className="text-gray-500 text-sm">{error}</p>
      </div>
    );
  }

  const s = summary ?? {};
  const sanity = s.sanity as Record<string, boolean> | undefined;
  const label = typeof s.run_label === "string" ? s.run_label : null;
  const title = typeof s.scenario_title === "string" ? s.scenario_title : null;
  const purpose = typeof s.scenario_purpose === "string" ? s.scenario_purpose : null;
  const takeaway = typeof s.scenario_takeaway === "string" ? s.scenario_takeaway : null;
  const seed = s.seed != null ? String(s.seed) : "---";

  // Build reproduction command
  const configHint = label ? `configs/demos/${label}.yaml` : "your_config.yaml";
  const reproCmd = `stresslab run --config ${configHint} --output-dir outputs --seed ${seed}`;

  return (
    <div className="bg-white min-h-screen">
      {/* Print styles */}
      <style>{`
        @media print {
          body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
          .no-print { display: none !important; }
          .page-break { page-break-before: always; }
        }
      `}</style>

      <div className="max-w-5xl mx-auto px-6 py-8 space-y-6">
        {/* ---- Print button (hidden when printing) ---- */}
        <div className="no-print flex gap-3">
          <button
            onClick={() => window.history.back()}
            className="rounded bg-gray-200 px-4 py-2 text-sm font-medium hover:bg-gray-300"
          >
            &larr; Back
          </button>
          <button
            onClick={() => window.print()}
            className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
          >
            Print / Save PDF
          </button>
        </div>

        {/* ==== HEADER ==== */}
        <div className="border-b-2 border-gray-800 pb-4">
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                StressLAB Run Report
              </h1>
              {title && (
                <p className="text-lg text-gray-700 mt-1">{title}</p>
              )}
              {label && (
                <p className="text-sm font-mono text-gray-500 mt-0.5">
                  {label}
                </p>
              )}
            </div>
            <div className="text-right text-xs text-gray-500 space-y-0.5">
              <p>
                <span className="font-medium">Run ID:</span>{" "}
                <span className="font-mono">{runId}</span>
              </p>
              <p>
                <span className="font-medium">Seed:</span> {seed}
              </p>
              <p>
                <span className="font-medium">Schema:</span>{" "}
                {typeof s.schema_version === "string" ? s.schema_version : "---"}
              </p>
              <p>
                <span className="font-medium">Version:</span>{" "}
                {typeof s.stresslab_version === "string"
                  ? s.stresslab_version
                  : "---"}
              </p>
            </div>
          </div>

          {/* Purpose + takeaway */}
          {(purpose || takeaway) && (
            <div className="mt-3 grid grid-cols-2 gap-4 text-sm text-gray-600">
              {purpose && (
                <div>
                  <span className="font-medium text-gray-700">Purpose: </span>
                  {purpose}
                </div>
              )}
              {takeaway && (
                <div>
                  <span className="font-medium text-gray-700">Takeaway: </span>
                  {takeaway}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ==== SANITY FLAGS ==== */}
        {sanity && (
          <div className="flex flex-wrap gap-2">
            {[
              { key: "degradation_active", label: "Degradation Active", ok: sanity.degradation_active },
              { key: "pc_diverged", label: "Pc Diverged", ok: sanity.pc_diverged },
              { key: "staleness_ramped", label: "Staleness Ramped", ok: sanity.staleness_ramped },
            ].map((f) => (
              <span
                key={f.key}
                className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${
                  f.ok
                    ? "bg-green-100 text-green-800"
                    : "bg-gray-100 text-gray-500"
                }`}
              >
                {f.ok ? "\u2713" : "\u2717"} {f.label}
              </span>
            ))}
          </div>
        )}

        {/* ==== OPERATOR SNAPSHOT ==== */}
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

        {/* ==== STORY NARRATIVE ==== */}
        <StoryPanel story={s.story as { bullets: string[] } | null | undefined} />

        {/* ==== FOUR CHART PANELS ==== */}
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
          </div>
        )}

        {/* ==== KPI TABLE ==== */}
        <div className="page-break">
          <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
            Model KPIs
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm border border-gray-200">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-3 py-2 text-left font-medium text-gray-600 border-b border-gray-200">
                    Metric
                  </th>
                  <th className="px-3 py-2 text-right font-medium text-gray-600 border-b border-gray-200">
                    Value
                  </th>
                </tr>
              </thead>
              <tbody>
                {KPI_ROWS.map((row, idx) => (
                  <tr
                    key={row.key}
                    className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}
                  >
                    <td className="px-3 py-1.5 text-gray-800 border-b border-gray-100">
                      {row.label}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-xs text-gray-700 border-b border-gray-100">
                      {row.format(s[row.key])}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* ==== FOOTER ==== */}
        <div className="border-t border-gray-300 pt-4 text-xs text-gray-500 space-y-1">
          <p className="font-medium">Reproduction Command:</p>
          <code className="block bg-gray-100 rounded px-3 py-2 text-[11px] break-all">
            {reproCmd}
          </code>
          <p className="mt-2">
            Generated by StressLAB{" "}
            {typeof s.stresslab_version === "string" ? `v${s.stresslab_version}` : ""}{" "}
            | Schema {typeof s.schema_version === "string" ? s.schema_version : "---"}{" "}
            | Metrics Contract{" "}
            {typeof s.metrics_contract_version === "string"
              ? s.metrics_contract_version
              : "---"}
          </p>
        </div>
      </div>
    </div>
  );
}
