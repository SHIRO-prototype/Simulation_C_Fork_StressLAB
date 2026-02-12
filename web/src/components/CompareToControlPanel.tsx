/**
 * CompareToControlPanel — compares the current run's key metrics
 * against the D0_nominal (control) run to show degradation deltas.
 *
 * Auto-fetches the control run by searching for run_label=D0_nominal.
 * If no control run exists, the panel silently hides.
 */

import { useEffect, useState } from "react";
import { fetchRuns, fetchRunSummary } from "../lib/api";

interface CompareToControlPanelProps {
  /** Current run's summary dict */
  summary: Record<string, unknown>;
  /** Current run's ID (to avoid comparing a run against itself) */
  runId: string;
}

interface Delta {
  label: string;
  current: string;
  control: string;
  delta: string;
  direction: "better" | "worse" | "neutral";
}

const CONTROL_LABEL = "D0_nominal";

function fmtSec(v: unknown): string {
  if (v == null) return "---";
  if (typeof v !== "number") return String(v);
  const h = Math.floor(Math.abs(v) / 3600);
  const m = Math.floor((Math.abs(v) % 3600) / 60);
  return `${v.toFixed(0)}s (${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")})`;
}

function fmtSci(v: unknown): string {
  if (v == null) return "---";
  if (typeof v === "number") return v.toPrecision(3).replace(/e\+?/, "e");
  return String(v);
}

function fmtFixed(v: unknown, d: number): string {
  if (v == null) return "---";
  if (typeof v === "number") return v.toFixed(d);
  return String(v);
}

function signedDelta(v: number): string {
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(1)}`;
}

function signedDeltaSci(v: number): string {
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toPrecision(2).replace(/e\+?/, "e")}`;
}

function computeDeltas(
  current: Record<string, unknown>,
  control: Record<string, unknown>,
): Delta[] {
  const deltas: Delta[] = [];

  // Trigger time: earlier (more negative time_to_tca) is worse
  const ctrig = current.threshold_v1_trigger_time;
  const dtrig = control.threshold_v1_trigger_time;
  if (typeof ctrig === "number" && typeof dtrig === "number") {
    const diff = ctrig - dtrig;
    deltas.push({
      label: "Threshold V1 Trigger",
      current: fmtSec(ctrig),
      control: fmtSec(dtrig),
      delta: `${signedDelta(diff)}s`,
      direction: diff < -60 ? "worse" : diff > 60 ? "better" : "neutral",
    });
  }

  // Decision compression: negative means less margin (worse)
  const ccomp = current.decision_compression_window;
  const dcomp = control.decision_compression_window;
  if (typeof ccomp === "number" && typeof dcomp === "number") {
    const diff = ccomp - dcomp;
    deltas.push({
      label: "Compression Window",
      current: fmtSec(ccomp),
      control: fmtSec(dcomp),
      delta: `${signedDelta(diff)}s`,
      direction: diff < -60 ? "worse" : diff > 60 ? "better" : "neutral",
    });
  }

  // Max Pc degraded: higher is worse
  const cpc = current.max_pc_degraded;
  const dpc = control.max_pc_degraded;
  if (typeof cpc === "number" && typeof dpc === "number") {
    const diff = cpc - dpc;
    deltas.push({
      label: "Max Pc Degraded",
      current: fmtSci(cpc),
      control: fmtSci(dpc),
      delta: signedDeltaSci(diff),
      direction: diff > 0 ? "worse" : diff < 0 ? "better" : "neutral",
    });
  }

  // Max staleness: higher is worse
  const cstal = current.max_staleness;
  const dstal = control.max_staleness;
  if (typeof cstal === "number" && typeof dstal === "number") {
    const diff = cstal - dstal;
    deltas.push({
      label: "Max Staleness",
      current: fmtSec(cstal),
      control: fmtSec(dstal),
      delta: `${signedDelta(diff)}s`,
      direction: diff > 60 ? "worse" : diff < -60 ? "better" : "neutral",
    });
  }

  // Mean freshness: lower is worse
  const cfresh = current.mean_freshness;
  const dfresh = control.mean_freshness;
  if (typeof cfresh === "number" && typeof dfresh === "number") {
    const diff = cfresh - dfresh;
    deltas.push({
      label: "Mean Freshness",
      current: fmtFixed(cfresh, 3),
      control: fmtFixed(dfresh, 3),
      delta: diff >= 0 ? `+${diff.toFixed(3)}` : diff.toFixed(3),
      direction: diff < -0.01 ? "worse" : diff > 0.01 ? "better" : "neutral",
    });
  }

  // Decision instability: higher is worse
  const cinst = current.decision_instability_index;
  const dinst = control.decision_instability_index;
  if (typeof cinst === "number" && typeof dinst === "number") {
    const diff = cinst - dinst;
    deltas.push({
      label: "Decision Instability",
      current: fmtFixed(cinst, 4),
      control: fmtFixed(dinst, 4),
      delta: diff >= 0 ? `+${diff.toFixed(4)}` : diff.toFixed(4),
      direction: diff > 0.001 ? "worse" : diff < -0.001 ? "better" : "neutral",
    });
  }

  return deltas;
}

const dirColors: Record<string, string> = {
  worse: "text-red-600",
  better: "text-green-600",
  neutral: "text-gray-500",
};

export default function CompareToControlPanel({
  summary,
  runId,
}: CompareToControlPanelProps) {
  const [controlSummary, setControlSummary] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    // Skip if this run IS the control
    if (summary.run_label === CONTROL_LABEL) {
      setHidden(true);
      return;
    }

    let cancelled = false;

    fetchRuns(CONTROL_LABEL, 10)
      .then((res) => {
        if (cancelled) return;
        const ctrl = res.runs.find(
          (r) => r.run_label === CONTROL_LABEL && r.run_id !== runId,
        );
        if (!ctrl) {
          setHidden(true);
          return;
        }
        return fetchRunSummary(ctrl.run_id).then((sr) => {
          if (!cancelled) setControlSummary(sr.summary);
        });
      })
      .catch(() => {
        if (!cancelled) setHidden(true);
      });

    return () => {
      cancelled = true;
    };
  }, [runId, summary.run_label]);

  if (hidden || !controlSummary) return null;

  const deltas = computeDeltas(summary, controlSummary);
  if (deltas.length === 0) return null;

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
      <h2 className="mb-3 text-sm font-semibold text-gray-700 uppercase tracking-wide">
        Compared to Control (D0 Nominal)
      </h2>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100">
              <th className="pb-2 pr-4 text-left font-medium text-gray-500">
                Metric
              </th>
              <th className="pb-2 px-4 text-right font-medium text-gray-500">
                This Run
              </th>
              <th className="pb-2 px-4 text-right font-medium text-gray-500">
                Control
              </th>
              <th className="pb-2 pl-4 text-right font-medium text-gray-500">
                Delta
              </th>
            </tr>
          </thead>
          <tbody>
            {deltas.map((d) => (
              <tr key={d.label} className="border-b border-gray-50">
                <td className="py-2 pr-4 text-gray-800 font-medium">
                  {d.label}
                </td>
                <td className="py-2 px-4 text-right text-gray-700 font-mono text-xs">
                  {d.current}
                </td>
                <td className="py-2 px-4 text-right text-gray-500 font-mono text-xs">
                  {d.control}
                </td>
                <td
                  className={`py-2 pl-4 text-right font-mono text-xs font-semibold ${dirColors[d.direction]}`}
                >
                  {d.delta}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
