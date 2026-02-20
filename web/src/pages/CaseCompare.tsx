import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import Plot from "react-plotly.js";
import { fetchCase, fetchCompare, fetchRunTimeseries } from "../lib/api";
import type { TimeseriesResponse } from "../lib/types";

function colSeries(ts: TimeseriesResponse, col: string): Array<number | string | boolean | null> {
  const idx = ts.columns.indexOf(col);
  if (idx < 0) return [];
  return ts.rows.map((r) => r[idx] ?? null);
}

function numericSeries(ts: TimeseriesResponse, col: string): number[] {
  return colSeries(ts, col).map((v) => (typeof v === "number" ? v : NaN));
}

function OverlayChart({
  title,
  baseline,
  stress,
  yCol,
}: {
  title: string;
  baseline: TimeseriesResponse;
  stress: TimeseriesResponse;
  yCol: string;
}) {
  const xBase = numericSeries(baseline, "timestamp");
  const yBase = numericSeries(baseline, yCol);
  const xStress = numericSeries(stress, "timestamp");
  const yStress = numericSeries(stress, yCol);

  return (
    <div className="rounded border border-gray-200 bg-white p-3">
      <h3 className="mb-2 text-sm font-semibold text-gray-800">{title}</h3>
      <Plot
        data={[
          { x: xBase, y: yBase, type: "scatter", mode: "lines", name: "Baseline", line: { color: "#1d4ed8", width: 2 } },
          { x: xStress, y: yStress, type: "scatter", mode: "lines", name: "Stress", line: { color: "#dc2626", width: 2 } },
        ]}
        layout={{
          margin: { l: 40, r: 10, t: 10, b: 35 },
          height: 260,
          xaxis: { title: "timestamp (s)" },
          yaxis: { title: yCol },
          legend: { orientation: "h", x: 0, y: 1.15 },
        }}
        config={{ displayModeBar: false, responsive: true }}
        className="w-full"
      />
    </div>
  );
}

export default function CaseCompare() {
  const { caseId } = useParams<{ caseId: string }>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [compare, setCompare] = useState<Record<string, unknown> | null>(null);
  const [baselineTs, setBaselineTs] = useState<TimeseriesResponse | null>(null);
  const [stressTs, setStressTs] = useState<TimeseriesResponse | null>(null);

  useEffect(() => {
    if (!caseId) return;
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [caseRes, cmpRes] = await Promise.all([fetchCase(caseId), fetchCompare(caseId)]);
        const baselineRunId = caseRes.case.baseline_run_id;
        const stressRunId = caseRes.case.stress_run_id;
        if (!baselineRunId || !stressRunId) {
          throw new Error("Baseline and stress runs are required before compare.");
        }
        const [bTs, sTs] = await Promise.all([
          fetchRunTimeseries(baselineRunId, { resample: "uniform", max_points: 300 }),
          fetchRunTimeseries(stressRunId, { resample: "uniform", max_points: 300 }),
        ]);
        if (cancelled) return;
        setCompare(cmpRes.compare);
        setBaselineTs(bTs);
        setStressTs(sTs);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [caseId]);

  const deltas = useMemo(() => (compare?.deltas as Record<string, unknown>) ?? {}, [compare]);

  if (loading) return <p className="text-gray-500">Loading compare view...</p>;
  if (error) return <p className="text-red-600">{error}</p>;
  if (!baselineTs || !stressTs || !caseId) return <p className="text-gray-500">No compare data.</p>;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Case Compare: {caseId}</h1>
        <div className="flex gap-2">
          <a
            href={`/api/cases/${caseId}/onepager`}
            target="_blank"
            rel="noreferrer"
            className="rounded bg-gray-800 px-3 py-2 text-sm text-white"
          >
            Open One-Pager
          </a>
          <a href={`/api/cases/${caseId}/export`} className="rounded bg-blue-700 px-3 py-2 text-sm text-white">
            Export Evidence Pack
          </a>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {Object.entries(deltas).map(([k, v]) => (
          <div key={k} className="rounded border border-gray-200 bg-white p-3">
            <div className="text-xs text-gray-500">{k}</div>
            <div className="text-lg font-semibold text-gray-900">{v == null ? "---" : String(v)}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <OverlayChart title="Pc Overlay" baseline={baselineTs} stress={stressTs} yCol="pc_degraded" />
        <OverlayChart title="Staleness Overlay" baseline={baselineTs} stress={stressTs} yCol="staleness_obj1" />
        <OverlayChart title="Uncertainty Overlay" baseline={baselineTs} stress={stressTs} yCol="cov_trace_obj1" />
        <OverlayChart title="State Bands Overlay" baseline={baselineTs} stress={stressTs} yCol="integrity_v1_score" />
      </div>

      <div>
        <Link to="/stress-test/new" className="text-sm text-blue-700 hover:underline">
          Back to Stress Test Builder
        </Link>
      </div>
    </div>
  );
}
