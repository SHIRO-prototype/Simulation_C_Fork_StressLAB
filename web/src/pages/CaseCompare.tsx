import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import Plot from "react-plotly.js";
import { fetchCase, fetchCompare, fetchRunTimeseries } from "../lib/api";
import { createPlotLayout, plotColors, plotConfig } from "../lib/chartTheme";
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
    <div className="plot-shell plot-shell-compact">
      <Plot
        data={[
          {
            x: xBase,
            y: yBase,
            type: "scatter",
            mode: "lines",
            name: "Baseline",
            line: { color: plotColors.baseline, width: 2.4 },
          },
          {
            x: xStress,
            y: yStress,
            type: "scatter",
            mode: "lines",
            name: "Stress",
            line: { color: plotColors.stress, width: 2.4 },
            fill: "tozeroy",
            fillcolor: "rgba(255, 143, 107, 0.08)",
          },
        ]}
        layout={createPlotLayout({
          title,
          height: 308,
          xAxis: { title: "Timestamp (s)" },
          yAxis: { title: yCol.replace(/_/g, " ") },
          legend: { y: -0.22 },
        })}
        config={plotConfig}
        useResizeHandler
        className="w-full"
        style={{ width: "100%", height: "308px" }}
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
    const currentCaseId: string = caseId;
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [caseRes, cmpRes] = await Promise.all([fetchCase(currentCaseId), fetchCompare(currentCaseId)]);
        const baselineRunId = caseRes.case.baseline_run_id;
        const stressRunId = caseRes.case.stress_run_id;
        if (typeof baselineRunId !== "string" || typeof stressRunId !== "string") {
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

  if (loading) {
    return (
      <div className="section-card section-pad">
        <p className="text-slate-500">Loading compare view...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="section-card section-pad">
        <p className="text-red-600">{error}</p>
      </div>
    );
  }

  if (!baselineTs || !stressTs || !caseId) {
    return (
      <div className="section-card section-pad">
        <p className="text-slate-500">No compare data.</p>
      </div>
    );
  }

  return (
    <div className="surface-grid">
      <section className="section-card section-card-dark section-pad">
        <div className="grid gap-8 xl:grid-cols-[1.2fr,0.8fr]">
          <div>
            <span className="section-kicker" style={{ color: "rgba(244, 238, 229, 0.72)" }}>
              Differential Analysis
            </span>
            <h1 className="page-title" style={{ color: "#fbf5ec", marginTop: 14 }}>
              Case compare for {caseId}
            </h1>
            <p className="page-subtitle" style={{ color: "rgba(244, 238, 229, 0.72)", marginTop: 18 }}>
              A clean visual overlay of the control run against the degraded case, tuned to show where uncertainty and decision timing start to separate.
            </p>
          </div>

          <div className="workflow-card">
            <span className="section-kicker">Exports</span>
            <div className="action-row mt-5">
              <a
                href={`/api/cases/${caseId}/onepager`}
                target="_blank"
                rel="noreferrer"
                className="action-button action-button-dark"
              >
                Open one-pager
              </a>
              <a
                href={`/api/cases/${caseId}/export`}
                className="action-button action-button-accent"
              >
                Export evidence pack
              </a>
            </div>
          </div>
        </div>
      </section>

      <section className="section-card section-pad">
        <span className="section-kicker">Metric deltas</span>
        <h2 className="section-heading">What changed under stress</h2>
        <div className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {Object.entries(deltas).map(([k, v]) => (
            <div key={k} className="compare-metric">
              <div className="compare-metric-label">{k.replace(/_/g, " ")}</div>
              <div className="compare-metric-value">{v == null ? "---" : String(v)}</div>
            </div>
          ))}
        </div>
      </section>

      <div className="grid gap-5 xl:grid-cols-2">
        <OverlayChart title="Collision probability overlay" baseline={baselineTs} stress={stressTs} yCol="pc_degraded" />
        <OverlayChart title="Tracking staleness overlay" baseline={baselineTs} stress={stressTs} yCol="staleness_obj1" />
        <OverlayChart title="Covariance uncertainty overlay" baseline={baselineTs} stress={stressTs} yCol="cov_trace_obj1" />
        <OverlayChart title="Integrity score overlay" baseline={baselineTs} stress={stressTs} yCol="integrity_v1_score" />
      </div>

      <div>
        <Link to="/stress-test/new" className="action-button action-button-secondary">
          Back to stress test builder
        </Link>
      </div>
    </div>
  );
}
