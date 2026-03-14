import Plot from "react-plotly.js";
import { createPlotLayout, plotConfig } from "../lib/chartTheme";

interface SweepResult {
  param_value: number;
  run_id?: string;
  decision_compression_window?: number;
  false_safe_rate?: number;
  false_alert_rate?: number;
  max_pc_degraded?: number;
  decision_instability_index?: number;
  staleness_pc_correlation?: number;
  outage_sensitivity_score?: number;
  [key: string]: unknown;
}

interface SweepPlotsProps {
  summary: Record<string, unknown>;
}

/** Individual sweep plot config */
const SWEEP_METRICS: {
  key: keyof SweepResult;
  label: string;
  yLabel: string;
  color: string;
}[] = [
  { key: "decision_compression_window", label: "DCW Shift", yLabel: "Decision Compression Window", color: "#3b82f6" },
  { key: "false_safe_rate", label: "False-Safe Rate", yLabel: "False-Safe Rate", color: "#ef4444" },
  { key: "false_alert_rate", label: "False-Alert Rate", yLabel: "False-Alert Rate", color: "#f97316" },
  { key: "max_pc_degraded", label: "Max Pc Degraded", yLabel: "Max Pc Degraded", color: "#8b5cf6" },
  { key: "decision_instability_index", label: "Decision Instability", yLabel: "Decision Instability Index", color: "#06b6d4" },
  { key: "staleness_pc_correlation", label: "Staleness-Pc Correlation", yLabel: "Staleness-Pc Correlation", color: "#10b981" },
  { key: "outage_sensitivity_score", label: "Outage Sensitivity", yLabel: "Outage Sensitivity Score", color: "#ec4899" },
];

export default function SweepPlots({ summary }: SweepPlotsProps) {
  const results = (summary.results ?? []) as SweepResult[];
  const sweepParam = (summary.sweep_param as string) ?? "param_value";

  if (results.length === 0) {
    return <p style={{ color: "#94a3b8" }}>No sweep results available.</p>;
  }

  const paramValues = results.map((r) => r.param_value);

  // Filter to only metrics that exist in the data
  const availableMetrics = SWEEP_METRICS.filter((m) =>
    results.some((r) => r[m.key] != null),
  );

  if (availableMetrics.length === 0) {
    return <p style={{ color: "#94a3b8" }}>No recognized metric columns found in sweep results.</p>;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      {availableMetrics.map((m) => (
        <div key={m.key} className="plot-shell">
          <Plot
            data={[
              {
                x: paramValues,
                y: results.map((r) => r[m.key] as number),
                type: "scatter",
                mode: "lines+markers",
                name: m.label,
                line: { color: m.color, width: 2.4 },
                marker: { size: 7, color: m.color, line: { color: "rgba(8, 21, 33, 0.3)", width: 1 } },
                fill: "tozeroy",
                fillcolor: `${m.color}22`,
              },
            ]}
            layout={createPlotLayout({
              title: `${sweepParam} vs ${m.label}`,
              height: 350,
              xAxis: { title: sweepParam },
              yAxis: { title: m.yLabel },
            })}
            config={plotConfig}
            useResizeHandler
            style={{ width: "100%", height: "350px" }}
          />
        </div>
      ))}
    </div>
  );
}
