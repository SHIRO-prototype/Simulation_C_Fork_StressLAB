import Plot from "react-plotly.js";
import { createPlotLayout, plotColors, plotConfig } from "../lib/chartTheme";

interface InstabilityPlotProps {
  columns: string[];
  rows: (number | string | null)[][];
  summary?: Record<string, unknown>;
}

function extractColumn(
  columns: string[],
  rows: (number | string | null)[][],
  name: string,
): (number | string | null)[] {
  const idx = columns.indexOf(name);
  if (idx === -1) return [];
  return rows.map((row) => row[idx]);
}

/** Map threshold_v1_alert_state strings to ordinals */
const THRESHOLD_MAP: Record<string, number> = { Safe: 0, Alert: 1 };
const THRESHOLD_LABELS = ["Safe", "Alert"];

/** Map integrity_v1_state strings to ordinals */
const INTEGRITY_MAP: Record<string, number> = {
  Monitor: 0,
  Watch: 1,
  Warning: 2,
  Critical: 3,
};
const INTEGRITY_LABELS = ["Monitor", "Watch", "Warning", "Critical"];

export default function InstabilityPlot({
  columns,
  rows,
  summary,
}: InstabilityPlotProps) {
  // Use time_to_tca (hours) as x-axis, fallback to timestamp
  const rawTimeTCA = extractColumn(columns, rows, "time_to_tca") as number[];
  const hasTimeTCA = rawTimeTCA.length > 0 && rawTimeTCA.some((v) => v != null);
  const timeTCAHours = rawTimeTCA.map((v) =>
    v != null ? v / 3600 : null,
  ) as number[];
  const xAxis = hasTimeTCA
    ? timeTCAHours
    : (extractColumn(columns, rows, "timestamp") as number[]);
  const xLabel = hasTimeTCA ? "Time to TCA (hours)" : "Time (s)";

  // Extract raw string columns and map to ordinals
  const thresholdRaw = extractColumn(
    columns,
    rows,
    "threshold_v1_alert_state",
  ) as string[];
  const integrityRaw = extractColumn(
    columns,
    rows,
    "integrity_v1_state",
  ) as string[];

  const thresholdOrdinal = thresholdRaw.map(
    (v) => (v != null ? (THRESHOLD_MAP[v] ?? null) : null) as number | null,
  );
  const integrityOrdinal = integrityRaw.map(
    (v) => (v != null ? (INTEGRITY_MAP[v] ?? null) : null) as number | null,
  );

  // Annotation for decision_transitions_per_hour
  const annotations: Partial<Plotly.Annotations>[] = [];
  if (summary) {
    const tph = summary.decision_transitions_per_hour;
    if (typeof tph === "number") {
      annotations.push({
        x: 0.02,
        y: 0.98,
        xref: "paper",
        yref: "paper",
        text: `Transitions/hr: ${tph.toFixed(2)}`,
        showarrow: false,
        font: { size: 11, color: "#6b7280" },
        bgcolor: "rgba(255,255,255,0.85)",
        borderpad: 4,
      });
    }
  }

  return (
    <div className="plot-shell">
      <Plot
        data={[
          {
            x: xAxis,
            y: thresholdOrdinal,
            type: "scatter",
            mode: "lines",
            name: "Threshold v1 Alert State",
            line: { shape: "hv", color: plotColors.plum, width: 2.2 },
            yaxis: "y",
          },
          {
            x: xAxis,
            y: integrityOrdinal,
            type: "scatter",
            mode: "lines",
            name: "Integrity v1 State",
            line: { shape: "hv", color: plotColors.baseline, width: 2.2 },
            yaxis: "y2",
          },
        ]}
        layout={createPlotLayout({
          title: "Decision state timeline",
          height: 336,
          xAxis: {
            title: xLabel,
            autorange: hasTimeTCA ? "reversed" : true,
          },
          yAxis: {
            title: "Threshold v1",
            tickvals: [0, 1],
            ticktext: THRESHOLD_LABELS,
            range: [-0.2, 1.4],
            side: "left",
          },
          yAxis2: {
            title: "Integrity v1",
            tickvals: [0, 1, 2, 3],
            ticktext: INTEGRITY_LABELS,
            overlaying: "y",
            side: "right",
            range: [-0.3, 3.5],
            showgrid: false,
          },
          annotations: annotations as Plotly.Layout["annotations"],
          legend: { y: -0.24 },
          margin: { l: 82, r: 82 },
        })}
        config={plotConfig}
        useResizeHandler
        style={{ width: "100%", height: "336px" }}
      />
    </div>
  );
}
