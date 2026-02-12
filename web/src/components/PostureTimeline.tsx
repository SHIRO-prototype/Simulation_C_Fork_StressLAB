import Plot from "react-plotly.js";

interface PostureTimelineProps {
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

/** Map integrity_v1_state strings to color bands */
const STATE_COLORS: Record<string, string> = {
  Monitor: "rgba(34,197,94,0.12)",   // green
  Watch: "rgba(234,179,8,0.15)",     // yellow
  Warning: "rgba(249,115,22,0.18)",  // orange
  Critical: "rgba(239,68,68,0.2)",   // red
  Safe: "rgba(34,197,94,0.08)",
  Alert: "rgba(239,68,68,0.15)",
};

export default function PostureTimeline({
  columns,
  rows,
  summary,
}: PostureTimelineProps) {
  // Use time_to_tca (seconds) -> convert to hours for x-axis
  const rawTimeTCA = extractColumn(columns, rows, "time_to_tca") as number[];
  const timeTCAHours = rawTimeTCA.map((v) =>
    v != null ? v / 3600 : null,
  ) as number[];

  // Fall back to timestamp if time_to_tca not available
  const hasTimeTCA = rawTimeTCA.length > 0 && rawTimeTCA.some((v) => v != null);
  const xAxis = hasTimeTCA ? timeTCAHours : (extractColumn(columns, rows, "timestamp") as number[]);
  const xLabel = hasTimeTCA ? "Time to TCA (hours)" : "Time (s)";

  const pcReference = extractColumn(columns, rows, "pc_reference") as number[];
  const pcDegraded = extractColumn(columns, rows, "pc_degraded") as number[];
  const staleness = extractColumn(columns, rows, "staleness_obj1") as number[];
  const covTrace = extractColumn(columns, rows, "cov_trace_obj1") as number[];
  const integrityStates = extractColumn(columns, rows, "integrity_v1_state") as string[];

  // Build decision state background bands (colored rectangles)
  const shapes: Partial<Plotly.Shape>[] = [];
  if (integrityStates.length > 0 && xAxis.length > 0) {
    let currentState = integrityStates[0];
    let bandStart = xAxis[0];
    for (let i = 1; i <= integrityStates.length; i++) {
      const state = i < integrityStates.length ? integrityStates[i] : null;
      if (state !== currentState || i === integrityStates.length) {
        const bandEnd = xAxis[Math.min(i, xAxis.length - 1)];
        if (currentState && STATE_COLORS[currentState]) {
          shapes.push({
            type: "rect",
            xref: "x",
            yref: "paper",
            x0: bandStart,
            x1: bandEnd,
            y0: 0,
            y1: 1,
            fillcolor: STATE_COLORS[currentState],
            line: { width: 0 },
            layer: "below",
          });
        }
        bandStart = xAxis[i] ?? bandEnd;
        currentState = state as string;
      }
    }
  }

  // Vertical trigger-time markers from summary
  const annotations: Partial<Plotly.Annotations>[] = [];
  if (summary && hasTimeTCA) {
    const tv1 = summary.threshold_v1_trigger_time;
    if (typeof tv1 === "number") {
      const tv1Hours = tv1 / 3600;
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: tv1Hours,
        x1: tv1Hours,
        y0: 0,
        y1: 1,
        line: { color: "#8b5cf6", width: 2, dash: "dot" },
      });
      annotations.push({
        x: tv1Hours,
        y: 1.05,
        xref: "x",
        yref: "paper",
        text: "T-v1 Trigger",
        showarrow: false,
        font: { size: 10, color: "#8b5cf6" },
      });
    }
    const iv1 = summary.integrity_v1_trigger_time;
    if (typeof iv1 === "number") {
      const iv1Hours = iv1 / 3600;
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: iv1Hours,
        x1: iv1Hours,
        y0: 0,
        y1: 1,
        line: { color: "#06b6d4", width: 2, dash: "dot" },
      });
      annotations.push({
        x: iv1Hours,
        y: 1.02,
        xref: "x",
        yref: "paper",
        text: "I-v1 Trigger",
        showarrow: false,
        font: { size: 10, color: "#06b6d4" },
      });
    }
  }

  // Build traces
  const traces: Plotly.Data[] = [
    {
      x: xAxis,
      y: pcReference,
      type: "scatter",
      mode: "lines",
      name: "Pc Reference",
      line: { color: "#3b82f6", width: 2 },
      yaxis: "y",
    },
    {
      x: xAxis,
      y: pcDegraded,
      type: "scatter",
      mode: "lines",
      name: "Pc Degraded",
      line: { color: "#ef4444", width: 2 },
      yaxis: "y",
    },
    {
      x: xAxis,
      y: staleness,
      type: "scatter",
      mode: "lines",
      name: "Staleness (obj1)",
      line: { color: "#f97316", dash: "dash", width: 1.5 },
      yaxis: "y2",
    },
  ];

  // Add cov_trace on secondary axis if available
  if (covTrace.length > 0 && covTrace.some((v) => v != null)) {
    traces.push({
      x: xAxis,
      y: covTrace,
      type: "scatter",
      mode: "lines",
      name: "Cov Trace (obj1)",
      line: { color: "#a855f7", dash: "dot", width: 1.5 },
      yaxis: "y2",
    });
  }

  return (
    <Plot
      data={traces}
      layout={{
        title: { text: "Posture Timeline Overlay" },
        height: 450,
        xaxis: {
          title: { text: xLabel },
          autorange: hasTimeTCA ? "reversed" : true,
        },
        yaxis: {
          title: { text: "Pc (log scale)" },
          type: "log",
          side: "left",
        },
        yaxis2: {
          title: { text: "Staleness / Cov Trace" },
          overlaying: "y",
          side: "right",
        },
        shapes: shapes as Plotly.Layout["shapes"],
        annotations: annotations as Plotly.Layout["annotations"],
        legend: {
          orientation: "h",
          y: -0.2,
        },
        margin: { t: 50, b: 60, l: 60, r: 60 },
      }}
      useResizeHandler
      style={{ width: "100%", height: "450px" }}
    />
  );
}
