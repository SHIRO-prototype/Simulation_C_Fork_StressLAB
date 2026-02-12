import Plot from "react-plotly.js";

interface PcPanelProps {
  columns: string[];
  rows: (number | string | null)[][];
  summary?: Record<string, unknown>;
}

function col(columns: string[], rows: (number | string | null)[][], name: string): (number | null)[] {
  const idx = columns.indexOf(name);
  if (idx === -1) return [];
  return rows.map((r) => (r[idx] != null ? Number(r[idx]) : null));
}

const PC_FLOOR = 1e-12;

export default function PcPanel({ columns, rows, summary }: PcPanelProps) {
  const rawTCA = col(columns, rows, "time_to_tca");
  const hasTCA = rawTCA.length > 0 && rawTCA.some((v) => v != null);
  const xAxis = hasTCA
    ? rawTCA.map((v) => (v != null ? v / 3600 : null))
    : col(columns, rows, "timestamp");
  const xLabel = hasTCA ? "Time to TCA (hours)" : "Time (s)";

  // Display-only floor clipping (stored data unaffected)
  const pcRef = col(columns, rows, "pc_reference").map((v) =>
    v != null ? Math.max(v, PC_FLOOR) : null,
  );
  const pcDeg = col(columns, rows, "pc_degraded").map((v) =>
    v != null ? Math.max(v, PC_FLOOR) : null,
  );

  // Pc threshold horizontal line
  const shapes: Partial<Plotly.Shape>[] = [];
  const annotations: Partial<Plotly.Annotations>[] = [];

  if (summary) {
    const cfg = summary.config as Record<string, unknown> | undefined;
    const pcThresh = cfg?.pc_threshold;
    if (typeof pcThresh === "number") {
      shapes.push({
        type: "line",
        xref: "paper",
        yref: "y",
        x0: 0,
        x1: 1,
        y0: Math.log10(pcThresh),
        y1: Math.log10(pcThresh),
        line: { color: "#dc2626", width: 1.5, dash: "dashdot" },
      });
      annotations.push({
        x: 0.02,
        y: Math.log10(pcThresh),
        xref: "paper",
        yref: "y",
        text: `Pc threshold: ${pcThresh.toPrecision(2)}`,
        showarrow: false,
        font: { size: 10, color: "#dc2626" },
        xanchor: "left",
        yanchor: "bottom",
      });
    }

    // Trigger markers
    const tv1 = summary.threshold_v1_trigger_time;
    if (typeof tv1 === "number") {
      const x = hasTCA ? tv1 / 3600 : tv1;
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: x,
        x1: x,
        y0: 0,
        y1: 1,
        line: { color: "#8b5cf6", width: 1.5, dash: "dot" },
      });
      annotations.push({
        x,
        y: 1.04,
        xref: "x",
        yref: "paper",
        text: "T-v1",
        showarrow: false,
        font: { size: 9, color: "#8b5cf6" },
      });
    }
    const iv1 = summary.integrity_v1_trigger_time;
    if (typeof iv1 === "number") {
      const x = hasTCA ? iv1 / 3600 : iv1;
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: x,
        x1: x,
        y0: 0,
        y1: 1,
        line: { color: "#06b6d4", width: 1.5, dash: "dot" },
      });
      annotations.push({
        x,
        y: 1.01,
        xref: "x",
        yref: "paper",
        text: "I-v1",
        showarrow: false,
        font: { size: 9, color: "#06b6d4" },
      });
    }
  }

  return (
    <Plot
      data={[
        {
          x: xAxis,
          y: pcRef,
          type: "scatter",
          mode: "lines",
          name: "Pc Reference",
          line: { color: "#3b82f6", width: 2 },
        },
        {
          x: xAxis,
          y: pcDeg,
          type: "scatter",
          mode: "lines",
          name: "Pc Degraded",
          line: { color: "#ef4444", width: 2 },
        },
      ]}
      layout={{
        title: { text: "Panel A: Collision Probability (Pc)" },
        height: 340,
        xaxis: {
          title: { text: xLabel },
          autorange: hasTCA ? "reversed" : true,
        },
        yaxis: {
          title: { text: "Pc (log scale)" },
          type: "log",
        },
        shapes: shapes as Plotly.Layout["shapes"],
        annotations: annotations as Plotly.Layout["annotations"],
        legend: { orientation: "h", y: -0.22 },
        margin: { t: 45, b: 55, l: 65, r: 20 },
      }}
      useResizeHandler
      style={{ width: "100%", height: "340px" }}
    />
  );
}
