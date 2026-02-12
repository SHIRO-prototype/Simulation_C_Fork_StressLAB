import Plot from "react-plotly.js";

interface InstabilityPlotProps {
  columns: string[];
  rows: (number | string | null)[][];
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

export default function InstabilityPlot({
  columns,
  rows,
}: InstabilityPlotProps) {
  const timestamp = extractColumn(columns, rows, "timestamp") as number[];
  const thresholdAlert = extractColumn(
    columns,
    rows,
    "threshold_v1_alert_state",
  ) as number[];
  const integrityState = extractColumn(
    columns,
    rows,
    "integrity_v1_state",
  ) as number[];

  return (
    <Plot
      data={[
        {
          x: timestamp,
          y: thresholdAlert,
          type: "scatter",
          mode: "lines",
          name: "Threshold v1 Alert State",
          line: { shape: "hv", color: "#8b5cf6" },
        },
        {
          x: timestamp,
          y: integrityState,
          type: "scatter",
          mode: "lines",
          name: "Integrity v1 State",
          line: { shape: "hv", color: "#06b6d4" },
        },
      ]}
      layout={{
        title: "Decision State Timeline",
        height: 300,
        xaxis: {
          title: "Time (s)",
        },
        yaxis: {
          title: "State",
          dtick: 1,
        },
        legend: {
          orientation: "h",
          y: -0.25,
        },
        margin: { t: 40, b: 60, l: 60, r: 20 },
      }}
      useResizeHandler
      style={{ width: "100%", height: "300px" }}
    />
  );
}
