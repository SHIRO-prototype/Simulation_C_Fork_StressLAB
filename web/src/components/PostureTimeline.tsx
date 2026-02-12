import Plot from "react-plotly.js";

interface PostureTimelineProps {
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

export default function PostureTimeline({
  columns,
  rows,
}: PostureTimelineProps) {
  const timestamp = extractColumn(columns, rows, "timestamp") as number[];
  const pcReference = extractColumn(
    columns,
    rows,
    "pc_reference",
  ) as number[];
  const pcDegraded = extractColumn(columns, rows, "pc_degraded") as number[];
  const staleness = extractColumn(
    columns,
    rows,
    "staleness_obj1",
  ) as number[];

  return (
    <Plot
      data={[
        {
          x: timestamp,
          y: pcReference,
          type: "scatter",
          mode: "lines",
          name: "Pc Reference",
          line: { color: "#3b82f6" },
          yaxis: "y",
        },
        {
          x: timestamp,
          y: pcDegraded,
          type: "scatter",
          mode: "lines",
          name: "Pc Degraded",
          line: { color: "#ef4444" },
          yaxis: "y",
        },
        {
          x: timestamp,
          y: staleness,
          type: "scatter",
          mode: "lines",
          name: "Staleness",
          line: { color: "#f97316", dash: "dash" },
          yaxis: "y2",
        },
      ]}
      layout={{
        title: "Posture Timeline Overlay",
        height: 400,
        xaxis: {
          title: "Time (s)",
        },
        yaxis: {
          title: "Pc (log scale)",
          type: "log",
          side: "left",
        },
        yaxis2: {
          title: "Staleness",
          overlaying: "y",
          side: "right",
        },
        legend: {
          orientation: "h",
          y: -0.2,
        },
        margin: { t: 40, b: 60, l: 60, r: 60 },
      }}
      useResizeHandler
      style={{ width: "100%", height: "400px" }}
    />
  );
}
