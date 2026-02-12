import Plot from "react-plotly.js";

interface PcDriftPlotProps {
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

export default function PcDriftPlot({ columns, rows }: PcDriftPlotProps) {
  const staleness = extractColumn(
    columns,
    rows,
    "staleness_obj1",
  ) as number[];
  const pcDrift = extractColumn(columns, rows, "pc_drift") as number[];
  const timestamp = extractColumn(columns, rows, "timestamp") as number[];

  return (
    <Plot
      data={[
        {
          x: staleness,
          y: pcDrift,
          type: "scatter",
          mode: "markers",
          marker: {
            color: timestamp,
            colorscale: "Viridis",
            colorbar: {
              title: { text: "Time (s)" },
            },
            size: 5,
          },
          name: "Pc Drift",
        },
      ]}
      layout={{
        title: { text: "Pc Drift vs Staleness" },
        height: 400,
        xaxis: {
          title: { text: "Staleness (obj1)" },
        },
        yaxis: {
          title: { text: "Pc Drift" },
        },
        margin: { t: 40, b: 60, l: 60, r: 60 },
      }}
      useResizeHandler
      style={{ width: "100%", height: "400px" }}
    />
  );
}
