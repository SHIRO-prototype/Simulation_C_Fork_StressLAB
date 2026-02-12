import { useState } from "react";
import Plot from "react-plotly.js";

interface PcDriftPlotProps {
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

export default function PcDriftPlot({ columns, rows, summary }: PcDriftPlotProps) {
  const [logScale, setLogScale] = useState(false);

  const staleness = extractColumn(columns, rows, "staleness_obj1") as number[];
  const pcDrift = extractColumn(columns, rows, "pc_drift") as number[];

  // Use time_to_tca (hours) for color coding
  const rawTimeTCA = extractColumn(columns, rows, "time_to_tca") as number[];
  const hasTimeTCA = rawTimeTCA.length > 0 && rawTimeTCA.some((v) => v != null);
  const colorValues = hasTimeTCA
    ? rawTimeTCA.map((v) => (v != null ? v / 3600 : null))
    : (extractColumn(columns, rows, "timestamp") as number[]);
  const colorLabel = hasTimeTCA ? "Time to TCA (hrs)" : "Time (s)";

  // Staleness-Pc correlation annotation from summary
  const annotations: Partial<Plotly.Annotations>[] = [];
  if (summary) {
    const corr = summary.staleness_pc_correlation;
    if (typeof corr === "number") {
      annotations.push({
        x: 0.98,
        y: 0.98,
        xref: "paper",
        yref: "paper",
        text: `Staleness-Pc Corr: ${corr.toFixed(3)}`,
        showarrow: false,
        font: { size: 11, color: "#1e40af" },
        bgcolor: "rgba(255,255,255,0.9)",
        bordercolor: "#3b82f6",
        borderwidth: 1,
        borderpad: 4,
        xanchor: "right",
      });
    }
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "0.25rem" }}>
        <label style={{ fontSize: "0.75rem", color: "#6b7280", cursor: "pointer", display: "flex", alignItems: "center", gap: "0.25rem" }}>
          <input
            type="checkbox"
            checked={logScale}
            onChange={(e) => setLogScale(e.target.checked)}
          />
          Log Y-axis
        </label>
      </div>
      <Plot
        data={[
          {
            x: staleness,
            y: pcDrift,
            type: "scatter",
            mode: "markers",
            marker: {
              color: colorValues as number[],
              colorscale: "Viridis",
              colorbar: {
                title: { text: colorLabel },
              },
              size: 5,
              opacity: 0.7,
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
            type: logScale ? "log" : "linear",
          },
          annotations: annotations as Plotly.Layout["annotations"],
          margin: { t: 40, b: 60, l: 60, r: 60 },
        }}
        useResizeHandler
        style={{ width: "100%", height: "400px" }}
      />
    </div>
  );
}
