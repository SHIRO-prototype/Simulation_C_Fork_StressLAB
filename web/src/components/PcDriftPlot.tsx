import { useState } from "react";
import Plot from "react-plotly.js";
import { createPlotLayout, plotConfig } from "../lib/chartTheme";

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
      <div className="mb-2 flex justify-end">
        <label className="field-chip cursor-pointer text-xs">
          <input
            type="checkbox"
            checked={logScale}
            onChange={(e) => setLogScale(e.target.checked)}
          />
          Log Y-axis
        </label>
      </div>
      <div className="plot-shell">
        <Plot
          data={[
            {
              x: staleness,
              y: pcDrift,
              type: "scatter",
              mode: "markers",
              marker: {
                color: colorValues as number[],
                colorscale: [
                  [0, "#4cc6d8"],
                  [0.5, "#f5b85c"],
                  [1, "#dc7c4c"],
                ],
                colorbar: {
                  title: { text: colorLabel },
                },
                size: 6,
                opacity: 0.78,
                line: { color: "rgba(8, 21, 33, 0.35)", width: 1 },
              },
              name: "Pc Drift",
            },
          ]}
          layout={createPlotLayout({
            title: "Pc drift versus staleness",
            height: 400,
            xAxis: {
              title: "Staleness (obj1)",
            },
            yAxis: {
              title: "Pc Drift",
              type: logScale ? "log" : "linear",
            },
            annotations: annotations as Plotly.Layout["annotations"],
            margin: { r: 72 },
          })}
          config={plotConfig}
          useResizeHandler
          style={{ width: "100%", height: "400px" }}
        />
      </div>
    </div>
  );
}
