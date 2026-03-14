import Plot from "react-plotly.js";
import { createPlotLayout, plotColors, plotConfig } from "../lib/chartTheme";

interface StalenessPanelProps {
  columns: string[];
  rows: (number | string | null)[][];
  summary?: Record<string, unknown>;
}

function col(columns: string[], rows: (number | string | null)[][], name: string): (number | null)[] {
  const idx = columns.indexOf(name);
  if (idx === -1) return [];
  return rows.map((r) => (r[idx] != null ? Number(r[idx]) : null));
}

export default function StalenessPanel({ columns, rows, summary }: StalenessPanelProps) {
  const rawTCA = col(columns, rows, "time_to_tca");
  const hasTCA = rawTCA.length > 0 && rawTCA.some((v) => v != null);
  const xAxis = hasTCA
    ? rawTCA.map((v) => (v != null ? v / 3600 : null))
    : col(columns, rows, "timestamp");
  const xLabel = hasTCA ? "Time to TCA (hours)" : "Time (s)";

  // Convert staleness to hours
  const staleness = col(columns, rows, "staleness_obj1").map((v) =>
    v != null ? v / 3600 : null,
  );

  // Outage window shading from summary.config.outage_windows
  const shapes: Partial<Plotly.Shape>[] = [];
  if (summary) {
    const cfg = summary.config as Record<string, unknown> | undefined;
    const outages = cfg?.outage_windows as { start: number; end: number }[] | undefined;
    if (Array.isArray(outages)) {
      for (const ow of outages) {
        // Convert to x-axis units
        if (hasTCA) {
          // time_to_tca = TCA_epoch - t, outage start/end are epoch-relative
          // We need to find what time_to_tca corresponds to those epoch times.
          // Since time_to_tca decreases linearly, and we use hours, we can
          // compute from the timeseries relationship: first timestamp maps to first time_to_tca.
          // Simpler approach: TCA epoch = timestamp[0] + time_to_tca[0]
          const ts0 = col(columns, rows, "timestamp");
          const ttca0 = rawTCA;
          if (ts0.length > 0 && ttca0.length > 0 && ts0[0] != null && ttca0[0] != null) {
            const tcaEpoch = ts0[0] + ttca0[0];
            const x0 = (tcaEpoch - ow.start) / 3600;
            const x1 = (tcaEpoch - ow.end) / 3600;
            shapes.push({
              type: "rect",
              xref: "x",
              yref: "paper",
              x0: Math.max(x0, x1), // reversed axis: larger value first
              x1: Math.min(x0, x1),
              y0: 0,
              y1: 1,
              fillcolor: "rgba(239,68,68,0.10)",
              line: { width: 1, color: "rgba(239,68,68,0.3)" },
              layer: "below",
            });
          }
        } else {
          shapes.push({
            type: "rect",
            xref: "x",
            yref: "paper",
            x0: ow.start,
            x1: ow.end,
            y0: 0,
            y1: 1,
            fillcolor: "rgba(239,68,68,0.10)",
            line: { width: 1, color: "rgba(239,68,68,0.3)" },
            layer: "below",
          });
        }
      }
    }
  }

  return (
    <div className="plot-shell">
      <Plot
        data={[
          {
            x: xAxis,
            y: staleness,
            type: "scatter",
            mode: "lines",
            name: "Staleness (obj1)",
            line: { shape: "hv", color: plotColors.amber, width: 2.5 },
            fill: "tozeroy",
            fillcolor: "rgba(245, 184, 92, 0.12)",
          },
        ]}
        layout={createPlotLayout({
          title: "Tracking staleness accumulation",
          height: 300,
          xAxis: {
            title: xLabel,
            autorange: hasTCA ? "reversed" : true,
          },
          yAxis: {
            title: "Staleness (hours)",
          },
          shapes: shapes as Plotly.Layout["shapes"],
          legend: { y: -0.27 },
        })}
        config={plotConfig}
        useResizeHandler
        style={{ width: "100%", height: "300px" }}
      />
    </div>
  );
}
