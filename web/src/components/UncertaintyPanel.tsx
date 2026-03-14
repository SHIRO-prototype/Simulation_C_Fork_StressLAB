import Plot from "react-plotly.js";
import { createPlotLayout, plotColors, plotConfig } from "../lib/chartTheme";

interface UncertaintyPanelProps {
  columns: string[];
  rows: (number | string | null)[][];
  summary?: Record<string, unknown>;
}

function col(columns: string[], rows: (number | string | null)[][], name: string): (number | null)[] {
  const idx = columns.indexOf(name);
  if (idx === -1) return [];
  return rows.map((r) => (r[idx] != null ? Number(r[idx]) : null));
}

export default function UncertaintyPanel({ columns, rows, summary }: UncertaintyPanelProps) {
  const rawTCA = col(columns, rows, "time_to_tca");
  const hasTCA = rawTCA.length > 0 && rawTCA.some((v) => v != null);
  const xAxis = hasTCA
    ? rawTCA.map((v) => (v != null ? v / 3600 : null))
    : col(columns, rows, "timestamp");
  const xLabel = hasTCA ? "Time to TCA (hours)" : "Time (s)";

  const covTrace = col(columns, rows, "cov_trace_obj1");
  const growthRate = col(columns, rows, "cov_growth_rate_obj1");

  // Outage shading (same logic as StalenessPanel)
  const shapes: Partial<Plotly.Shape>[] = [];
  if (summary) {
    const cfg = summary.config as Record<string, unknown> | undefined;
    const outages = cfg?.outage_windows as { start: number; end: number }[] | undefined;
    if (Array.isArray(outages)) {
      const ts0 = col(columns, rows, "timestamp");
      const ttca0 = rawTCA;
      for (const ow of outages) {
        if (hasTCA && ts0.length > 0 && ttca0.length > 0 && ts0[0] != null && ttca0[0] != null) {
          const tcaEpoch = ts0[0] + ttca0[0];
          const x0 = (tcaEpoch - ow.start) / 3600;
          const x1 = (tcaEpoch - ow.end) / 3600;
          shapes.push({
            type: "rect",
            xref: "x",
            yref: "paper",
            x0: Math.max(x0, x1),
            x1: Math.min(x0, x1),
            y0: 0,
            y1: 1,
            fillcolor: "rgba(168,85,247,0.08)",
            line: { width: 1, color: "rgba(168,85,247,0.25)" },
            layer: "below",
          });
        } else {
          shapes.push({
            type: "rect",
            xref: "x",
            yref: "paper",
            x0: ow.start,
            x1: ow.end,
            y0: 0,
            y1: 1,
            fillcolor: "rgba(168,85,247,0.08)",
            line: { width: 1, color: "rgba(168,85,247,0.25)" },
            layer: "below",
          });
        }
      }
    }
  }

  const traces: Plotly.Data[] = [
    {
      x: xAxis,
      y: covTrace,
      type: "scatter",
      mode: "lines",
      name: "Cov Trace (obj1)",
      line: { color: plotColors.plum, width: 2.4 },
      yaxis: "y",
    },
  ];

  if (growthRate.length > 0 && growthRate.some((v) => v != null)) {
    traces.push({
      x: xAxis,
      y: growthRate,
      type: "scatter",
      mode: "lines",
      name: "Growth Rate",
      line: { color: plotColors.mint, width: 1.8, dash: "dot" },
      yaxis: "y2",
    });
  }

  return (
    <div className="plot-shell">
      <Plot
        data={traces}
        layout={createPlotLayout({
          title: "Covariance spread and growth pressure",
          height: 320,
          xAxis: {
            title: xLabel,
            autorange: hasTCA ? "reversed" : true,
          },
          yAxis: {
            title: "Cov Trace (km\u00B2)",
            type: "log",
            side: "left",
          },
          yAxis2: {
            title: "Growth Rate (km\u00B2/s)",
            overlaying: "y",
            side: "right",
            showgrid: false,
          },
          shapes: shapes as Plotly.Layout["shapes"],
          legend: { y: -0.24 },
          margin: { r: 72 },
        })}
        config={plotConfig}
        useResizeHandler
        style={{ width: "100%", height: "320px" }}
      />
    </div>
  );
}
