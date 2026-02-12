import Plot from "react-plotly.js";

interface SweepResult {
  param_value: number;
  decision_compression_window: number;
  false_safe_rate: number;
}

interface SweepPlotsProps {
  summary: Record<string, unknown>;
}

export default function SweepPlots({ summary }: SweepPlotsProps) {
  const results = (summary.results ?? []) as SweepResult[];

  if (results.length === 0) {
    return <p style={{ color: "#94a3b8" }}>No sweep results available.</p>;
  }

  const paramValues = results.map((r) => r.param_value);
  const dcw = results.map((r) => r.decision_compression_window);
  const falseSafe = results.map((r) => r.false_safe_rate);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <Plot
        data={[
          {
            x: paramValues,
            y: dcw,
            type: "scatter",
            mode: "lines+markers",
            name: "DCW",
            line: { color: "#3b82f6" },
            marker: { size: 6 },
          },
        ]}
        layout={{
          title: { text: "Outage Duration vs DCW Shift" },
          height: 350,
          xaxis: { title: { text: "Param Value" } },
          yaxis: { title: { text: "Decision Compression Window" } },
          margin: { t: 40, b: 60, l: 70, r: 20 },
        }}
        useResizeHandler
        style={{ width: "100%", height: "350px" }}
      />

      <Plot
        data={[
          {
            x: paramValues,
            y: falseSafe,
            type: "scatter",
            mode: "lines+markers",
            name: "False-Safe Rate",
            line: { color: "#ef4444" },
            marker: { size: 6 },
          },
        ]}
        layout={{
          title: { text: "Outage Duration vs False-Safe Rate" },
          height: 350,
          xaxis: { title: { text: "Param Value" } },
          yaxis: { title: { text: "False-Safe Rate" } },
          margin: { t: 40, b: 60, l: 70, r: 20 },
        }}
        useResizeHandler
        style={{ width: "100%", height: "350px" }}
      />
    </div>
  );
}
