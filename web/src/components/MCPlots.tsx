import Plot from "react-plotly.js";

interface MCPlotsProps {
  summary: Record<string, unknown>;
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

const STAT_KEYS = [
  "mean_dcw",
  "std_dcw",
  "mean_false_safe_rate",
  "std_false_safe_rate",
  "mean_false_alert_rate",
  "std_false_alert_rate",
  "mean_decision_instability_index",
  "n_runs",
] as const;

/** Histogram configs to render from CSV data */
const HISTOGRAMS: {
  column: string;
  label: string;
  color: string;
}[] = [
  { column: "decision_compression_window", label: "Decision Compression Window Distribution", color: "#3b82f6" },
  { column: "false_safe_rate", label: "False-Safe Rate Distribution", color: "#ef4444" },
  { column: "false_alert_rate", label: "False-Alert Rate Distribution", color: "#f97316" },
  { column: "max_pc_degraded", label: "Max Pc Degraded Distribution", color: "#8b5cf6" },
  { column: "decision_instability_index", label: "Decision Instability Distribution", color: "#06b6d4" },
  { column: "staleness_pc_correlation", label: "Staleness-Pc Correlation Distribution", color: "#10b981" },
  { column: "outage_sensitivity_score", label: "Outage Sensitivity Distribution", color: "#ec4899" },
];

export default function MCPlots({ summary }: MCPlotsProps) {
  const csvColumns = summary.csv_columns as string[] | undefined;
  const csvRows = summary.csv_rows as (number | string | null)[][] | undefined;

  const hasStats = STAT_KEYS.some((k) => k in summary);
  const hasCsv = csvColumns && csvRows && csvRows.length > 0;

  if (!hasCsv && !hasStats) {
    return (
      <p style={{ color: "#94a3b8" }}>
        Monte Carlo distribution plots will appear when batch data is available.
      </p>
    );
  }

  // Filter histograms to only those with available columns
  const availableHistograms = hasCsv
    ? HISTOGRAMS.filter((h) => csvColumns.includes(h.column))
    : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      {hasStats && (
        <div
          style={{
            background: "#1e293b",
            borderRadius: "8px",
            padding: "1rem",
            color: "#e2e8f0",
            fontFamily: "monospace",
            fontSize: "0.875rem",
          }}
        >
          <h3 style={{ margin: "0 0 0.5rem", fontSize: "1rem" }}>
            Monte Carlo Summary Statistics
          </h3>
          <table style={{ borderCollapse: "collapse", width: "100%" }}>
            <tbody>
              {STAT_KEYS.filter((k) => k in summary).map((k) => (
                <tr key={k}>
                  <td style={{ padding: "2px 12px 2px 0", color: "#94a3b8" }}>
                    {k}
                  </td>
                  <td style={{ padding: "2px 0" }}>
                    {String(summary[k])}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {typeof summary.csv_total_rows === "number" && (
            <p style={{ marginTop: "0.5rem", color: "#94a3b8", fontSize: "0.75rem" }}>
              Total runs in CSV: {summary.csv_total_rows}
            </p>
          )}
        </div>
      )}

      {availableHistograms.map((h) => (
        <Plot
          key={h.column}
          data={[
            {
              x: extractColumn(
                csvColumns!,
                csvRows!,
                h.column,
              ) as number[],
              type: "histogram",
              name: h.label,
              marker: { color: h.color },
            },
          ]}
          layout={{
            title: { text: h.label },
            height: 350,
            xaxis: { title: { text: h.column.replace(/_/g, " ") } },
            yaxis: { title: { text: "Count" } },
            margin: { t: 40, b: 60, l: 60, r: 20 },
          }}
          useResizeHandler
          style={{ width: "100%", height: "350px" }}
        />
      ))}

      {hasCsv && availableHistograms.length === 0 && (
        <p style={{ color: "#94a3b8" }}>
          No recognized metric columns found in Monte Carlo CSV data.
        </p>
      )}
    </div>
  );
}
