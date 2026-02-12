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
  "n_runs",
] as const;

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
        </div>
      )}

      {hasCsv && csvColumns.includes("decision_compression_window") && (
        <Plot
          data={[
            {
              x: extractColumn(
                csvColumns,
                csvRows,
                "decision_compression_window",
              ) as number[],
              type: "histogram",
              name: "DCW Distribution",
              marker: { color: "#3b82f6" },
            },
          ]}
          layout={{
            title: "Decision Compression Window Distribution",
            height: 350,
            xaxis: { title: "Decision Compression Window" },
            yaxis: { title: "Count" },
            margin: { t: 40, b: 60, l: 60, r: 20 },
          }}
          useResizeHandler
          style={{ width: "100%", height: "350px" }}
        />
      )}

      {hasCsv && csvColumns.includes("false_safe_rate") && (
        <Plot
          data={[
            {
              x: extractColumn(
                csvColumns,
                csvRows,
                "false_safe_rate",
              ) as number[],
              type: "histogram",
              name: "False-Safe Rate Distribution",
              marker: { color: "#ef4444" },
            },
          ]}
          layout={{
            title: "False-Safe Rate Distribution",
            height: 350,
            xaxis: { title: "False-Safe Rate" },
            yaxis: { title: "Count" },
            margin: { t: 40, b: 60, l: 60, r: 20 },
          }}
          useResizeHandler
          style={{ width: "100%", height: "350px" }}
        />
      )}

      {hasCsv &&
        !csvColumns.includes("decision_compression_window") &&
        !csvColumns.includes("false_safe_rate") && (
          <p style={{ color: "#94a3b8" }}>
            Monte Carlo distribution plots will appear when batch data is
            available.
          </p>
        )}
    </div>
  );
}
