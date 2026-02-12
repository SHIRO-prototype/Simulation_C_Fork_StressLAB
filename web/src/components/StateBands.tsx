import Plot from "react-plotly.js";

interface StateBandsProps {
  columns: string[];
  rows: (number | string | null)[][];
  summary?: Record<string, unknown>;
}

function colStr(columns: string[], rows: (number | string | null)[][], name: string): (string | null)[] {
  const idx = columns.indexOf(name);
  if (idx === -1) return [];
  return rows.map((r) => (r[idx] != null ? String(r[idx]) : null));
}

function colNum(columns: string[], rows: (number | string | null)[][], name: string): (number | null)[] {
  const idx = columns.indexOf(name);
  if (idx === -1) return [];
  return rows.map((r) => (r[idx] != null ? Number(r[idx]) : null));
}

const INTEGRITY_COLORS: Record<string, string> = {
  Monitor: "#22c55e",
  Watch: "#eab308",
  Warning: "#f97316",
  Critical: "#ef4444",
};

const THRESHOLD_COLORS: Record<string, string> = {
  Safe: "#22c55e",
  Alert: "#ef4444",
};

interface BandSegment {
  x0: number;
  x1: number;
  color: string;
  label: string;
}

function buildBands(
  xAxis: (number | null)[],
  states: (string | null)[],
  colorMap: Record<string, string>,
): BandSegment[] {
  const bands: BandSegment[] = [];
  if (states.length === 0 || xAxis.length === 0) return bands;

  let current = states[0];
  let start = xAxis[0] ?? 0;

  for (let i = 1; i <= states.length; i++) {
    const st = i < states.length ? states[i] : null;
    if (st !== current || i === states.length) {
      const end = xAxis[Math.min(i, xAxis.length - 1)] ?? start;
      if (current && colorMap[current]) {
        bands.push({ x0: start, x1: end, color: colorMap[current], label: current });
      }
      start = xAxis[i] ?? end;
      current = st;
    }
  }
  return bands;
}

export default function StateBands({ columns, rows }: StateBandsProps) {
  const rawTCA = colNum(columns, rows, "time_to_tca");
  const hasTCA = rawTCA.length > 0 && rawTCA.some((v) => v != null);
  const xAxis = hasTCA
    ? rawTCA.map((v) => (v != null ? v / 3600 : null))
    : colNum(columns, rows, "timestamp");
  const xLabel = hasTCA ? "Time to TCA (hours)" : "Time (s)";

  const thresholdStates = colStr(columns, rows, "threshold_v1_alert_state");
  const integrityStates = colStr(columns, rows, "integrity_v1_state");

  const thresholdBands = buildBands(xAxis, thresholdStates, THRESHOLD_COLORS);
  const integrityBands = buildBands(xAxis, integrityStates, INTEGRITY_COLORS);

  // Build bar-like traces using shapes on two subplot rows
  const shapes: Partial<Plotly.Shape>[] = [];

  // Row 1: Threshold v1 (y=0..1 in subplot y)
  for (const b of thresholdBands) {
    shapes.push({
      type: "rect",
      xref: "x",
      yref: "y",
      x0: b.x0,
      x1: b.x1,
      y0: 0,
      y1: 1,
      fillcolor: b.color,
      line: { width: 0 },
      opacity: 0.6,
    });
  }

  // Row 2: Integrity v1 (y=0..1 in subplot y2)
  for (const b of integrityBands) {
    shapes.push({
      type: "rect",
      xref: "x",
      yref: "y2",
      x0: b.x0,
      x1: b.x1,
      y0: 0,
      y1: 1,
      fillcolor: b.color,
      line: { width: 0 },
      opacity: 0.6,
    });
  }

  // Invisible traces just for legend entries
  const legendTraces: Plotly.Data[] = [
    ...Object.entries(THRESHOLD_COLORS).map(([label, color]) => ({
      x: [null],
      y: [null],
      type: "scatter" as const,
      mode: "markers" as const,
      marker: { color, size: 10, symbol: "square" },
      name: `T-v1: ${label}`,
      showlegend: true,
    })),
    ...Object.entries(INTEGRITY_COLORS).map(([label, color]) => ({
      x: [null],
      y: [null],
      type: "scatter" as const,
      mode: "markers" as const,
      marker: { color, size: 10, symbol: "square" },
      name: `I-v1: ${label}`,
      showlegend: true,
    })),
  ];

  return (
    <Plot
      data={legendTraces}
      layout={{
        title: { text: "Decision State Bands" },
        height: 180,
        xaxis: {
          title: { text: xLabel },
          autorange: hasTCA ? "reversed" : true,
        },
        yaxis: {
          title: { text: "Threshold v1" },
          range: [-0.05, 1.05],
          showticklabels: false,
          showgrid: false,
          domain: [0.55, 1],
        },
        yaxis2: {
          title: { text: "Integrity v1" },
          range: [-0.05, 1.05],
          showticklabels: false,
          showgrid: false,
          anchor: "x",
          domain: [0, 0.45],
        },
        shapes: shapes as Plotly.Layout["shapes"],
        legend: { orientation: "h", y: -0.45 },
        margin: { t: 35, b: 65, l: 65, r: 20 },
      }}
      useResizeHandler
      style={{ width: "100%", height: "180px" }}
    />
  );
}
