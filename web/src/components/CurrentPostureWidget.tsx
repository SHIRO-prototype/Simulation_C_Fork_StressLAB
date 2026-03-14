interface CurrentPostureWidgetProps {
  columns: string[];
  rows: (number | string | null)[][];
  summary: Record<string, unknown>;
}

function col(columns: string[], name: string): number {
  return columns.indexOf(name);
}

function lastVal(
  columns: string[],
  rows: (number | string | null)[][],
  name: string,
): number | string | null {
  const idx = col(columns, name);
  if (idx < 0 || rows.length === 0) return null;
  return rows[rows.length - 1][idx];
}

function fmtNum(v: number | string | null, digits: number): string {
  if (v == null) return "---";
  if (typeof v === "number") return v.toFixed(digits);
  return String(v);
}

function fmtSci(v: number | string | null): string {
  if (v == null) return "---";
  if (typeof v === "number") return v.toPrecision(3).replace(/e\+?/, "e");
  return String(v);
}

function fmtHours(seconds: number | string | null): string {
  if (seconds == null) return "---";
  if (typeof seconds !== "number") return String(seconds);
  const h = seconds / 3600;
  return `${h.toFixed(1)}h`;
}

function stateTone(state: string | null | undefined): string {
  if (!state) return "status-pill status-pill-warn";
  const s = String(state).toLowerCase();
  if (s === "safe" || s === "monitor") return "status-pill status-pill-ok";
  if (s === "alert" || s === "critical") return "status-pill status-pill-critical";
  return "status-pill status-pill-warn";
}

export default function CurrentPostureWidget({
  columns,
  rows,
  summary: _summary,
}: CurrentPostureWidgetProps) {
  if (rows.length === 0) return null;

  const threshState = lastVal(columns, rows, "threshold_v1_alert_state");
  const integrityState = lastVal(columns, rows, "integrity_v1_state");
  const integrityScore = lastVal(columns, rows, "integrity_v1_score");
  const timeTCA = lastVal(columns, rows, "time_to_tca");
  const staleness = lastVal(columns, rows, "staleness_obj1");
  const pcRef = lastVal(columns, rows, "pc_reference");
  const pcDeg = lastVal(columns, rows, "pc_degraded");
  const freshness = lastVal(columns, rows, "freshness_score");

  const pcRatio =
    typeof pcRef === "number" && typeof pcDeg === "number" && pcRef > 0
      ? pcDeg / pcRef
      : null;

  const items: { label: string; value: string; pill?: string }[] = [
    {
      label: "Threshold V1",
      value: threshState != null ? String(threshState) : "---",
      pill: stateTone(threshState != null ? String(threshState) : null),
    },
    {
      label: "Integrity V1",
      value: integrityState != null ? String(integrityState) : "---",
      pill: stateTone(integrityState != null ? String(integrityState) : null),
    },
    { label: "Integrity Score", value: fmtNum(integrityScore, 3) },
    { label: "Time to TCA", value: fmtHours(timeTCA) },
    { label: "Staleness", value: fmtHours(staleness) },
    { label: "Freshness", value: fmtNum(freshness, 3) },
    { label: "Pc Reference", value: fmtSci(pcRef) },
    { label: "Pc Degraded", value: fmtSci(pcDeg) },
    { label: "Pc Ratio", value: pcRatio != null ? `${pcRatio.toFixed(2)}x` : "---" },
  ];

  return (
    <div className="section-card section-card-dark section-pad">
      <span className="section-kicker" style={{ color: "rgba(244, 238, 229, 0.72)" }}>
        Live Posture
      </span>
      <h2 className="section-heading" style={{ color: "#fbf5ec", fontSize: "1.7rem" }}>
        Final-timestep operator snapshot
      </h2>
      <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {items.map((item) => (
          <div
            key={item.label}
            className="rounded-[22px] border border-white/10 bg-white/5 px-4 py-4 backdrop-blur-sm"
          >
            <p className="text-[0.72rem] uppercase tracking-[0.18em] text-white/50">
              {item.label}
            </p>
            {item.pill ? (
              <span className={`${item.pill} mt-3`}>{item.value}</span>
            ) : (
              <p className="mt-3 text-xl font-semibold text-[#fbf5ec]">{item.value}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
