/**
 * CurrentPostureWidget — shows last-timestep operational snapshot.
 *
 * Displays the current (final) state of the conjunction scenario:
 * decision states, TCA countdown, staleness, and Pc values.
 */

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

/** State label styling. */
function stateStyle(state: string | null | undefined): string {
  if (!state) return "bg-gray-100 text-gray-600";
  const s = String(state).toLowerCase();
  if (s === "safe" || s === "monitor") return "bg-green-100 text-green-800";
  if (s === "alert" || s === "critical") return "bg-red-100 text-red-800";
  if (s === "warning") return "bg-orange-100 text-orange-800";
  if (s === "watch") return "bg-yellow-100 text-yellow-800";
  return "bg-gray-100 text-gray-600";
}

export default function CurrentPostureWidget({
  columns,
  rows,
  summary,
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

  const items: { label: string; value: string; extra?: string }[] = [
    {
      label: "Threshold V1",
      value: threshState != null ? String(threshState) : "---",
      extra: stateStyle(threshState != null ? String(threshState) : null),
    },
    {
      label: "Integrity V1",
      value: integrityState != null ? String(integrityState) : "---",
      extra: stateStyle(integrityState != null ? String(integrityState) : null),
    },
    { label: "Integrity Score", value: fmtNum(integrityScore, 3) },
    { label: "Time to TCA", value: fmtHours(timeTCA) },
    { label: "Staleness", value: fmtHours(staleness) },
    { label: "Freshness", value: fmtNum(freshness, 3) },
    { label: "Pc Reference", value: fmtSci(pcRef) },
    { label: "Pc Degraded", value: fmtSci(pcDeg) },
    { label: "Pc Ratio", value: pcRatio != null ? pcRatio.toFixed(2) + "x" : "---" },
  ];

  const _ = summary; // consumed for type compat; data comes from last row

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
      <h2 className="mb-3 text-sm font-semibold text-gray-700 uppercase tracking-wide">
        Current Posture (Last Timestep)
      </h2>
      <div className="grid grid-cols-3 gap-3 sm:grid-cols-5 lg:grid-cols-9">
        {items.map((item) => (
          <div key={item.label} className="text-center">
            <p className="text-xs text-gray-500 truncate" title={item.label}>
              {item.label}
            </p>
            {item.extra ? (
              <span
                className={`mt-1 inline-block rounded-full px-2 py-0.5 text-xs font-semibold ${item.extra}`}
              >
                {item.value}
              </span>
            ) : (
              <p className="mt-1 text-sm font-semibold text-gray-900 truncate" title={item.value}>
                {item.value}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
