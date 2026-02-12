/**
 * ConfidenceGauge — visual gauge showing tracking confidence.
 *
 * Computes a normalized 0-1 confidence score from freshness and
 * covariance growth rate at the last timestep, displayed as a
 * horizontal bar with color gradient.
 */

interface ConfidenceGaugeProps {
  columns: string[];
  rows: (number | string | null)[][];
}

function col(columns: string[], name: string): number {
  return columns.indexOf(name);
}

function lastNum(
  columns: string[],
  rows: (number | string | null)[][],
  name: string,
): number | null {
  const idx = col(columns, name);
  if (idx < 0 || rows.length === 0) return null;
  const v = rows[rows.length - 1][idx];
  return typeof v === "number" ? v : null;
}

/**
 * Confidence = 0.6 * freshness + 0.4 * (1 - clamp(growthRate / growthRef, 0, 1))
 * where growthRef is 1.0 by default.
 */
function computeConfidence(freshness: number | null, growthRate: number | null): number {
  const f = freshness ?? 0;
  const g = growthRate ?? 0;
  const growthRef = 1.0;
  const growthComponent = 1 - Math.min(Math.max(Math.abs(g) / growthRef, 0), 1);
  return Math.min(Math.max(0.6 * f + 0.4 * growthComponent, 0), 1);
}

function gaugeColor(conf: number): string {
  if (conf >= 0.7) return "bg-green-500";
  if (conf >= 0.4) return "bg-yellow-500";
  return "bg-red-500";
}

function gaugeLabel(conf: number): string {
  if (conf >= 0.7) return "High";
  if (conf >= 0.4) return "Moderate";
  return "Low";
}

export default function ConfidenceGauge({ columns, rows }: ConfidenceGaugeProps) {
  if (rows.length === 0) return null;

  const freshness = lastNum(columns, rows, "freshness_score");
  const growthRate = lastNum(columns, rows, "cov_growth_rate_obj1");
  const confidence = computeConfidence(freshness, growthRate);
  const pct = (confidence * 100).toFixed(0);

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
      <h2 className="mb-3 text-sm font-semibold text-gray-700 uppercase tracking-wide">
        Tracking Confidence
      </h2>
      <div className="flex items-center gap-4">
        {/* Bar */}
        <div className="flex-1 h-5 rounded-full bg-gray-200 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${gaugeColor(confidence)}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        {/* Label */}
        <div className="flex-shrink-0 text-right" style={{ minWidth: "6rem" }}>
          <span className="text-lg font-bold text-gray-900">{pct}%</span>
          <span className="ml-1 text-xs text-gray-500">{gaugeLabel(confidence)}</span>
        </div>
      </div>
      <div className="mt-2 flex gap-4 text-xs text-gray-500">
        <span>Freshness: {freshness != null ? freshness.toFixed(3) : "---"}</span>
        <span>Growth rate: {growthRate != null ? growthRate.toPrecision(3) : "---"}</span>
      </div>
    </div>
  );
}
