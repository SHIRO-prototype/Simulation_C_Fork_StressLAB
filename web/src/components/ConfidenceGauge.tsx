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

function computeConfidence(freshness: number | null, growthRate: number | null): number {
  const f = freshness ?? 0;
  const g = growthRate ?? 0;
  const growthRef = 1.0;
  const growthComponent = 1 - Math.min(Math.max(Math.abs(g) / growthRef, 0), 1);
  return Math.min(Math.max(0.6 * f + 0.4 * growthComponent, 0), 1);
}

function gaugeLabel(conf: number): string {
  if (conf >= 0.7) return "High confidence";
  if (conf >= 0.4) return "Moderate confidence";
  return "Low confidence";
}

function gaugeGradient(conf: number): string {
  if (conf >= 0.7) return "linear-gradient(90deg, #78f0c7 0%, #4cc6d8 100%)";
  if (conf >= 0.4) return "linear-gradient(90deg, #f5b85c 0%, #dc7c4c 100%)";
  return "linear-gradient(90deg, #ff9d7a 0%, #ff7c7c 100%)";
}

export default function ConfidenceGauge({ columns, rows }: ConfidenceGaugeProps) {
  if (rows.length === 0) return null;

  const freshness = lastNum(columns, rows, "freshness_score");
  const growthRate = lastNum(columns, rows, "cov_growth_rate_obj1");
  const confidence = computeConfidence(freshness, growthRate);
  const pct = confidence * 100;

  return (
    <div className="section-card section-pad">
      <span className="section-kicker">Tracking Trust</span>
      <h2 className="section-heading" style={{ fontSize: "1.55rem" }}>
        Confidence envelope
      </h2>
      <p className="section-copy">
        A compressed signal from freshness and covariance growth, tuned to highlight whether the tracking picture still supports timely action.
      </p>

      <div className="mt-6 rounded-[24px] border border-slate-900/10 bg-white/60 p-4">
        <div className="flex items-end justify-between gap-4">
          <div>
            <p className="text-[0.72rem] uppercase tracking-[0.18em] text-slate-500">
              Current level
            </p>
            <p className="mt-2 text-3xl font-semibold text-slate-900">
              {pct.toFixed(0)}%
            </p>
          </div>
          <span className={confidence >= 0.7 ? "status-pill status-pill-ok" : confidence >= 0.4 ? "status-pill status-pill-warn" : "status-pill status-pill-critical"}>
            {gaugeLabel(confidence)}
          </span>
        </div>

        <div className="mt-5 h-5 overflow-hidden rounded-full bg-slate-950/10">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${pct}%`,
              background: gaugeGradient(confidence),
              boxShadow: "0 0 28px rgba(76, 198, 216, 0.28)",
            }}
          />
        </div>

        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <div className="meta-card">
            <div className="meta-card-label">Freshness</div>
            <div className="meta-card-value">{freshness != null ? freshness.toFixed(3) : "---"}</div>
          </div>
          <div className="meta-card">
            <div className="meta-card-label">Growth rate</div>
            <div className="meta-card-value">{growthRate != null ? growthRate.toPrecision(3) : "---"}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
