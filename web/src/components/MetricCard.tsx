interface MetricCardProps {
  label: string;
  value: string | number | null | undefined;
  unit?: string;
  highlight?: boolean;
  className?: string;
  status?: "ok" | "warn" | "critical" | null;
}

export default function MetricCard({
  label,
  value,
  unit,
  highlight = false,
  className = "",
  status,
}: MetricCardProps) {
  const displayValue = value != null ? String(value) : "---";

  const toneClass = status
    ? `metric-card-${status}`
    : highlight
      ? "metric-card-highlight"
      : "";

  return (
    <div className={["metric-card", toneClass, className].filter(Boolean).join(" ")}>
      <p className="text-[0.72rem] font-semibold uppercase tracking-[0.18em] text-slate-500">
        {label}
      </p>
      <p className="mt-4 text-2xl font-semibold tracking-tight text-slate-900" title={displayValue}>
        {displayValue}
        {value != null && unit && (
          <span className="ml-2 text-sm font-medium text-slate-500">
            {unit}
          </span>
        )}
      </p>
    </div>
  );
}
