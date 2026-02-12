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

  const statusColors: Record<string, string> = {
    ok: "border-l-4 border-green-500",
    warn: "border-l-4 border-yellow-500",
    critical: "border-l-4 border-red-500",
  };
  const borderClass = status && statusColors[status]
    ? statusColors[status]
    : highlight
    ? "border-l-4 border-blue-600"
    : "";

  return (
    <div
      className={[
        "rounded-lg bg-white shadow p-4 overflow-hidden",
        borderClass,
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <p className="text-xs font-medium text-gray-500 uppercase tracking-wide truncate">
        {label}
      </p>
      <p className="mt-1 text-xl font-semibold text-gray-900 truncate" title={displayValue}>
        {displayValue}
        {value != null && unit && (
          <span className="ml-1 text-sm font-normal text-gray-500">
            {unit}
          </span>
        )}
      </p>
    </div>
  );
}
