interface MetricCardProps {
  label: string;
  value: string | number | null | undefined;
  unit?: string;
  highlight?: boolean;
  className?: string;
}

export default function MetricCard({
  label,
  value,
  unit,
  highlight = false,
  className = "",
}: MetricCardProps) {
  const displayValue = value != null ? String(value) : "---";

  return (
    <div
      className={[
        "rounded-lg bg-white shadow p-4",
        highlight ? "border-l-4 border-blue-600" : "",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
        {label}
      </p>
      <p className="mt-1 text-2xl font-semibold text-gray-900">
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
