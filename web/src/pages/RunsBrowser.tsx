import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { fetchRuns } from "../lib/api";
import type { RunIndex } from "../lib/types";

// ---------------------------------------------------------------------------
// Column definitions
// ---------------------------------------------------------------------------

const col = createColumnHelper<RunIndex>();

const columns = [
  col.accessor("run_id", {
    header: "Run ID",
    cell: (info) => (
      <span className="font-mono text-sm">
        {info.getValue().slice(0, 8)}
      </span>
    ),
  }),
  col.accessor("seed", {
    header: "Seed",
    cell: (info) => info.getValue() ?? "---",
  }),
  col.accessor("dynamics_model", {
    header: "Dynamics",
    cell: (info) => info.getValue() ?? "---",
  }),
  col.accessor("created_at", {
    header: "Created",
    cell: (info) => {
      const v = info.getValue();
      if (!v) return "---";
      return new Date(v).toLocaleString();
    },
  }),
  col.accessor("threshold_v1_trigger_time", {
    header: "T-v1 Trigger",
    cell: (info) => {
      const v = info.getValue();
      return v != null ? `${v}s` : "---";
    },
  }),
  col.accessor("integrity_v1_trigger_time", {
    header: "I-v1 Trigger",
    cell: (info) => {
      const v = info.getValue();
      return v != null ? `${v}s` : "---";
    },
  }),
  col.accessor("decision_compression_window", {
    header: "DCW",
    cell: (info) => {
      const v = info.getValue();
      return v != null ? `${v}s` : "---";
    },
  }),
  col.accessor("false_safe_rate", {
    header: "False-Safe",
    cell: (info) => {
      const v = info.getValue();
      return v != null ? v.toFixed(3) : "---";
    },
  }),
  col.accessor("false_alert_rate", {
    header: "False-Alert",
    cell: (info) => {
      const v = info.getValue();
      return v != null ? v.toFixed(3) : "---";
    },
  }),
  col.accessor("max_pc_degraded", {
    header: "Max Pc Deg",
    cell: (info) => {
      const v = info.getValue();
      return v != null ? v.toExponential(2) : "---";
    },
  }),
  col.accessor("decision_instability_index", {
    header: "Instability",
    cell: (info) => {
      const v = info.getValue();
      return v != null ? v.toFixed(4) : "---";
    },
  }),
  col.accessor("total_timesteps", {
    header: "Steps",
    cell: (info) => info.getValue() ?? "---",
  }),
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function RunsBrowser() {
  const navigate = useNavigate();

  const [runs, setRuns] = useState<RunIndex[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [sorting, setSorting] = useState<SortingState>([]);

  // Debounced fetch: fires 300 ms after the user stops typing (or on mount).
  useEffect(() => {
    let cancelled = false;

    const timer = setTimeout(() => {
      setLoading(true);
      setError(null);

      fetchRuns(query || undefined)
        .then((res) => {
          if (!cancelled) {
            setRuns(res.runs);
          }
        })
        .catch((err: unknown) => {
          if (!cancelled) {
            setError(err instanceof Error ? err.message : String(err));
          }
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    }, query === "" ? 0 : 300);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [query]);

  // Table instance
  const table = useReactTable({
    data: runs,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  return (
    <div className="space-y-4">
      {/* Search bar */}
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search runs…"
        className="w-full rounded border border-gray-300 bg-white px-4 py-2 text-sm
                   shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1
                   focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-800
                   dark:text-gray-100"
      />

      {/* Loading */}
      {loading && (
        <p className="py-8 text-center text-gray-500">Loading runs…</p>
      )}

      {/* Error */}
      {!loading && error && (
        <p className="py-8 text-center text-red-600">{error}</p>
      )}

      {/* Empty */}
      {!loading && !error && runs.length === 0 && (
        <p className="py-8 text-center text-gray-500">No runs found</p>
      )}

      {/* Table */}
      {!loading && !error && runs.length > 0 && (
        <div className="overflow-x-auto rounded border border-gray-200 dark:border-gray-700">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-gray-100 text-xs uppercase tracking-wider text-gray-600 dark:bg-gray-700 dark:text-gray-300">
              {table.getHeaderGroups().map((hg) => (
                <tr key={hg.id}>
                  {hg.headers.map((header) => (
                    <th
                      key={header.id}
                      className="cursor-pointer select-none px-4 py-3"
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      <span className="flex items-center gap-1">
                        {flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                        {{
                          asc: " ▲",
                          desc: " ▼",
                        }[header.column.getIsSorted() as string] ?? null}
                      </span>
                    </th>
                  ))}
                </tr>
              ))}
            </thead>

            <tbody>
              {table.getRowModel().rows.map((row, idx) => (
                <tr
                  key={row.id}
                  onClick={() => navigate(`/runs/${row.original.run_id}`)}
                  className={`cursor-pointer border-t border-gray-200 transition-colors
                    hover:bg-blue-50 dark:border-gray-700 dark:hover:bg-gray-600
                    ${idx % 2 === 0 ? "bg-white dark:bg-gray-800" : "bg-gray-50 dark:bg-gray-750"}`}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="whitespace-nowrap px-4 py-2">
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
