import { useEffect, useState } from "react";
import { fetchSweeps, fetchSweepSummary } from "../lib/api";
import type { SweepIndex } from "../lib/types";
import SweepPlots from "../components/SweepPlots";

export default function SweepViewer() {
  const [sweeps, setSweeps] = useState<SweepIndex[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  /* ---- load sweep index on mount ---- */
  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchSweeps()
      .then((res) => setSweeps(res.sweeps))
      .catch((err: unknown) =>
        setError(err instanceof Error ? err.message : String(err)),
      )
      .finally(() => setLoading(false));
  }, []);

  /* ---- load summary when a sweep is selected ---- */
  useEffect(() => {
    if (!selectedId) {
      setSummary(null);
      return;
    }
    setSummaryLoading(true);
    setSummaryError(null);
    fetchSweepSummary(selectedId)
      .then((res) => setSummary(res.summary))
      .catch((err: unknown) =>
        setSummaryError(err instanceof Error ? err.message : String(err)),
      )
      .finally(() => setSummaryLoading(false));
  }, [selectedId]);

  /* ---- render ---- */

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <p className="text-gray-500 text-sm">Loading sweeps…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-300 bg-red-50 p-6 text-sm text-red-700">
        <p className="font-semibold">Failed to load sweeps</p>
        <p className="mt-1">{error}</p>
      </div>
    );
  }

  if (sweeps.length === 0) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-10 text-center text-gray-500">
        No sweep results found. Run{" "}
        <code className="rounded bg-gray-100 px-1.5 py-0.5 text-sm font-mono">
          stresslab sweep
        </code>{" "}
        to generate data.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-gray-900">Sweep Results</h1>

      {/* ---- sweep table ---- */}
      <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white shadow-sm">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-50">
            <tr>
              {["Sweep ID", "Parameter", "Values", "Seed", "Created"].map(
                (h) => (
                  <th
                    key={h}
                    className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wide text-gray-500"
                  >
                    {h}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sweeps.map((s) => {
              const active = s.sweep_id === selectedId;
              return (
                <tr
                  key={s.sweep_id}
                  onClick={() =>
                    setSelectedId(active ? null : s.sweep_id)
                  }
                  className={[
                    "cursor-pointer transition-colors",
                    active
                      ? "bg-blue-50"
                      : "hover:bg-gray-50",
                  ].join(" ")}
                >
                  <td className="whitespace-nowrap px-4 py-3 font-mono text-xs text-gray-800">
                    {s.sweep_id}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-gray-700">
                    {s.sweep_param ?? "—"}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-gray-700">
                    {s.n_values ?? "—"}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-gray-700">
                    {s.seed ?? "—"}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-gray-500">
                    {s.created_at ?? "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ---- selected sweep detail ---- */}
      {selectedId && (
        <div className="space-y-4">
          {summaryLoading && (
            <p className="text-sm text-gray-500">Loading summary…</p>
          )}

          {summaryError && (
            <div className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-700">
              <p className="font-semibold">Failed to load sweep summary</p>
              <p className="mt-1">{summaryError}</p>
            </div>
          )}

          {summary && !summaryLoading && (
            <SweepPlots summary={summary} />
          )}
        </div>
      )}
    </div>
  );
}
