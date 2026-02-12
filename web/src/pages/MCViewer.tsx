import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { fetchMCBatches, fetchMCSummary } from "../lib/api";
import type { MCBatchIndex } from "../lib/types";
import MCPlots from "../components/MCPlots";

export default function MCViewer() {
  const { batchId: urlBatchId } = useParams<{ batchId: string }>();
  const navigate = useNavigate();

  const [batches, setBatches] = useState<MCBatchIndex[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(urlBatchId ?? null);
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  /* ---- load batch index on mount ---- */
  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchMCBatches()
      .then((res) => setBatches(res.batches))
      .catch((err: unknown) =>
        setError(err instanceof Error ? err.message : String(err)),
      )
      .finally(() => setLoading(false));
  }, []);

  /* ---- sync URL param to selection ---- */
  useEffect(() => {
    if (urlBatchId && urlBatchId !== selectedId) {
      setSelectedId(urlBatchId);
    }
  }, [urlBatchId]);

  /* ---- load summary when a batch is selected ---- */
  useEffect(() => {
    if (!selectedId) {
      setSummary(null);
      return;
    }
    setSummaryLoading(true);
    setSummaryError(null);
    fetchMCSummary(selectedId)
      .then((res) => setSummary(res.summary))
      .catch((err: unknown) =>
        setSummaryError(err instanceof Error ? err.message : String(err)),
      )
      .finally(() => setSummaryLoading(false));
  }, [selectedId]);

  const handleSelect = (id: string) => {
    const newId = id === selectedId ? null : id;
    setSelectedId(newId);
    navigate(newId ? `/mc/${newId}` : "/mc", { replace: true });
  };

  /* ---- render ---- */

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <p className="text-gray-500 text-sm">Loading Monte Carlo batches...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-300 bg-red-50 p-6 text-sm text-red-700">
        <p className="font-semibold">Failed to load Monte Carlo batches</p>
        <p className="mt-1">{error}</p>
      </div>
    );
  }

  if (batches.length === 0) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-10 text-center text-gray-500">
        No Monte Carlo results found. Run{" "}
        <code className="rounded bg-gray-100 px-1.5 py-0.5 text-sm font-mono">
          stresslab monte-carlo
        </code>{" "}
        to generate data.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-gray-900">
        Monte Carlo Results
      </h1>

      {/* ---- batch table ---- */}
      <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white shadow-sm">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-50">
            <tr>
              {["Batch ID", "Runs", "Successful", "Created"].map((h) => (
                <th
                  key={h}
                  className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wide text-gray-500"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {batches.map((b) => {
              const active = b.batch_id === selectedId;
              return (
                <tr
                  key={b.batch_id}
                  onClick={() => handleSelect(b.batch_id)}
                  className={[
                    "cursor-pointer transition-colors",
                    active
                      ? "bg-blue-50"
                      : "hover:bg-gray-50",
                  ].join(" ")}
                >
                  <td className="whitespace-nowrap px-4 py-3 font-mono text-xs text-gray-800">
                    {b.batch_id}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-gray-700">
                    {b.n_runs ?? "\u2014"}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-gray-700">
                    {b.successful_runs ?? "\u2014"}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-gray-500">
                    {b.created_at ?? "\u2014"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ---- selected batch detail ---- */}
      {selectedId && (
        <div className="space-y-4">
          {summaryLoading && (
            <p className="text-sm text-gray-500">Loading summary...</p>
          )}

          {summaryError && (
            <div className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-700">
              <p className="font-semibold">
                Failed to load Monte Carlo summary
              </p>
              <p className="mt-1">{summaryError}</p>
            </div>
          )}

          {summary && !summaryLoading && (
            <MCPlots summary={summary} />
          )}
        </div>
      )}
    </div>
  );
}
