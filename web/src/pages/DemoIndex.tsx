import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { fetchRuns } from "../lib/api";
import type { RunIndex } from "../lib/types";

// ---------------------------------------------------------------------------
// D0-D4 demo scenario definitions (fixed order)
// ---------------------------------------------------------------------------

interface DemoScenario {
  label: string;
  title: string;
  purpose: string;
  takeaway: string;
  knobs: string;
  tags: string[];
}

const DEMOS: DemoScenario[] = [
  {
    label: "D0_nominal",
    title: "Nominal Tracking",
    purpose:
      "Control case with continuous 1-hour tracking and no outages",
    takeaway:
      "Continuous tracking keeps reference and degraded Pc streams aligned",
    knobs: "update_interval=3600s, no outages, process_noise.scale=1.0",
    tags: ["control", "no-outage", "nominal"],
  },
  {
    label: "D1_short_outage",
    title: "Short Sensor Outage",
    purpose: "Mild degradation from a 2-hour tracking gap",
    takeaway:
      "Brief outages cause transient staleness but limited Pc divergence",
    knobs: "outage 86400-93600s (2h), update_interval=3600s",
    tags: ["outage", "mild", "2-hour"],
  },
  {
    label: "D2_long_outage",
    title: "Extended Sensor Outage",
    purpose: "Severe degradation from a 12-hour tracking gap",
    takeaway:
      "Prolonged outages drive covariance blow-up and decision timing compression",
    knobs: "outage 77760-120960s (12h), seed=42",
    tags: ["outage", "severe", "12-hour"],
  },
  {
    label: "D3_sparse_cadence",
    title: "Sparse Tracking Cadence",
    purpose:
      "Slow decay from 6-hour update intervals without explicit outage",
    takeaway:
      "Low update frequency alone degrades situational awareness over time",
    knobs: "update_interval=21600s (6h), no outages",
    tags: ["sparse", "cadence", "6-hour"],
  },
  {
    label: "D4_high_process_noise",
    title: "High Process Noise",
    purpose: "Environmental uncertainty from 10x elevated dynamics noise",
    takeaway:
      "High process noise amplifies covariance growth even with continuous tracking",
    knobs: "process_noise.scale=10.0, update_interval=3600s, no outages",
    tags: ["noise", "environmental", "10x"],
  },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function DemoIndex() {
  const navigate = useNavigate();
  const [runMap, setRunMap] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);

  // Fetch all runs and build label -> run_id map
  useEffect(() => {
    fetchRuns(undefined, 500)
      .then((res) => {
        const map: Record<string, string> = {};
        for (const r of res.runs) {
          if (r.run_label && !map[r.run_label]) {
            map[r.run_label] = r.run_id;
          }
        }
        setRunMap(map);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-800">
          Demo Walkthrough
        </h2>
        <p className="mt-1 text-sm text-gray-500">
          Five canonical scenarios spanning the degradation spectrum.
          Walk through D0 to D4 in order to see how tracking gaps, sparse
          cadence, and process noise affect collision-risk classification.
        </p>
      </div>

      {loading && (
        <p className="py-8 text-center text-gray-500">
          Loading demo runs...
        </p>
      )}

      {!loading && (
        <div className="space-y-4">
          {DEMOS.map((demo, idx) => {
            const runId = runMap[demo.label];
            return (
              <div
                key={demo.label}
                className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm"
              >
                {/* Header row */}
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-xs font-bold text-white">
                        {idx}
                      </span>
                      <h3 className="text-lg font-semibold text-gray-800">
                        {demo.title}
                      </h3>
                      <span className="text-xs font-mono text-gray-400">
                        {demo.label}
                      </span>
                    </div>

                    <p className="mt-2 text-sm text-gray-600">
                      <span className="font-medium">Purpose:</span>{" "}
                      {demo.purpose}
                    </p>
                    <p className="mt-1 text-sm text-gray-600">
                      <span className="font-medium">Takeaway:</span>{" "}
                      {demo.takeaway}
                    </p>
                    <p className="mt-1 text-xs text-gray-400">
                      <span className="font-medium">Key knobs:</span>{" "}
                      {demo.knobs}
                    </p>

                    {/* Tags */}
                    <div className="mt-2 flex flex-wrap gap-1">
                      {demo.tags.map((tag) => (
                        <span
                          key={tag}
                          className="inline-block rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Action button */}
                  <div className="shrink-0">
                    {runId ? (
                      <button
                        onClick={() => navigate(`/runs/${runId}`)}
                        className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white
                                   hover:bg-blue-700 transition-colors"
                      >
                        Open Run
                      </button>
                    ) : (
                      <div className="text-right">
                        <span className="block text-xs font-medium text-amber-600">
                          Not generated
                        </span>
                        <code className="mt-1 block text-[10px] text-gray-400">
                          stresslab run --config configs/demos/{demo.label}.yaml
                        </code>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
