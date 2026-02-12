/**
 * Typed API client for the StressLAB dashboard backend.
 *
 * In dev mode (Vite proxy), requests go to the same origin.
 * In production, requests go to the same origin (served by FastAPI).
 */

import type {
  HealthResponse,
  RunListResponse,
  RunSummaryResponse,
  TimeseriesResponse,
  SweepListResponse,
  SweepSummaryResponse,
  MCListResponse,
  MCSummaryResponse,
} from "./types";

const API_BASE = "/api";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ---------- Health ----------

export function fetchHealth(): Promise<HealthResponse> {
  return apiFetch("/health");
}

// ---------- Runs ----------

export function fetchRuns(
  q?: string,
  limit?: number
): Promise<RunListResponse> {
  const params = new URLSearchParams();
  if (q) params.set("q", q);
  if (limit) params.set("limit", String(limit));
  const qs = params.toString();
  return apiFetch(`/runs${qs ? `?${qs}` : ""}`);
}

export function fetchRunSummary(runId: string): Promise<RunSummaryResponse> {
  return apiFetch(`/runs/${runId}/summary`);
}

export interface TimeseriesOptions {
  format?: "json" | "arrow";
  cols?: string[];
  downsample?: number;
}

export function fetchRunTimeseries(
  runId: string,
  opts?: TimeseriesOptions
): Promise<TimeseriesResponse> {
  const params = new URLSearchParams();
  params.set("format", opts?.format ?? "json");
  if (opts?.cols?.length) params.set("cols", opts.cols.join(","));
  if (opts?.downsample) params.set("downsample", String(opts.downsample));
  return apiFetch(`/runs/${runId}/timeseries?${params}`);
}

export function runDownloadUrl(runId: string, filename: string): string {
  return `${API_BASE}/runs/${runId}/download/${filename}`;
}

// ---------- Sweeps ----------

export function fetchSweeps(): Promise<SweepListResponse> {
  return apiFetch("/sweeps");
}

export function fetchSweepSummary(
  sweepId: string
): Promise<SweepSummaryResponse> {
  return apiFetch(`/sweeps/${sweepId}/summary`);
}

// ---------- Monte Carlo ----------

export function fetchMCBatches(): Promise<MCListResponse> {
  return apiFetch("/mc");
}

export function fetchMCSummary(batchId: string): Promise<MCSummaryResponse> {
  return apiFetch(`/mc/${batchId}/summary`);
}
