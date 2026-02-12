/** TypeScript interfaces matching backend Pydantic schemas. */

export interface VersionEnvelope {
  schema_version: string;
  contract_version: string;
  code_version: string;
}

export interface HealthResponse extends VersionEnvelope {
  status: string;
}

export interface SanityFlags {
  degradation_active: boolean;
  pc_diverged: boolean;
  staleness_ramped: boolean;
}

export interface ConfigSnapshot {
  update_interval: number | null;
  outage_windows: { start: number; end: number }[];
  process_noise_scale: number | null;
  pc_threshold: number | null;
  t_start: number | null;
  t_end: number | null;
  dt: number | null;
  combined_hard_body_radius: number | null;
}

export interface RunStory {
  bullets: string[];
}

export interface RunIndex {
  run_id: string;
  path: string;
  created_at: string | null;
  seed: number | null;
  dynamics_model: string | null;
  run_label: string | null;
  scenario_title: string | null;
  tags: string[];
  threshold_v1_trigger_time: number | null;
  integrity_v1_trigger_time: number | null;
  decision_compression_window: number | null;
  false_safe_rate: number | null;
  false_alert_rate: number | null;
  max_pc_degraded: number | null;
  max_staleness: number | null;
  decision_instability_index: number | null;
  total_timesteps: number | null;
  outage_sensitivity_score: number | null;
  max_pc_reference: number | null;
  max_cov_trace: number | null;
  decision_transitions_per_hour: number | null;
  decision_entropy: number | null;
  mean_pc_drift: number | null;
  max_pc_drift: number | null;
  staleness_pc_correlation: number | null;
  mean_freshness: number | null;
  min_freshness: number | null;
  sanity: SanityFlags | null;
  config: ConfigSnapshot | null;
  story: RunStory | null;
}

export interface RunListResponse extends VersionEnvelope {
  runs: RunIndex[];
  total: number;
}

export interface RunSummaryResponse extends VersionEnvelope {
  summary: Record<string, unknown>;
}

export interface TimeseriesResponse extends VersionEnvelope {
  meta: Record<string, unknown>;
  columns: string[];
  rows: (number | string | null)[][];
}

export interface SweepIndex {
  sweep_id: string;
  path: string;
  sweep_param: string | null;
  n_values: number | null;
  seed: number | null;
  created_at: string | null;
}

export interface SweepListResponse extends VersionEnvelope {
  sweeps: SweepIndex[];
  total: number;
}

export interface SweepSummaryResponse extends VersionEnvelope {
  summary: Record<string, unknown>;
}

export interface MCBatchIndex {
  batch_id: string;
  path: string;
  n_runs: number | null;
  successful_runs: number | null;
  created_at: string | null;
}

export interface MCListResponse extends VersionEnvelope {
  batches: MCBatchIndex[];
  total: number;
}

export interface MCSummaryResponse extends VersionEnvelope {
  summary: Record<string, unknown>;
}
