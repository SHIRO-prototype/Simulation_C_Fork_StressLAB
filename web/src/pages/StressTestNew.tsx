import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { createCase, fetchCase, runBaseline, runStress, validateCaseSnapshot } from "../lib/api";

type JsonMap = Record<string, unknown>;

const PRESETS: Record<string, JsonMap> = {
  "LEO close pass (nominal)": {
    state_obj1: [6878.137, 0, 0, 0, 7.6126, 0],
    state_obj2: [6878.137, 0, 0.8, 0, 7.6126, -0.001],
    cov_obj1: [[0.01, 0, 0, 0, 0, 0], [0, 0.01, 0, 0, 0, 0], [0, 0, 0.01, 0, 0, 0], [0, 0, 0, 1e-8, 0, 0], [0, 0, 0, 0, 1e-8, 0], [0, 0, 0, 0, 0, 1e-8]],
    cov_obj2: [[0.01, 0, 0, 0, 0, 0], [0, 0.01, 0, 0, 0, 0], [0, 0, 0.01, 0, 0, 0], [0, 0, 0, 1e-8, 0, 0], [0, 0, 0, 0, 1e-8, 0], [0, 0, 0, 0, 0, 1e-8]],
    seed: 42,
    dynamics_model: "two_body_plus_J2",
    t_start: 0,
    t_end: 259200,
    dt: 60,
    combined_hard_body_radius: 0.02,
    measurement_update_interval: 3600,
    measurement_noise_sigma_pos: 0.01,
    process_noise_scale: 1,
    process_noise_sigma_radial: 1e-9,
    process_noise_sigma_tangential: 1e-9,
    process_noise_sigma_normal: 1e-9,
    threshold_pc: 1e-4,
    threshold_time_to_tca_gate: 86400,
    threshold_miss_distance: 1,
  },
  "LEO close pass (high covariance)": {
    state_obj1: [6878.137, 0, 0, 0, 7.6126, 0],
    state_obj2: [6878.137, 0, 1.5, 0, 7.6126, -0.0012],
    cov_obj1: [[0.08, 0, 0, 0, 0, 0], [0, 0.08, 0, 0, 0, 0], [0, 0, 0.08, 0, 0, 0], [0, 0, 0, 2e-8, 0, 0], [0, 0, 0, 0, 2e-8, 0], [0, 0, 0, 0, 0, 2e-8]],
    cov_obj2: [[0.08, 0, 0, 0, 0, 0], [0, 0.08, 0, 0, 0, 0], [0, 0, 0.08, 0, 0, 0], [0, 0, 0, 2e-8, 0, 0], [0, 0, 0, 0, 2e-8, 0], [0, 0, 0, 0, 0, 2e-8]],
    seed: 52,
    dynamics_model: "two_body_plus_J2",
    t_start: 0,
    t_end: 259200,
    dt: 60,
    combined_hard_body_radius: 0.02,
    measurement_update_interval: 3600,
    measurement_noise_sigma_pos: 0.02,
    process_noise_scale: 1,
    process_noise_sigma_radial: 1e-9,
    process_noise_sigma_tangential: 1e-9,
    process_noise_sigma_normal: 1e-9,
    threshold_pc: 1e-4,
    threshold_time_to_tca_gate: 86400,
    threshold_miss_distance: 1,
  },
  "Cross-plane high Vrel": {
    state_obj1: [6878.137, 0, 0, 0, 7.6126, 0],
    state_obj2: [6878.137, -2.0, 1.2, 0.003, 7.608, -0.004],
    cov_obj1: [[0.02, 0, 0, 0, 0, 0], [0, 0.02, 0, 0, 0, 0], [0, 0, 0.02, 0, 0, 0], [0, 0, 0, 1e-8, 0, 0], [0, 0, 0, 0, 1e-8, 0], [0, 0, 0, 0, 0, 1e-8]],
    cov_obj2: [[0.02, 0, 0, 0, 0, 0], [0, 0.02, 0, 0, 0, 0], [0, 0, 0.02, 0, 0, 0], [0, 0, 0, 1e-8, 0, 0], [0, 0, 0, 0, 1e-8, 0], [0, 0, 0, 0, 0, 1e-8]],
    seed: 64,
    dynamics_model: "two_body_plus_J2",
    t_start: 0,
    t_end: 172800,
    dt: 60,
    combined_hard_body_radius: 0.02,
    measurement_update_interval: 1800,
    measurement_noise_sigma_pos: 0.01,
    process_noise_scale: 1,
    process_noise_sigma_radial: 1e-9,
    process_noise_sigma_tangential: 1e-9,
    process_noise_sigma_normal: 1e-9,
    threshold_pc: 1e-4,
    threshold_time_to_tca_gate: 86400,
    threshold_miss_distance: 1,
  },
  "Sparse tracking baseline": {
    state_obj1: [6878.137, 0, 0, 0, 7.6126, 0],
    state_obj2: [6878.137, 0, 1.0, 0, 7.6126, -0.001],
    cov_obj1: [[0.02, 0, 0, 0, 0, 0], [0, 0.02, 0, 0, 0, 0], [0, 0, 0.02, 0, 0, 0], [0, 0, 0, 2e-8, 0, 0], [0, 0, 0, 0, 2e-8, 0], [0, 0, 0, 0, 0, 2e-8]],
    cov_obj2: [[0.02, 0, 0, 0, 0, 0], [0, 0.02, 0, 0, 0, 0], [0, 0, 0.02, 0, 0, 0], [0, 0, 0, 2e-8, 0, 0], [0, 0, 0, 0, 2e-8, 0], [0, 0, 0, 0, 0, 2e-8]],
    seed: 91,
    dynamics_model: "two_body_plus_J2",
    t_start: 0,
    t_end: 259200,
    dt: 120,
    combined_hard_body_radius: 0.02,
    measurement_update_interval: 7200,
    measurement_noise_sigma_pos: 0.02,
    process_noise_scale: 1,
    process_noise_sigma_radial: 1e-9,
    process_noise_sigma_tangential: 1e-9,
    process_noise_sigma_normal: 1e-9,
    threshold_pc: 1e-4,
    threshold_time_to_tca_gate: 86400,
    threshold_miss_distance: 1,
  },
};

function asNum(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function trace6(cov: unknown): number | null {
  if (!Array.isArray(cov) || cov.length !== 6) return null;
  let t = 0;
  for (let i = 0; i < 6; i += 1) {
    const row = cov[i];
    if (!Array.isArray(row) || row.length !== 6) return null;
    const d = asNum(row[i]);
    if (d == null) return null;
    t += d;
  }
  return t;
}

function separation(snapshot: JsonMap | null): number | null {
  if (!snapshot) return null;
  const a = snapshot.state_obj1;
  const b = snapshot.state_obj2;
  if (!Array.isArray(a) || !Array.isArray(b) || a.length < 3 || b.length < 3) return null;
  const dx = asNum(a[0]);
  const dy = asNum(a[1]);
  const dz = asNum(a[2]);
  const ex = asNum(b[0]);
  const ey = asNum(b[1]);
  const ez = asNum(b[2]);
  if (dx == null || dy == null || dz == null || ex == null || ey == null || ez == null) return null;
  const rx = ex - dx;
  const ry = ey - dy;
  const rz = ez - dz;
  return Math.sqrt(rx * rx + ry * ry + rz * rz);
}

function tcaGuess(snapshot: JsonMap | null): number | null {
  if (!snapshot) return null;
  const a = snapshot.state_obj1;
  const b = snapshot.state_obj2;
  if (!Array.isArray(a) || !Array.isArray(b) || a.length < 6 || b.length < 6) return null;
  const r = [asNum(b[0]), asNum(b[1]), asNum(b[2]), asNum(a[0]), asNum(a[1]), asNum(a[2])];
  const v = [asNum(b[3]), asNum(b[4]), asNum(b[5]), asNum(a[3]), asNum(a[4]), asNum(a[5])];
  if (r.some((x) => x == null) || v.some((x) => x == null)) return null;
  const rx = (r[0] as number) - (r[3] as number);
  const ry = (r[1] as number) - (r[4] as number);
  const rz = (r[2] as number) - (r[5] as number);
  const vx = (v[0] as number) - (v[3] as number);
  const vy = (v[1] as number) - (v[4] as number);
  const vz = (v[2] as number) - (v[5] as number);
  const vv = vx * vx + vy * vy + vz * vz;
  if (vv <= 1e-16) return null;
  return -((rx * vx + ry * vy + rz * vz) / vv);
}

function mergeCDM(snapshot: JsonMap, cdm: JsonMap): JsonMap {
  const next = { ...snapshot };
  if (Array.isArray(cdm.primary_state) && cdm.primary_state.length === 6) next.state_obj1 = cdm.primary_state;
  if (Array.isArray(cdm.secondary_state) && cdm.secondary_state.length === 6) next.state_obj2 = cdm.secondary_state;
  if (Array.isArray(cdm.primary_covariance) && cdm.primary_covariance.length === 6) next.cov_obj1 = cdm.primary_covariance;
  if (Array.isArray(cdm.secondary_covariance) && cdm.secondary_covariance.length === 6) next.cov_obj2 = cdm.secondary_covariance;
  if (typeof cdm.epoch === "string") next.epoch = cdm.epoch;
  return next;
}

const WORKFLOW_STEPS = [
  {
    title: "Load or choose a case snapshot",
    copy: "Start from an upload or a curated LEO preset with states and covariance already aligned to the schema.",
  },
  {
    title: "Validate the orbital picture",
    copy: "Run structural, PSD, and sanity checks before creating a case record in the local workspace.",
  },
  {
    title: "Execute baseline and stress runs",
    copy: "Preserve a clean control, then degrade cadence, inject outages, and widen uncertainty to see decision drift.",
  },
];

function formatPreviewSeconds(value: number | null): string {
  if (value == null) return "-";
  if (Math.abs(value) >= 3600) return `${(value / 3600).toFixed(2)} h`;
  return `${value.toFixed(1)} s`;
}

export default function StressTestNew() {
  const navigate = useNavigate();
  const { caseId: routeCaseId } = useParams<{ caseId: string }>();
  const [usePreset, setUsePreset] = useState(false);
  const [presetName, setPresetName] = useState(Object.keys(PRESETS)[0]);
  const [snapshot, setSnapshot] = useState<JsonMap | null>(PRESETS[Object.keys(PRESETS)[0]]);
  const [snapshotRawText, setSnapshotRawText] = useState<string>("");
  const [snapshotFormat, setSnapshotFormat] = useState<"json" | "yaml" | "yml" | "auto">("json");
  const [snapshotName, setSnapshotName] = useState("preset");
  const [cdmName, setCdmName] = useState("");
  const [validationOk, setValidationOk] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [status, setStatus] = useState("");
  const [caseId, setCaseId] = useState(routeCaseId ?? "");
  const [baselineDone, setBaselineDone] = useState(false);
  const [stressDone, setStressDone] = useState(false);
  const [measurementCadenceSeconds, setMeasurementCadenceSeconds] = useState(60);
  const [outageStartH, setOutageStartH] = useState(3);
  const [outageDurationH, setOutageDurationH] = useState(2);
  const [processNoiseScale, setProcessNoiseScale] = useState(1);
  const [maneuverEnabled, setManeuverEnabled] = useState(false);
  const [deltaVSigmaMps, setDeltaVSigmaMps] = useState(0);

  useEffect(() => {
    if (!routeCaseId) return;
    fetchCase(routeCaseId)
      .then((res) => {
        setCaseId(res.case.case_id);
        setSnapshot(res.snapshot);
        setValidationOk(true);
        setBaselineDone(Boolean(res.case.baseline_run_id));
        setStressDone(Boolean(res.case.stress_run_id));
      })
      .catch((e) => {
        setStatus(e instanceof Error ? e.message : String(e));
      });
  }, [routeCaseId]);

  useEffect(() => {
    if (usePreset) {
      setSnapshot(PRESETS[presetName]);
      setSnapshotName(presetName);
      setValidationOk(false);
      setValidationErrors([]);
    }
  }, [usePreset, presetName]);

  const preview = useMemo(() => {
    const sep = separation(snapshot);
    const tca = tcaGuess(snapshot);
    const tr1 = trace6(snapshot?.cov_obj1);
    const tr2 = trace6(snapshot?.cov_obj2);
    const epoch = typeof snapshot?.epoch === "string" ? snapshot.epoch : "not provided";
    return { sep, tca, tr1, tr2, epoch };
  }, [snapshot]);

  async function onSnapshotUpload(file: File) {
    const text = await file.text();
    const lower = file.name.toLowerCase();
    const fmt: "json" | "yaml" | "yml" = lower.endsWith(".json") ? "json" : lower.endsWith(".yml") ? "yml" : "yaml";
    setSnapshotFormat(fmt);
    setSnapshotRawText(text);
    let parsed: JsonMap | null = null;
    if (fmt === "json") {
      try {
        parsed = JSON.parse(text) as JsonMap;
      } catch {
        parsed = null;
      }
    }
    setSnapshot(parsed);
    setSnapshotName(file.name);
    setValidationOk(false);
    setValidationErrors([]);
    setStatus(`Loaded snapshot: ${file.name}`);
  }

  async function onCdmUpload(file: File) {
    const text = await file.text();
    setCdmName(file.name);
    try {
      const parsed = JSON.parse(text) as JsonMap;
      if (snapshot) {
        const merged = mergeCDM(snapshot, parsed);
        setSnapshot(merged);
        setStatus(`Applied CDM mapping from ${file.name}`);
      }
    } catch {
      setStatus(`CDM uploaded (${file.name}). Automatic mapping supports JSON-only CDM right now.`);
    }
  }

  async function onValidate() {
    if (!snapshot && !snapshotRawText) {
      setStatus("Upload a snapshot or choose a preset first.");
      return;
    }
    try {
      const validatePayload = snapshot
        ? snapshot
        : { raw_text: snapshotRawText, source_format: snapshotFormat };
      const result = await validateCaseSnapshot(validatePayload);
      const normalized = result.snapshot;
      if (normalized && typeof normalized === "object") {
        setSnapshot(normalized as JsonMap);
      }
      setValidationOk(true);
      setValidationErrors([]);
      setStatus("Validation passed.");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setValidationOk(false);
      setValidationErrors([msg]);
      setStatus("Validation failed.");
    }
  }

  async function onCreateCase() {
    if (!validationOk) return;
    try {
      const payload = snapshot
        ? { source: usePreset ? "preset" : "upload", notes: snapshotName, snapshot }
        : { source: usePreset ? "preset" : "upload", notes: snapshotName, raw_text: snapshotRawText, source_format: snapshotFormat };
      const res = await createCase(payload as never);
      setCaseId(res.case.case_id);
      setStatus(`Case created: ${res.case.case_id}`);
      navigate(`/stress-test/${res.case.case_id}`);
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
    }
  }

  async function onRunBaseline() {
    if (!caseId) return;
    try {
      const res = await runBaseline(caseId);
      setBaselineDone(String(res.status) === "done");
      setStatus(`Baseline: ${JSON.stringify(res)}`);
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
    }
  }

  async function onRunStress() {
    if (!caseId || !baselineDone) return;
    const outageStart = Math.max(0, outageStartH * 3600);
    const outageEnd = outageStart + Math.max(0, outageDurationH * 3600);
    const knobs = {
      measurement_cadence_s: measurementCadenceSeconds,
      outage_windows: [{ start: outageStart, end: outageEnd }],
      process_noise_scale: processNoiseScale,
      maneuver_enabled: maneuverEnabled,
      maneuver_delta_v_sigma: deltaVSigmaMps / 1000,
      maneuver_execution_time: 0,
    };
    try {
      const res = await runStress(caseId, knobs);
      setStressDone(String(res.status) === "done");
      setStatus(`Stress: ${JSON.stringify(res)}`);
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
    }
  }

  const statusTone = validationErrors.length > 0
    ? "status-pill status-pill-critical"
    : stressDone
      ? "status-pill status-pill-ok"
      : validationOk
        ? "status-pill status-pill-warn"
        : "status-pill status-pill-warn";

  return (
    <div className="surface-grid">
      <section className="section-card section-card-dark section-pad">
        <div className="grid gap-8 xl:grid-cols-[1.3fr,0.8fr]">
          <div>
            <span className="section-kicker" style={{ color: "rgba(244, 238, 229, 0.72)" }}>
              Orbital Stress Studio
            </span>
            <h1 className="page-title" style={{ color: "#fbf5ec", marginTop: 14 }}>
              Make the case, bend the sensing chain, compare the decisions.
            </h1>
            <p className="page-subtitle" style={{ color: "rgba(244, 238, 229, 0.72)", marginTop: 18 }}>
              Upload a conjunction snapshot or start from a preset, then drive a cinematic operator workflow from validation through evidence-pack export.
            </p>

            <div className="mt-6 flex flex-wrap gap-3">
              <span className={statusTone}>
                {validationErrors.length > 0 ? "Validation blocked" : stressDone ? "Stress complete" : validationOk ? "Ready to create case" : "Awaiting validation"}
              </span>
              {caseId && <span className="status-pill status-pill-ok">Case {caseId}</span>}
              {baselineDone && <span className="status-pill status-pill-ok">Baseline done</span>}
              {stressDone && <span className="status-pill status-pill-ok">Stress done</span>}
            </div>
          </div>

          <div className="workflow-card">
            <span className="section-kicker">Workflow</span>
            <div className="workflow-list mt-4">
              {WORKFLOW_STEPS.map((step, index) => (
                <div key={step.title} className="workflow-step">
                  <span className="workflow-index">{index + 1}</span>
                  <div>
                    <strong>{step.title}</strong>
                    <span>{step.copy}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
          <div className="meta-card">
            <div className="meta-card-label">Epoch</div>
            <div className="meta-card-value" style={{ fontSize: "1rem" }}>{preview.epoch}</div>
          </div>
          <div className="meta-card">
            <div className="meta-card-label">Initial separation</div>
            <div className="meta-card-value">{preview.sep == null ? "-" : `${preview.sep.toFixed(3)} km`}</div>
          </div>
          <div className="meta-card">
            <div className="meta-card-label">Primary covariance trace</div>
            <div className="meta-card-value">{preview.tr1 == null ? "-" : preview.tr1.toExponential(3)}</div>
          </div>
          <div className="meta-card">
            <div className="meta-card-label">Secondary covariance trace</div>
            <div className="meta-card-value">{preview.tr2 == null ? "-" : preview.tr2.toExponential(3)}</div>
          </div>
          <div className="meta-card">
            <div className="meta-card-label">Rough TCA guess</div>
            <div className="meta-card-value">{formatPreviewSeconds(preview.tca)}</div>
          </div>
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-[1.2fr,0.8fr]">
        <section className="section-card section-pad">
          <span className="section-kicker">Inputs</span>
          <h2 className="section-heading">Snapshot ingestion</h2>
          <p className="section-copy">
            Bring in a raw case snapshot, optionally merge a CDM payload, or switch into preset mode for a fast starting point.
          </p>

          <div className="mt-8 grid gap-5 lg:grid-cols-2">
            <div>
              <label className="field-label">Case snapshot</label>
              <input
                type="file"
                accept=".json,.yaml,.yml"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onSnapshotUpload(f);
                }}
                className="field-input"
              />
              <p className="field-hint">
                Upload epoch, state vectors, and 6x6 covariance matrices in the StressLAB case schema.
              </p>
              <div className="mt-3 field-chip">Loaded snapshot: {snapshotName}</div>
            </div>

            <div>
              <label className="field-label">CDM overlay</label>
              <input
                type="file"
                accept=".xml,.json,.txt"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onCdmUpload(f);
                }}
                className="field-input"
              />
              <p className="field-hint">
                Optional mapping layer for supported JSON CDM fields. Unsupported formats are still retained for status feedback.
              </p>
              <div className="mt-3 field-chip">Loaded CDM: {cdmName || "none"}</div>
            </div>
          </div>

          <div className="mt-8 rounded-[24px] border border-slate-900/10 bg-white/55 p-4">
            <label className="inline-flex items-center gap-3 text-sm font-semibold text-slate-700">
              <input
                type="checkbox"
                checked={usePreset}
                onChange={(e) => setUsePreset(e.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-slate-900"
              />
              Use a curated preset instead of a custom upload
            </label>
            {usePreset && (
              <div className="mt-4">
                <label className="field-label">Preset</label>
                <select
                  className="field-select"
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                >
                  {Object.keys(PRESETS).map((name) => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </section>

        <section className="section-card section-pad">
          <span className="section-kicker">Validation</span>
          <h2 className="section-heading">Readiness panel</h2>
          <p className="section-copy">
            Resolve structural and covariance issues before the case is committed to the local evidence workspace.
          </p>

          <div className="mt-8 grid gap-3">
            <div className="meta-card">
              <div className="meta-card-label">Schema completeness</div>
              <div className="meta-card-value">{validationOk ? "Passed" : "Pending"}</div>
            </div>
            <div className="meta-card">
              <div className="meta-card-label">Covariance symmetry and PSD</div>
              <div className="meta-card-value">{validationOk ? "Passed" : "Pending"}</div>
            </div>
            <div className="meta-card">
              <div className="meta-card-label">Unit sanity bounds</div>
              <div className="meta-card-value">{validationOk ? "Passed" : "Pending"}</div>
            </div>
          </div>

          {validationErrors.length > 0 && (
            <div className="status-banner mt-6" style={{ borderColor: "rgba(196, 70, 70, 0.18)", background: "rgba(255, 124, 124, 0.08)" }}>
              <strong>Validation errors</strong>
              {validationErrors.map((err) => (
                <p key={err}>{err}</p>
              ))}
            </div>
          )}

          <div className="action-row mt-6">
            <button onClick={onValidate} className="action-button action-button-primary">
              Validate snapshot
            </button>
          </div>
        </section>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="section-card section-pad">
          <span className="section-kicker">Control Run</span>
          <h2 className="section-heading">Baseline execution</h2>
          <p className="section-copy">
            Create the case record, preserve the nominal run, and lock the reference trace before applying any stressors.
          </p>

          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            <div className="meta-card">
              <div className="meta-card-label">Case identifier</div>
              <div className="meta-card-value">{caseId || "Not created"}</div>
            </div>
            <div className="meta-card">
              <div className="meta-card-label">Baseline status</div>
              <div className="meta-card-value">{baselineDone ? "Complete" : caseId ? "Ready to run" : "Waiting for case"}</div>
            </div>
          </div>

          <div className="action-row mt-8">
            <button
              onClick={onCreateCase}
              disabled={!validationOk}
              className="action-button action-button-primary"
            >
              Create case
            </button>
            <button
              onClick={onRunBaseline}
              disabled={!caseId}
              className="action-button action-button-secondary"
            >
              Run baseline
            </button>
          </div>
        </section>

        <section className="section-card section-pad">
          <span className="section-kicker">Stress Model</span>
          <h2 className="section-heading">Degrade the sensing picture</h2>
          <p className="section-copy">
            Dial cadence, outages, process noise, and maneuver uncertainty until the decision envelope starts to deform.
          </p>

          <div className="mt-8 grid gap-4 md:grid-cols-2">
            <div>
              <label className="field-label">Measurement cadence seconds</label>
              <input
                type="number"
                className="field-input"
                value={measurementCadenceSeconds}
                onChange={(e) => setMeasurementCadenceSeconds(Number(e.target.value))}
              />
            </div>
            <div>
              <label className="field-label">Outage start hours</label>
              <input
                type="number"
                className="field-input"
                value={outageStartH}
                onChange={(e) => setOutageStartH(Number(e.target.value))}
              />
            </div>
            <div>
              <label className="field-label">Outage duration hours</label>
              <input
                type="number"
                className="field-input"
                value={outageDurationH}
                onChange={(e) => setOutageDurationH(Number(e.target.value))}
              />
            </div>
            <div>
              <label className="field-label">Process noise scale</label>
              <input
                type="number"
                step="0.1"
                className="field-input"
                value={processNoiseScale}
                onChange={(e) => setProcessNoiseScale(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="mt-6 rounded-[24px] border border-slate-900/10 bg-white/55 p-4">
            <label className="inline-flex items-center gap-3 text-sm font-semibold text-slate-700">
              <input
                type="checkbox"
                checked={maneuverEnabled}
                onChange={(e) => setManeuverEnabled(e.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-slate-900"
              />
              Enable maneuver uncertainty
            </label>
            {maneuverEnabled && (
              <div className="mt-4">
                <label className="field-label">Delta-v sigma m/s</label>
                <input
                  type="number"
                  step="0.1"
                  className="field-input"
                  value={deltaVSigmaMps}
                  onChange={(e) => setDeltaVSigmaMps(Number(e.target.value))}
                />
              </div>
            )}
          </div>

          <div className="action-row mt-8">
            <button
              onClick={onRunStress}
              disabled={!baselineDone}
              className="action-button action-button-accent"
            >
              Run stress test
            </button>
          </div>
        </section>
      </div>

      <section className="section-card section-pad">
        <div className="grid gap-6 xl:grid-cols-[1.1fr,0.9fr]">
          <div>
            <span className="section-kicker">Result Access</span>
            <h2 className="section-heading">Open the comparison view</h2>
            <p className="section-copy">
              Once both runs complete, move directly into the compare dashboard for overlays, deltas, the one-pager, and the export pack.
            </p>
            <div className="action-row mt-8">
              <button
                onClick={() => caseId && navigate(`/cases/${caseId}/compare`)}
                disabled={!stressDone || !caseId}
                className="action-button action-button-dark"
              >
                Open baseline vs stress compare
              </button>
            </div>
          </div>

          <div className="status-banner">
            <strong>Workspace status</strong>
            <p>Case ID: {caseId || "not created"}</p>
            <p>Status: {status || "idle"}</p>
            <p>Snapshot source: {usePreset ? `preset: ${presetName}` : snapshotName}</p>
          </div>
        </div>
      </section>
    </div>
  );
}
