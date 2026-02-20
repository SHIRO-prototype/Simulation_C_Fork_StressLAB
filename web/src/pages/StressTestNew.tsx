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
    if (parsed) {
      setSnapshot(parsed);
    } else {
      setSnapshot(null);
    }
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
    if (!snapshot) {
      if (!snapshotRawText) {
        setStatus("Upload a snapshot or choose a preset first.");
        return;
      }
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

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Stress Tester</h1>
        <p className="mt-1 text-sm text-gray-600">
          Upload a conjunction snapshot (primary + secondary) and stress test decision behavior under degraded tracking.
        </p>
      </div>

      <section className="rounded border border-gray-200 bg-white p-4">
        <h2 className="text-lg font-semibold text-gray-900">Upload Inputs</h2>
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-700">Case Snapshot</label>
            <input
              type="file"
              accept=".json,.yaml,.yml"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onSnapshotUpload(f);
              }}
              className="block w-full rounded border border-gray-300 p-2 text-sm"
            />
            <p className="mt-1 text-xs text-gray-500">Upload snapshot containing epoch, state vectors, and 6x6 covariances.</p>
            <p className="mt-1 text-xs text-gray-700">Loaded: {snapshotName}</p>
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-gray-700">CDM (Optional)</label>
            <input
              type="file"
              accept=".xml,.json,.txt"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onCdmUpload(f);
              }}
              className="block w-full rounded border border-gray-300 p-2 text-sm"
            />
            <p className="mt-1 text-xs text-gray-500">Optional parse + map into CaseSnapshot if known JSON fields are present.</p>
            <p className="mt-1 text-xs text-gray-700">Loaded: {cdmName || "none"}</p>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <label className="inline-flex items-center gap-2 text-sm text-gray-700">
            <input type="checkbox" checked={usePreset} onChange={(e) => setUsePreset(e.target.checked)} />
            Use preset instead of upload
          </label>
          {usePreset && (
            <select
              className="rounded border border-gray-300 px-3 py-2 text-sm"
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
            >
              {Object.keys(PRESETS).map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          )}
        </div>
      </section>

      <section className="rounded border border-gray-200 bg-white p-4">
        <h2 className="text-lg font-semibold text-gray-900">Validation and Preview</h2>
        <div className="mt-3 grid gap-4 lg:grid-cols-2">
          <div className="rounded border border-gray-200 bg-gray-50 p-3">
            <div className="text-sm font-medium text-gray-800">Validation Panel</div>
            <ul className="mt-2 space-y-1 text-xs text-gray-700">
              <li>missing fields: {validationOk ? "ok" : "pending"}</li>
              <li>covariance symmetry check: {validationOk ? "ok" : "pending"}</li>
              <li>PSD check (eigenvalues &gt;= -eps): {validationOk ? "ok" : "pending"}</li>
              <li>unit sanity bounds: {validationOk ? "ok" : "pending"}</li>
            </ul>
            {validationErrors.length > 0 && (
              <div className="mt-2 rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">
                {validationErrors.map((err) => (
                  <div key={err}>{err}</div>
                ))}
              </div>
            )}
            <button onClick={onValidate} className="mt-3 rounded bg-gray-800 px-3 py-2 text-sm text-white">Validate</button>
          </div>

          <div className="rounded border border-gray-200 bg-gray-50 p-3">
            <div className="text-sm font-medium text-gray-800">Preview Cards</div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
              <div className="rounded bg-white p-2"><div className="text-gray-500">epoch</div><div className="font-semibold">{preview.epoch}</div></div>
              <div className="rounded bg-white p-2"><div className="text-gray-500">rough separation</div><div className="font-semibold">{preview.sep == null ? "-" : `${preview.sep.toFixed(3)} km`}</div></div>
              <div className="rounded bg-white p-2"><div className="text-gray-500">cov trace primary</div><div className="font-semibold">{preview.tr1 == null ? "-" : preview.tr1.toExponential(3)}</div></div>
              <div className="rounded bg-white p-2"><div className="text-gray-500">cov trace secondary</div><div className="font-semibold">{preview.tr2 == null ? "-" : preview.tr2.toExponential(3)}</div></div>
              <div className="rounded bg-white p-2 col-span-2"><div className="text-gray-500">initial TCA guess</div><div className="font-semibold">{preview.tca == null ? "-" : `${preview.tca.toFixed(1)} s`}</div></div>
            </div>
          </div>
        </div>
      </section>

      <section className="rounded border border-gray-200 bg-white p-4">
        <h2 className="text-lg font-semibold text-gray-900">Baseline</h2>
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            onClick={onCreateCase}
            disabled={!validationOk}
            className="rounded bg-blue-700 px-3 py-2 text-sm text-white disabled:cursor-not-allowed disabled:bg-gray-300"
          >
            Create Case
          </button>
          <button
            onClick={onRunBaseline}
            disabled={!caseId}
            className="rounded bg-green-700 px-3 py-2 text-sm text-white disabled:cursor-not-allowed disabled:bg-gray-300"
          >
            Run Baseline
          </button>
        </div>
      </section>

      <section className="rounded border border-gray-200 bg-white p-4">
        <h2 className="text-lg font-semibold text-gray-900">Apply Stress</h2>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <label className="text-sm text-gray-700">measurement_cadence_seconds
            <input type="number" className="mt-1 w-full rounded border border-gray-300 px-2 py-1" value={measurementCadenceSeconds} onChange={(e) => setMeasurementCadenceSeconds(Number(e.target.value))} />
          </label>
          <label className="text-sm text-gray-700">outage start hours before now
            <input type="number" className="mt-1 w-full rounded border border-gray-300 px-2 py-1" value={outageStartH} onChange={(e) => setOutageStartH(Number(e.target.value))} />
          </label>
          <label className="text-sm text-gray-700">outage duration hours
            <input type="number" className="mt-1 w-full rounded border border-gray-300 px-2 py-1" value={outageDurationH} onChange={(e) => setOutageDurationH(Number(e.target.value))} />
          </label>
          <label className="text-sm text-gray-700">process_noise_scale
            <input type="number" step="0.1" className="mt-1 w-full rounded border border-gray-300 px-2 py-1" value={processNoiseScale} onChange={(e) => setProcessNoiseScale(Number(e.target.value))} />
          </label>
          <label className="inline-flex items-center gap-2 pt-6 text-sm text-gray-700">
            <input type="checkbox" checked={maneuverEnabled} onChange={(e) => setManeuverEnabled(e.target.checked)} />
            maneuver_uncertainty_enabled
          </label>
          {maneuverEnabled && (
            <label className="text-sm text-gray-700">delta_v_sigma_mps
              <input type="number" step="0.1" className="mt-1 w-full rounded border border-gray-300 px-2 py-1" value={deltaVSigmaMps} onChange={(e) => setDeltaVSigmaMps(Number(e.target.value))} />
            </label>
          )}
        </div>
        <button
          onClick={onRunStress}
          disabled={!baselineDone}
          className="mt-3 rounded bg-amber-600 px-3 py-2 text-sm text-white disabled:cursor-not-allowed disabled:bg-gray-300"
        >
          Run Stress Test
        </button>
      </section>

      <section className="rounded border border-gray-200 bg-white p-4">
        <h2 className="text-lg font-semibold text-gray-900">Results</h2>
        <button
          onClick={() => caseId && navigate(`/cases/${caseId}/compare`)}
          disabled={!stressDone || !caseId}
          className="mt-3 rounded bg-indigo-700 px-3 py-2 text-sm text-white disabled:cursor-not-allowed disabled:bg-gray-300"
        >
          Open Baseline vs Stress Compare
        </button>
      </section>

      <div className="rounded border border-gray-200 bg-gray-50 p-3 text-xs text-gray-700">
        <div><span className="font-semibold">Case ID:</span> {caseId || "not created"}</div>
        <div><span className="font-semibold">Status:</span> {status || "idle"}</div>
      </div>
    </div>
  );
}
