# StressLAB Demo Walkthrough

This guide walks through the five canonical demo scenarios (D0-D4) that
span the degradation spectrum from nominal tracking to severe outages and
elevated process noise.

## Prerequisites

- Python 3.10+ with StressLAB installed (`pip install -e .`)
- Node.js 20+ for the web dashboard
- All demo configs live in `configs/demos/`

## 1. Generate the D0-D4 Demo Runs

### Quick Start (PowerShell)

```powershell
.\scripts\demo_generate.ps1
```

### Quick Start (Bash / macOS / Linux)

```bash
bash scripts/demo_generate.sh
```

### Manual Generation

Run each scenario individually:

```bash
stresslab run --config configs/demos/D0_nominal.yaml         --output-dir outputs --seed 42
stresslab run --config configs/demos/D1_short_outage.yaml     --output-dir outputs --seed 42
stresslab run --config configs/demos/D2_long_outage.yaml      --output-dir outputs --seed 42
stresslab run --config configs/demos/D3_sparse_cadence.yaml   --output-dir outputs --seed 42
stresslab run --config configs/demos/D4_high_process_noise.yaml --output-dir outputs --seed 42
```

Each run produces a `summary_<run_id>.json` and `timeseries_<run_id>.parquet`
in the `outputs/` directory.

## 2. Start the Dashboard

```bash
stresslab serve --output-dir outputs --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## 3. Navigate the Demo Page

1. Click **Demo** in the sidebar (second item after Runs)
2. The Demo Walkthrough page shows all five scenarios in order:
   - **D0 Nominal Tracking** — control case, no outages
   - **D1 Short Sensor Outage** — 2-hour gap
   - **D2 Extended Sensor Outage** — 12-hour gap
   - **D3 Sparse Tracking Cadence** — 6-hour update intervals
   - **D4 High Process Noise** — 10x elevated dynamics noise
3. Click **Open Run** on any card to view the full run detail page

## 4. Compare to Control

When viewing any non-D0 run, the **Compared to Control (D0 Nominal)** panel
appears automatically below the story narrative. It shows metric deltas for:

- Threshold V1 Trigger time
- Decision Compression Window
- Max Pc Degraded
- Max Staleness
- Mean Freshness
- Decision Instability

Red deltas indicate degradation relative to the control; green indicates
improvement.

## 5. Export a One-Pager

From any run detail page:

1. Click **Open One-Pager** to view the print-ready report in a new tab
2. Click **Print to PDF** to trigger the browser print dialog (save as PDF)
3. The one-pager includes:
   - Header with run label, ID, seed, and version info
   - Operator snapshot (posture widget + confidence gauge)
   - Story narrative
   - Four chart panels (Pc, Staleness, Uncertainty, State Bands)
   - Full KPI table
   - Reproduction command in the footer

## 6. Scenario Metadata

Each demo YAML includes a `scenario_metadata` section:

```yaml
scenario_metadata:
  run_label: D2_long_outage
  scenario_title: "Extended Sensor Outage"
  scenario_purpose: "Severe degradation from a 12-hour tracking gap"
  scenario_takeaway: "Prolonged outages drive covariance blow-up..."
  tags: [outage, severe, 12-hour]
```

These fields are embedded in every summary JSON and surfaced in the
Runs Browser (Label/Scenario columns), Demo page, and One-Pager header.

## 7. Sanity Flags

Each run includes three sanity flags indicating whether the scenario
meaningfully exercised the degradation pipeline:

| Flag | Meaning |
|------|---------|
| `degradation_active` | At least one outage or sparse cadence was configured |
| `pc_diverged` | Degraded Pc diverged from reference Pc by a measurable amount |
| `staleness_ramped` | Staleness exceeded the baseline threshold |

D0 (control) is expected to have all flags false. D1-D4 should have
relevant flags true.

## 8. Case-Driven Stress Test Flow

StressLAB also supports an operator-first case workflow:

1. Open `/stress-test/new`
2. Upload/paste a `CaseSnapshot` (or start from preset)
3. Validate and create case
4. Run baseline (`run_label=BASELINE`)
5. Apply `StressKnobs` and run stress (`run_label=STRESS`)
6. Open `/cases/<case_id>/compare` for overlay plots + deltas
7. Open `/api/cases/<case_id>/onepager` and print to PDF

Evidence pack export:

- `/api/cases/<case_id>/export`

API endpoints added for case workflow:

- `GET /api/cases`
- `GET /api/cases/{case_id}`
- `POST /api/cases/validate`
- `POST /api/cases`
- `POST /api/cases/{case_id}/baseline`
- `POST /api/cases/{case_id}/stress`
- `GET /api/cases/{case_id}/compare`
