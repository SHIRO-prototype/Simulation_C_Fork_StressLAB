# StressLAB

StressLAB is a local simulation platform for stress-testing conjunction decision performance under nominal and degraded sensing conditions.

## What it does

- Runs configurable conjunction scenarios (single runs, sweeps, Monte Carlo).
- Produces run artifacts in `outputs/` (summary JSON + time series parquet).
- Serves a local dashboard/API for browsing, comparing, and exporting results.

## Quick start

From the repository root:

1. Install Python package (editable mode):

```bash
pip install -e .
```

2. Generate demo outputs (optional):

```bash
python scripts/demo.py
```

3. Start dashboard API + static web app:

```bash
python -m stresslab.cli serve --workspace outputs --port 8000 --dev
```

Open `http://127.0.0.1:8000`.

## Frontend dev mode (Vite)

Use the repo launcher scripts:

- PowerShell: `./run-web.ps1`
- Bash: `./run-web.sh`

These scripts start the Vite app from `web/` and expect repo-local Node/npm setup.

## Repository layout

- `src/stresslab/` - simulation, metrics, CLI, and server code
- `configs/` - scenario and demo configuration
- `outputs/` - generated artifacts
- `web/` - React/Vite dashboard source
- `tests/` - automated test suite
