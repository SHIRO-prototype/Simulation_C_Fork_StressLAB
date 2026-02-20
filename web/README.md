# StressLAB Dashboard (frontend)

React + Vite dashboard for browsing simulation runs, sweeps, and Monte Carlo batches. Proxies `/api` to the StressLAB backend (default `http://127.0.0.1:5179`).

## Repo-local runtime

This project expects Node/npm to be provided by the repository setup. Keep all runtime/tooling usage scoped to this repo.

If you keep Node in-repo, the launcher scripts auto-detect these locations:

- `.tools/node`
- `tools/node`
- `.node`
- `node`

## Start frontend

From the repo root:

- **Windows (PowerShell):** `./run-web.ps1`
- **macOS/Linux:** `./run-web.sh`

The scripts enter `web/`, install dependencies if `node_modules/` is missing, and run `npm run dev`.

## Start backend first

For the dashboard to load runs, start the API server (from repo root):

```bash
stresslab serve --workspace outputs --port 5179 --dev
```

Or:

```bash
python -m stresslab.cli serve --workspace outputs --port 5179 --dev
```

Then open the Vite URL printed in the terminal (typically `http://127.0.0.1:5173`).
