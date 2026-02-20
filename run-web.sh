#!/usr/bin/env bash
# Start StressLAB dashboard (Vite) from repo root.
# Uses repo-local Node/npm when available.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$SCRIPT_DIR/web"

CANDIDATE_NODE_DIRS=(
  "$SCRIPT_DIR/.tools/node"
  "$SCRIPT_DIR/tools/node"
  "$SCRIPT_DIR/.node"
  "$SCRIPT_DIR/node"
)

for dir in "${CANDIDATE_NODE_DIRS[@]}"; do
  if [ -x "$dir/node" ]; then
    export PATH="$dir:$PATH"
    break
  fi
done

NPM_CMD="npm"
if ! command -v npm &>/dev/null; then
  for dir in "${CANDIDATE_NODE_DIRS[@]}"; do
    if [ -x "$dir/npm" ]; then
      NPM_CMD="$dir/npm"
      break
    fi
  done
fi

cd "$WEB_DIR"

if ! command -v node &>/dev/null; then
  echo ""
  echo "Node.js runtime not found in this repo shell."
  echo "  - Use the repo-local Node/npm toolchain configured for this project."
  echo "  - If you keep Node in-repo, place it under .tools/node, tools/node, .node, or node."
  echo "  - See web/README.md for repo-local run instructions."
  echo ""
  exit 1
fi

if [ ! -d "node_modules" ]; then
  echo "Installing dependencies (npm install)..."
  "$NPM_CMD" install
fi
echo "Starting Vite dev server..."
exec "$NPM_CMD" run dev
