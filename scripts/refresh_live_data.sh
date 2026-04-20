#!/usr/bin/env bash
set -euo pipefail

SPORT="${1:-cricket}"
TOURNAMENT="${2:-IPL}"

cd "$(dirname "$0")/.."

if [ -f "backend/.venv/bin/python" ]; then
  PYTHON_BIN="backend/.venv/bin/python"
elif [ -f "ml/.venv/bin/python" ]; then
  PYTHON_BIN="ml/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m ml.refresh_live_data --sport "$SPORT" --tournament "$TOURNAMENT"
