#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$BACKEND_DIR"

echo "[setup_backend] Using Python: $($PYTHON_BIN --version)"
$PYTHON_BIN -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python - <<'PY'
import importlib.util
import sys

required = ["fastapi", "uvicorn", "sqlalchemy", "pydantic", "numpy", "pandas"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(f"[setup_backend] Missing packages after install: {missing}")
    print("[setup_backend] Re-run: source backend/.venv/bin/activate && pip install -r backend/requirements.txt")
    sys.exit(1)

print("[setup_backend] Backend dependencies are installed and importable.")
PY

echo "[setup_backend] Success. Start backend with:"
echo "  cd backend && source .venv/bin/activate && python -m uvicorn app.main:app --reload --port 8000"
