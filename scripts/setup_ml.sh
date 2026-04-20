#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_DIR="$ROOT_DIR/ml"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"

cd "$ML_DIR"

echo "[setup_ml] Using Python: $($PYTHON_BIN --version)"
$PYTHON_BIN -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if [[ "$INSTALL_TORCH" == "1" ]]; then
  if python -m pip install -r requirements-torch.txt; then
    echo "[setup_ml] Optional torch install succeeded."
  else
    echo "[setup_ml] Optional torch install failed. Continuing in fallback mode."
    echo "[setup_ml] You can still run: python -m ml.features --force-bootstrap"
  fi
else
  echo "[setup_ml] Skipping torch install (INSTALL_TORCH=$INSTALL_TORCH)."
fi

python - <<'PY'
import importlib.util

core = ["numpy", "pandas", "requests"]
missing = [name for name in core if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"[setup_ml] Missing core ML dependencies: {missing}")

if importlib.util.find_spec("torch") is None:
    print("[setup_ml] Torch is not installed. Running in fallback mode (no trained model available).")
else:
    print("[setup_ml] Torch is installed. Trained model mode is available.")

print("[setup_ml] ML environment ready.")
PY

echo "[setup_ml] Success. Common commands:"
echo "  source ml/.venv/bin/activate"
echo "  python -m ml.features --force-bootstrap"
echo "  python -m ml.football_features --force-bootstrap"
echo "  python -m ml.train        # requires torch"
