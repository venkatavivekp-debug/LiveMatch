from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh live sports context cache for LiveMatch")
    parser.add_argument("--sport", default="cricket")
    parser.add_argument("--tournament", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    backend_dir = repo_root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.append(str(backend_dir))

    try:
        from app.services.prediction_service import PredictionService
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Live refresh requires backend dependencies. "
            "Activate backend environment first (./scripts/setup_backend.sh), then rerun this command. "
            f"import_error={exc}"
        ) from exc

    result = PredictionService.refresh_live_data(
        sport=args.sport,
        tournament=args.tournament,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
