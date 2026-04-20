from __future__ import annotations

import importlib.util
import logging
import sys
from platform import python_version

logger = logging.getLogger(__name__)


def run_startup_checks() -> None:
    check_python_version()
    check_backend_dependencies()


def check_python_version() -> None:
    version_string = python_version()
    if sys.version_info < (3, 10):
        logger.warning(
            "Python %s detected. Recommended: Python 3.10+. "
            "Compatible mode enabled for Python 3.9 with numpy<2.",
            version_string,
        )
    else:
        logger.info("Python %s detected.", version_string)


def check_backend_dependencies() -> None:
    required = ["fastapi", "uvicorn", "sqlalchemy", "pydantic", "numpy", "pandas"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        logger.error(
            "Missing backend dependencies: %s. "
            "Fix: run scripts/setup_backend.sh then start with `python -m uvicorn app.main:app --reload --port 8000`.",
            ", ".join(missing),
        )
    else:
        logger.info("Backend dependency check passed.")

    if importlib.util.find_spec("torch") is None:
        logger.info("Torch not found. ML will run in fallback mode unless torch is installed.")
