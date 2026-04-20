from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.db.init_db import init_db
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def _initialize_db() -> None:
    init_db()


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
