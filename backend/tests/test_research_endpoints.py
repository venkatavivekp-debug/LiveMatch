from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


def test_dataset_experiment_training_flow(client: TestClient) -> None:
    dataset_key = f"pytest_dataset_{uuid.uuid4().hex[:8]}"

    dataset_response = client.post(
        "/datasets/register",
        json={
            "dataset_key": dataset_key,
            "sport": "cricket",
            "tournament": "IPL",
            "source_type": "local",
            "manifest_path": "data/processed/feature_manifest.json",
            "row_count": 100,
            "details": {"purpose": "pytest smoke"},
        },
    )
    assert dataset_response.status_code == 200, dataset_response.text
    assert dataset_response.json()["dataset_key"] == dataset_key

    experiment_response = client.post(
        "/experiments/create",
        json={
            "name": "Pytest experiment",
            "sport": "cricket",
            "task": "match_forecast",
            "dataset_key": dataset_key,
            "config": {"num_heads": 4},
        },
    )
    assert experiment_response.status_code == 200, experiment_response.text
    experiment_id = experiment_response.json()["experiment_id"]

    start_response = client.post(
        "/training/start",
        json={
            "experiment_id": experiment_id,
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "sport": "cricket",
        },
    )
    assert start_response.status_code == 200, start_response.text
    job_id = start_response.json()["job_id"]

    status_response = client.get(f"/training/status/{job_id}")
    assert status_response.status_code == 200, status_response.text
    assert status_response.json()["job_id"] == job_id

    metrics_response = client.get(f"/metrics/{experiment_id}")
    assert metrics_response.status_code == 200, metrics_response.text
    assert metrics_response.json()["experiment_id"] == experiment_id
