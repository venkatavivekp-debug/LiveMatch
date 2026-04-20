# Evaluation Artifacts Guide

This project keeps model/evaluation outputs under `ml/artifacts/`.

## Key Files

- `time_mcl.pt`
  TimeMCL-inspired checkpoint with model state + training metadata, including:
  `encoder_type`, patch-encoder config (when used), and conformal calibration payload.

- `training_metrics.json`
  Per-epoch training/validation metrics history.

- `training_run.json`
  Reproducibility metadata for the latest training run:
  params, split ratios, sample counts, best epoch, test metrics, and calibration summary.

- `evaluation_metrics.json`
  Full evaluation output from `python -m ml.evaluate`.

- `latest_evaluation_summary.json`
  Latest compact summary intended for backend status reporting (`/model/status`).
  Includes `encoder_type`, status metrics, and conformal calibration metadata.

## Backend Visibility

`GET /model/status` includes:

- `latest_evaluation_summary`
- `artifact_paths`
- `encoder_type` / `encoder_config`
- `calibration`

so the latest metrics and file locations are discoverable from the API.

## Recommended Workflow

1. `python -m ml.train`
2. `python -m ml.evaluate`
3. check `GET /model/status` for mode/version/summary

This keeps trained-vs-fallback behavior explicit while maintaining stable local development flow.
