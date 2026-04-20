# Research Alignment: LiveMatch

## Purpose

This document explains how LiveMatch aligns with research literature while remaining an engineering-first, reproducible backend project.

## Primary Foundation: TimeMCL

Main paper:

- Winner-Takes-All for Multivariate Probabilistic Time Series Forecasting (TimeMCL)
- https://arxiv.org/abs/2506.05515

Directly inspired components in this repository:

1. Multi-hypothesis forecasting with shared encoder + `K` heads (`ml/model.py`)
2. Winner-Takes-All-style supervision (`wta_diverse_loss`)
3. Diversity regularization to reduce head collapse (`wta_diverse_loss`)
4. Multi-scenario inference and uncertainty summaries (`ml/inference.py`)
5. Optional PatchTST-style encoder path while preserving TimeMCL heads (`ml/model.py`)
6. Optional split-conformal interval calibration layered on TimeMCL outputs (`ml/calibration.py`, `ml/train.py`)

## Supporting References (Non-claim)

The project also borrows implementation patterns from broader forecasting practice:

- **DeepAR**: probabilistic forecasting workflow discipline (training/evaluation organization)
- **TFT**: feature-level explanation structure (factor-level reasoning objects)
- **PatchTST**: patch-based encoder organization for stronger representation in the shared encoder path
- **MLflow concepts**: runs/params/metrics/artifacts/model-version lifecycle ideas
- **CQR / conformal ideas**: interval calibration layer over multi-head outputs

Important: LiveMatch does **not** claim direct reproduction of these methods.

## What Is Standard Engineering

These components are pragmatic backend/ML engineering rather than paper novelty:

- train/val/test split handling
- checkpoint + run metadata artifacts
- latest evaluation summary artifacts for status APIs
- optional conformal calibration summaries and interval diagnostics
- experiment/dataset/job/artifact registry models
- fallback mode for reliability when training artifacts are unavailable

## What Is Project Extension

Extensions added for practical usability:

- multi-sport route abstraction (cricket-first, football extension)
- provider adapters for local/historical/mock realtime contexts
- live cricket provider path with cache/retry/fallback for daily conditioning
- API-first explanation contracts (feature, value, baseline, delta, impact, explanation)

## Reproducibility and Lifecycle

Current run artifacts:

- `ml/artifacts/time_mcl.pt`
- `ml/artifacts/training_metrics.json`
- `ml/artifacts/training_run.json`
- `ml/artifacts/evaluation_metrics.json`
- `ml/artifacts/latest_evaluation_summary.json`

Current backend lifecycle endpoints:

- datasets: `/datasets/register`, `/datasets/list`
- experiments: `/experiments/create`, `/experiments/list`
- training jobs: `/training/start`, `/training/status/{job_id}`
- metrics: `/metrics/{experiment_id}`

## Known Boundaries

- Cricket remains the strongest implemented model path.
- Football remains a cleaner extension path and can be upgraded further with a dedicated trained scoreline model.
- This repository is research-aligned engineering, not a benchmark-complete reproduction.

## Future Research-Grade Extensions

1. Calibrated scenario probability learning (beyond proxy confidence)
2. Joint multi-target training (score + player impact)
3. Better temporal encoders for longer-context forecasting
4. Standardized experiment report exports for professor/research review
