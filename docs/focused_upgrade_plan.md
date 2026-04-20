# Focused Upgrade Plan: TimeMCL-Centered Backend

## Why This Plan

LiveMatch already has a strong TimeMCL-oriented backbone and a working backend/service architecture.
This pass avoids broad model expansion and applies only targeted upgrades aligned with:

1. TimeMCL (main forecasting idea)
2. PatchTST-style encoder ideas (encoder-only improvement)
3. TFT-inspired explanation structuring (interpretability organization only)
4. CQR-inspired conformal calibration (interval reliability layer)
5. MLflow-style registry/reporting concepts (ops visibility only)

No baseline zoo, no architecture reset, and no claim of full paper reproductions.

## Current Implementation Snapshot

### Already implemented and should remain

- FastAPI backend with clean route/service/schema structure
- JWT auth, catalog/predict/insight/research workflow endpoints
- TimeMCL-like multi-head model with WTA + diversity regularization
- trained vs fallback behavior with robust setup scripts
- experiment/job/artifact/metric persistence models and endpoints
- evaluation metrics and status summaries
- backend/API smoke tests

### Existing gaps to upgrade (focused)

- Encoder remains mostly MLP-style; no patch-based representation option
- Uncertainty intervals are scenario-derived but not explicitly conformal-calibrated
- Explanation format is structured but inconsistent across scenario/player factors
- `/model/status` can better surface calibration/model artifact discoverability

## What Will Be Upgraded

## 1) TimeMCL Core (preserved, not replaced)

- Keep shared encoder + K heads + WTA + diversity regularization
- Keep scenario ranking and uncertainty summaries
- Maintain backward-compatible API response contracts

## 2) PatchTST-style Encoder Option (encoder-only)

- Add optional patch-based sequence encoder path in `ml/model.py`
- Keep existing MLP encoder path as default for compatibility
- Add config-driven selector (e.g., `encoder_type=mlp|patch`)
- Persist encoder metadata in checkpoint for correct reload in inference

Note: This is a PatchTST-style inspiration for representation, not a full PatchTST reproduction.

## 3) CQR-inspired Conformal Calibration Layer

- Add split-conformal interval calibration from validation residuals in training
- Save calibration metadata/artifacts alongside training run metadata
- In inference, optionally apply calibrated interval expansion/contraction around scenario range
- Expose calibration summary in prediction metadata and model status when available

Note: This is conformal calibration on top of TimeMCL outputs, not replacing TimeMCL with quantile regression.

## 4) Explanation Engine Consistency

- Centralize explanation factor construction for deterministic structure:
  - `feature`, `value`, `baseline`, `impact`, `delta`, `unit`, `explanation`
- Improve consistency for scenario + player explanations
- Keep wording stable and data-grounded

## 5) Model Status / Registry Visibility

- Improve `/model/status` to include calibration and encoder details when available
- Ensure artifact paths and latest summary are easy to locate
- Keep research workflow endpoints unchanged and compatible

## 6) Evaluation and Artifact Discoverability

- Preserve existing metrics (MAE, RMSE, best-of-K, diversity, coverage, interval width, CRPS approx)
- Add calibration diagnostics where available
- Keep JSON artifacts simple and easy to find under `ml/artifacts/`

## 7) Tests

- Keep current backend smoke coverage
- Add calibration utility tests
- Add tests ensuring fallback and calibrated status paths remain stable

## What Will Not Be Done

- No new unrelated forecasting model families
- No major frontend feature expansion
- No broad architectural rewrite
- No overclaim of full TimeMCL/PatchTST/TFT/CQR implementations

## Expected Outcome

LiveMatch remains a TimeMCL-centered backend-heavy forecasting system, upgraded with:

- optional patch-based encoder representation
- optional conformal interval calibration
- clearer structured explanations
- stronger model status and artifact transparency

while preserving working behavior and API compatibility.
