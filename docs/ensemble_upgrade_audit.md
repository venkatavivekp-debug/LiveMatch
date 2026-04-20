# Ensemble Upgrade Audit

## Current Prediction Flow (as-is)
- API routes call `PredictionService.predict`, which enriches payload with live context + residual context, then calls `ml.inference.LiveMatchPredictor`.
- `LiveMatchPredictor` loads one checkpoint (`ml/artifacts/time_mcl.pt`) with shared encoder + K heads.
- Cricket scenarios are built by sorting head outputs, mapping labels (`Low/Baseline/High/Aggressive`), then formatting branch views (`team_a_first`, `team_b_first`).
- Fallback path is data-driven but still heuristic-heavy when model artifacts are missing.
- Evaluation exists for completed matches (`/matches/{id}/evaluation`) and compares predicted heads against actual.

## Weak Behaviors Still Present
- Scenario spread often reflects head ordering and formatting logic more than a stable predictive distribution.
- Single-model uncertainty can collapse into small variations; scenarios can look deterministic.
- Some scenario confidence/probability values are derived from post-processing heuristics.
- Winner behavior can look fixed across scenarios for many matches because diversity source is limited to one trained model.
- Redundant/stale code before this pass:
  - `ml/config.py` had unused legacy `SCENARIO_LABELS`.
  - `ml/inference.py` had interpolation-based scenario projection from one model path and branch winner shortcuts tied to probability thresholds.
  - Scenario shaping relied on equal-quantile slicing that produced near-uniform scenario weights.
  - Fallback inference path used fixed score ladders when no nearest-neighbor samples existed.

## Upgrade Direction (in-place)
- Keep route contracts and response shape intact.
- Replace single-model inference core with deep ensemble inference:
  - train `N` TimeMCL members with different seeds/initialization (and optional bootstrap sampling);
  - aggregate all member outcomes into one distribution of `(team_a_score, team_b_score, winner_prob_a)`;
  - generate `K` scenarios from distribution bins/quantiles (not manual offsets).
- Keep existing fallback path, but remove now-redundant manual scenario expansion in trained path.
- Preserve completed-match evaluation API and strengthen artifacts with ensemble metrics.

## What Stays Unchanged
- Backend endpoints and auth flow.
- Prediction payload top-level structure (`match`, `predictions`, `uncertainty`, `metadata`, player blocks).
- Head-to-head and completed-match evaluation integration in backend.
- Frontend contract expectations for scenario cards.

## Robustness Checks Required
- Train/infer with missing torch -> graceful fallback.
- Infer with missing/partial ensemble checkpoints -> continue with available members; fallback if none.
- Preserve deterministic output shape even with sparse features/live data failures.
- Keep evaluation artifact generation stable and easy to inspect under `ml/artifacts/evaluation/`.
- Ensure old projection helpers/legacy constants are removed without breaking API response compatibility.
- Ensure completed-match evaluation picks best scenario using both innings when both actual scores are available.
