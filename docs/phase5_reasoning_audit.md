# Phase 5 Reasoning Audit

## Current flow
- Scenarios are generated in `ml/inference.py` from ensemble/conditional outputs (`_predict_conditional_raw_outcomes`, `_distribution_conditional_scenarios`).
- Scenario reasons are produced by `_cricket_scenario_reasons` and normalized by `_normalize_reason_factors`.
- Player blocks are built in `_predict_cricket_players` and `_predict_football_players`.
- Fallback responses are assembled in `backend/app/services/prediction_service.py` (`_fallback_cricket_prediction`, `_fallback_football_prediction`, `_fallback_players`).
- Frontend renders `reason` arrays directly in `frontend/src/components/PredictionDashboard.jsx`.

## Current flaws
- Scenario reasons are mostly feature dumps with repeated phrasing (`"...trend."`, `"...baseline."`) and often overlap across labels.
- Scenario reason count can exceed what UI needs and includes low-value fields in some paths.
- Fallback reason text remains numeric-debug style (`feature: value`) and leaks into user-facing cards.
- Cricket player reasons are generic and repetitive across candidates; confidence formatting is noisy.
- Football player role quality is weaker in sparse-data paths and can default to shallow fallback wording.
- UI reason rendering prioritizes raw feature/delta display over concise outcome reasoning.

## Misleading or weak logic
- Reason generation is only loosely tied to scenario context (Low/Baseline/High/Aggressive) even when model outputs differ.
- Player selection is mostly rank-based but explanation quality does not reflect role-specific strength cleanly.
- Some fallback branches still produce synthetic-looking reason payloads even when the name output is valid.

## Phase 5 fix plan
1. Replace scenario reason generation with compact, scenario-aware, feature-driven signals (max 3, deduped).
2. Keep scenario scores/winners fully model-derived; remove any remaining reason templates that look static.
3. Rebuild player reason generation for cricket/football into role-specific top-2/3 candidates with max-2 concise reasons.
4. Simplify fallback reason payloads to match production style and avoid stat-dump labels.
5. Update frontend reason rendering to prefer concise explanation text and hide noisy raw fields.
6. Add tests for reason uniqueness/limits and player contract robustness under sparse data.
