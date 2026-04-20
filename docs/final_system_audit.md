# Final System Audit

## End-to-End Trace (Prediction Path)
1. **Data source**: `CatalogService.list_matches()` loads historical rows from `LocalDemoProvider` and overlays live/upcoming rows from `CricAPIRealtimeProvider`.
2. **Feature generation**: `PredictionService.predict()` calls `LiveMatchPredictor._feature_dict_from_match()`, which builds historical features and optionally blends live context.
3. **Model inference**: `LiveMatchPredictor._predict_conditional_raw_outcomes()` runs ensemble/single model per batting-order branch.
4. **Scenario generation**: `_distribution_conditional_scenarios()` clusters predictive samples into K scenarios, then `_predict_cricket()` builds per-scenario branch outcomes.
5. **Reasoning generation**: `_cricket_scenario_reasons()` maps feature deltas to short scenario reasons.
6. **API response shaping**: `PredictionService._finalize_prediction_contract()` normalizes scenario fields and confidence for `POST /predict`.
7. **Frontend rendering**: `PredictionDashboard` shows forecast summary, batting-order outcomes, scenario cards, players, and historical evaluation.

## Integrity Gaps Found

1. **Synthetic upcoming rows were still injected**
   - File: `backend/app/services/catalog_service.py`
   - Path: `_merge_live_rows()` -> `_historical_upcoming_fallback()`
   - Problem: when live provider was unavailable, upcoming rows were fabricated from historical pairings and shown as scheduled matches.
   - Impact: upcoming catalog could look populated without real provider-backed fixtures.

2. **Confidence values were being post-processed heavily after inference**
   - File: `backend/app/services/prediction_service.py`
   - Path: `_finalize_prediction_contract()`
   - Problem: confidence could be materially altered by formatting logic that used spread penalties and winner-mix penalties.
   - Impact: displayed confidence was not always a direct reflection of inference output.

3. **Noisy fallback wording leaked into metadata summaries**
   - File: `backend/app/services/prediction_service.py`
   - Path: `_resolve_live_context()`
   - Problem: metadata used labels like `historical fallback` that added noise and implied a special mode in normal no-live situations.
   - Impact: inconsistent UX signaling across historical/upcoming views.

## Consistency Checks

- Scenario generation is distribution-driven (cluster/quantile over predicted samples), not static arithmetic expansion.
- Winner selection in branch outputs is score-based with tie resolved by winner probability.
- Historical evaluation runs through the same prediction path via `PredictionService.evaluate_match()`.
- UI consumes API outputs directly; no frontend-side synthetic scenario generation was found.

## Corrections Applied In This Pass

1. Remove synthetic upcoming fallback injection from catalog merge.
2. Simplify confidence normalization to preserve model/inference confidence semantics.
3. Simplify live-context fallback wording in metadata to reduce noisy mode labels.
4. Update tests to reflect truthful upcoming behavior (empty when provider has no usable rows).
