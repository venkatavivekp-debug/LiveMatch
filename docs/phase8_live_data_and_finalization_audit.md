# Phase 8 Audit: Live Data + Finalization

## What currently works
- Backend already uses a real CricAPI provider path with timeout/retry/cache support.
- Catalog merges historical and realtime rows and preserves response compatibility.
- Forecast output already contains multi-scenario results, player candidates, and uncertainty metadata.
- UI has cleaner empty states and can show fallback rows as fallback.

## What is still weak
- Realtime provider relies mainly on `currentMatches`; upcoming fixtures can be missed when they only appear in `matches`.
- Invalid matchup rows are not fully filtered (`Tbc`/`TBD` style teams can leak).
- Live/upcoming row normalization is strict in some places and weak in others (state/source mapping inconsistencies).
- Scenario reasons are still partially template-like and can repeat across scenario bands.
- Player reasons can still be repetitive when fallback or weak-signal pools are used.
- Response has no single concise forecast summary line for quick trust/readability.

## Fallback and filtering observations
- Fallback scheduling exists in catalog merge and is labeled as `data_source: fallback`.
- Fallback currently triggers mainly on degraded/unavailable health; no-usable-row handling should be explicit and deterministic.
- Tournament alias mapping is present but should stay aligned with live/upcoming state parsing and endpoint differences.

## Files to change in this pass
- `backend/app/services/providers/cricapi_realtime_provider.py`
  - dual endpoint fetch, stronger row normalization, TBC/TBD filtering, dedupe, state cleanup
- `backend/app/services/catalog_service.py`
  - fallback trigger policy cleanup when provider yields no usable upcoming rows
- `ml/inference.py`
  - sharpen scenario reasons, improve concise player reasons, add one final match insight
- `backend/app/schemas/prediction.py`
  - optional `match_insight` response field
- `frontend/src/components/PredictionDashboard.jsx`
  - show final match insight and keep runtime/fallback labels subtle
- `frontend/src/styles.css`
  - small style support for insight block
- tests under `backend/tests` and `ml/tests`
  - provider normalization/filtering, fallback trigger behavior, concise reasoning, final insight presence
- `README.md`
  - concise live-data/fallback behavior and safe `LIVE_CRICKET_API_KEY` setup wording
