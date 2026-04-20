# LiveMatch Model Truth Audit

Date: 2026-04-17

## What Is Truly Model-Driven Today
- `ml/model.py` + trained checkpoint path in `ml/inference.py` is a real TimeMCL-style multi-head predictor.
- Trained path supports:
  - shared encoder + K heads
  - diversity post-processing
  - scenario probabilities/confidence proxy
  - uncertainty interval summary
- Residual memory, anomaly signal, and evaluation endpoints are wired and usable.

## What Is Heuristic/Templated Today
- `PredictionService._heuristic_fallback` uses fixed score ladders and fixed player names.
- Cricket branch outputs in fallback are generated from static arithmetic offsets.
- Football fallback scorelines are fixed templates, not learned from local match history.
- `InsightsService.live_insights` falls back to historical matches, then still renders “Live Context”, which is misleading.
- Tournament list is static catalog-driven; categories can appear even when no dataset rows exist.

## Misleading Outputs
- Scenario branches can look deterministic (formatting math) when model is unavailable.
- “Live” cards can appear for historical-only conditions through context enrichment fallback.
- Some fallback player outputs look plausible but are not tied to current match evidence.

## Simplification/Correction Plan
1. Replace fallback scenario generation with data-driven historical lookup quantiles (no fixed ladders).
2. Keep TimeMCL trained path untouched; make fallback explicitly “data-driven fallback” and honest.
3. Make live insights strict:
   - no historical fallback inside live panel
   - return empty cards + clear metadata when no real live/upcoming feed rows exist.
4. Reduce synthetic player fallback:
   - use only historical player tables when available
   - avoid fabricated cross-team names in primary prediction path.
5. Keep completed-match evaluation central and visible in API/UI.

## New Clean Modeling Rule
- Trained mode: TimeMCL multi-head prediction is primary.
- Fallback mode: historical-data-driven quantile scenarios only.
- No synthetic live injection in primary user flow.
