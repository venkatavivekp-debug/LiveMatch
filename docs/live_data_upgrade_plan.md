# Live Data Upgrade Plan (Backend-First, TimeMCL-Centered)

## Current Snapshot (What Already Exists)

LiveMatch already has a solid backend/ML foundation:

- TimeMCL-style forecasting core with shared encoder, multi-head hypotheses, WTA/soft-WTA logic, diversity regularization, calibration metadata, and structured scenario explanations.
- FastAPI route/service structure with auth, catalog, prediction, insights, and research workflows.
- Provider abstraction with:
  - `SportsDataProvider` (catalog/historical-style data)
  - `RealtimeContextProvider` (currently mock context only)
- Predictor runtime stability with trained vs fallback mode behavior.
- `/model/status` with model artifact visibility and runtime metadata.

## Static Limitations Today

The system is still mostly static for match-day freshness:

- Catalog is primarily local processed CSV + demo fixtures.
- Realtime provider is mock-only; no real cricket API conditioning path.
- Prediction conditioning is mostly historical feature lookup/derived baselines.
- No robust live-provider retry/cache/fallback orchestration layer.
- No first-class live refresh flow (scheduler/CLI/admin refresh).
- Status does not yet clearly expose live data mode/freshness/provider health as a first-class data plane.

## Focused Upgrade Direction

This upgrade will **not** replace TimeMCL.
It strengthens data conditioning and backend reliability around the existing model.

### A) Keep unchanged (TimeMCL Core)

- Multi-head architecture (`K` diverse hypotheses)
- WTA / soft-WTA supervision
- Diversity regularization
- Scenario ranking and uncertainty outputs
- Structured explanation objects

### B) Upgrade (Data + Backend Infra)

1. Add robust live cricket provider architecture:
   - historical provider
   - live provider (CricAPI-style integration)
   - fallback/mock provider
2. Add resilient provider operations:
   - timeout
   - retries with backoff
   - cache with stale fallback
   - health checks
3. Add blended feature conditioning:
   - live context features + historical baselines
   - recency weighting into model features
4. Add explicit data-plane metadata in prediction responses:
   - `data_mode` (LIVE/HYBRID/HISTORICAL/FALLBACK)
   - `provider_used`
   - `last_refresh_time`
   - `freshness_seconds` / summary
5. Add stronger status and refresh UX:
   - richer `/model/status`
   - new `/system/status`
   - refresh trigger path (`POST /admin/refresh-live-data`)
   - CLI refresh command
6. Add backend tests for live provider fallback/cache/mode switching.
7. Add lightweight frontend clarity:
   - data mode/source badges
   - last updated/freshness
   - short interpretation panel for scenarios/uncertainty.

## Research Boundary (Honest Positioning)

- TimeMCL remains the primary forecasting model idea.
- Live-provider and blending logic are engineering infrastructure improvements.
- This is an applied research-aligned forecasting backend, not a full paper reproduction of external live APIs or data products.

## Expected Outcome

After this pass, LiveMatch should remain TimeMCL-centered while becoming materially more realistic for day-to-day usage:

- fresher conditioning for current/upcoming cricket fixtures
- robust fallback behavior when live APIs fail
- transparent model/data mode reporting
- stronger trust in forecast context and uncertainty communication
