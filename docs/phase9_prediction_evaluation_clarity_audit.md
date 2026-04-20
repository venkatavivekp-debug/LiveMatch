# Phase 9 Audit: Prediction and Evaluation Clarity

## Current gaps
- The top section reads like a generic analytics dashboard, not a forecast-first product.
- Forecast confidence is distributed across scenario cards; there is no single primary "who is favored" summary.
- Batting-order branch impact exists in data but is buried inside each scenario card.
- Historical evaluation is present but visually mixed with forecasting content and not interpreted in one clear line.
- Status text (live sync, coverage counts, fallback labels) still competes with core forecast/evaluation information.

## Why evaluation is not obvious enough
- Users must parse multiple lines (actual, best scenario, error, range) without a verdict sentence.
- There is no compact yes/no block for winner correctness and range correctness at first glance.

## Planned adjustments in this pass
- Backend:
  - Add `forecast_summary` to prediction response with:
    - `favored_team`
    - `favored_team_confidence`
    - `predicted_band_low`
    - `predicted_band_high`
    - `key_risk`
  - Add evaluation interpretation field (`evaluation_summary`) in evaluation output.
- Frontend:
  - Add a primary forecast summary card at top of prediction view.
  - Add a compact batting-order impact card (team A first vs team B first outcomes).
  - Upgrade historical evaluation section into a clearer evaluation panel with verdict line.
  - De-emphasize status noise in the main dashboard header and filter area.
  - Keep scenario/player cards but reduce clutter and improve scanability.
- Tests:
  - Validate forecast summary fields in `/predict`.
  - Validate evaluation summary in historical evaluation endpoint.
