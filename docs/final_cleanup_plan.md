# LiveMatch Final Cleanup Plan

## What is cluttered
- Dashboard shows too many secondary details in primary cards.
- Interpretation block adds noise and takes space from useful match context.
- Raw backend error details can leak into UI.
- Historical pair analysis is missing a dedicated head-to-head + evaluation view.

## What is unnecessary
- Long filler phrasing in user-facing explanation copy.
- Displaying internal/debug-like signals in main UI sections.
- Repeating low-value labels across cards.

## What is strong and should stay
- TimeMCL-centered multi-scenario forecasting pipeline.
- Trained vs fallback mode switching.
- Live/hybrid/historical data mode handling.
- Residual/anomaly metadata generation and evaluation endpoint.

## What will be cleaned up
- Simplify dashboard hierarchy and remove the Interpretation panel.
- Add head-to-head history with compact predicted-vs-actual summaries.
- Improve match selection/prediction error handling.
- Hide internal reason tags from primary UI cards.
- Tighten football player fallback presentation.

## User-facing fixes
- Friendly missing-match/empty-state messages.
- Clear “Last 5 meetings” panel with actual result + winner.
- Clear “Best match to actual / Error / In range” summaries for historical rows.
