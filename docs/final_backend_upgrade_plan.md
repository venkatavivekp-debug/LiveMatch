# LiveMatch Final Backend Upgrade Plan

## Audit Summary

### What already works
- TimeMCL-centered pipeline is in place:
  - shared encoder + multi-head outputs (`ml/model.py`)
  - WTA / soft-WTA training path (`ml/train.py`, `ml/model.py`)
  - diversity-aware head separation and uncertainty summaries (`ml/inference.py`)
- Backend architecture is modular and stable:
  - clear route/service/provider separation
  - auth, catalog, prediction, insights, research endpoints
  - model status + system status visibility
- Live cricket provider exists with cache + stale fallback path.
- Fallback mode and trained mode both run without crashing.
- Test suite is present and currently passing.

### What still looks static / weak / AI-like
- Placeholder player names appear in generated cricket datasets (`* Player 1/2/...`) and can leak into outputs.
- No dedicated completed-match evaluation endpoint (forecast vs actual + best-of-K error).
- No residual-memory loop: model does not persist historical forecast errors and reuse them during inference.
- No explicit anomaly/odd-variant signal exposed from backend prediction flow.
- Explanation factors are structured but can still be generic in some paths.
- Live context is blended, but real-name resolution from live/historical rosters is not centralized.

### What should stay unchanged
- TimeMCL as the main forecasting concept and core architecture.
- Existing FastAPI route structure and authentication flow.
- Existing fallback behavior and model status reporting contract.
- Existing training/evaluation artifact structure and research endpoints.
- Existing multi-sport abstraction (cricket primary, football secondary).

## Focused Upgrade Scope

### 1) Real-name mapping and placeholder elimination
- Add a backend name resolver layer:
  - load roster priors from processed player tables
  - resolve placeholder names deterministically to real team rosters when possible
  - degrade to `"Unavailable"` when no reliable mapping exists
- Apply resolver in:
  - prediction output (`best_player`, `best_bowler`, `man_of_the_match`)
  - top player insights path
  - fallback player selection paths

### 2) Completed-match forecast-vs-actual evaluation endpoint
- Add `GET /matches/{match_id}/evaluation`:
  - reuse prediction flow for scenario generation
  - fetch actual first-innings score when available
  - compute:
    - best matching hypothesis
    - best-of-K error
    - center error
    - interval coverage flag
    - winning scenario label/index
- For upcoming/live without actual:
  - return forecast-only response with clear evaluation availability flag.

### 3) Residual memory feature layer
- Add lightweight persistence for prediction residuals (SQLite table).
- Record residuals by match context:
  - team/opponent/venue/tournament/state
  - predicted mean / interval / selected scenario
  - actual and error
- Build residual aggregates for inference-time conditioning:
  - venue residual bias
  - team-pair residual bias
  - recent under/over-prediction rates
- Inject residual context into cricket inference path as small feature adjustments + metadata.

### 4) Odd-variant / anomaly signal
- Add interpretable anomaly features:
  - z-score style deviation on current blended context vs historical baselines
  - residual-shift contribution from memory layer
- Expose in prediction metadata:
  - `anomaly_score`
  - `odd_variant_flag`
  - `residual_shift_score`
- Use anomaly signal to widen caution messaging and add a concise explanation factor.

### 5) Provider and fallback hardening for this flow
- Keep live provider stable under:
  - timeout / invalid key / malformed payload / stale cache
  - missing squads/player context
- Ensure every prediction still returns valid shape and explicit mode metadata.

### 6) Output and frontend readability cleanup
- Keep scenario naming simple (`Low`, `Baseline`, `High`, `Aggressive`, ...).
- Keep explanations numeric and concise.
- Ensure frontend emphasizes:
  - scenario cards
  - confidence
  - data mode/source/freshness
  - short uncertainty interpretation panel

## Why this upgrade is high-value
- Converts LiveMatch from forecast-only to forecast + evaluation + learning loop.
- Makes outputs credibly data-driven by removing placeholder naming artifacts.
- Improves trust with explicit residual memory and anomaly signals.
- Keeps TimeMCL central while making the backend more realistic for research demo and MS CS portfolio review.
