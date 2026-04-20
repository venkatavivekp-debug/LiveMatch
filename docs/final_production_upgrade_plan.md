# Final Production Upgrade Plan

## Current strengths
- Backend architecture is already modular (`api`, `services`, `providers`, `schemas`, `db`).
- TimeMCL-style multi-head flow is implemented with WTA-oriented training/inference paths.
- Live provider abstraction, cache, fallback, residual memory, anomaly signal, and evaluation routes already exist.
- Frontend has core workflow (auth, match select, predict, history, evaluation) and is wired to backend APIs.

## Current weaknesses
- Primary data path still allows synthetic/demo leakage (demo live fixtures, mock defaults).
- Processed player tables include placeholder names (`Team Player 1` style).
- Scenario payloads are still mostly scalar-first; full match branch outcomes are not explicit.
- UI is still dense in some sections and exposes too much low-value context at once.
- Scenario labels include synthetic suffixes (`High+`, `Aggressive+`) in some fallback/model paths.

## What will be upgraded
1. **Data path cleanup**
   - Remove automatic demo fixture injection from catalog provider.
   - Keep live provider graceful fallback without exposing demo/mock content in normal flow.
   - Regenerate processed football/cricket player data with real names.

2. **TimeMCL scenario contract upgrade**
   - Keep existing `predictions` contract for backward compatibility.
   - Add structured full-match scenario branches:
     - cricket: `team_a_first` and `team_b_first` with projected scores and winner.
     - football: explicit `likely_result` with scoreline branch.
   - Keep labels user-clean: `Low`, `Baseline`, `High`, `Aggressive`.

3. **Player output cleanup**
   - Eliminate placeholder player names from model outputs.
   - Return compact top candidate lists while preserving existing fields.

4. **Frontend cleanup**
   - Reduce dense context sections and prioritize: match, status, scenarios, players, compact history/evaluation.
   - Collapse head-to-head to latest meeting by default with “view more”.
   - Hide technical/debug metadata from primary cards.

5. **Robustness + verification**
   - Harden missing match / missing live / missing player edge cases.
   - Keep route compatibility and run backend tests, ML tests, and frontend build.

## Keep unchanged
- TimeMCL-centered architecture (shared encoder + K heads + WTA logic).
- Trained vs fallback mode switch.
- Existing core routes and auth flow.
- Residual memory + anomaly signal logic as backend features.

## Cricket vs football handling
- Cricket remains strongest path with live/hybrid/historical conditioning.
- Football remains secondary but real-data-backed where available; degrade cleanly when player-level context is sparse.

## Scope discipline
- No architecture rewrite.
- No new model families.
- No placeholder output in primary user path.
- No verbose AI-style copy in UI/docs.
