# Phase 7 Audit: Scope Cleanup and Live Prediction Priorities

## Current Strengths Supporting Forecasting Goal
- Ensemble-based multi-scenario prediction is active in backend inference.
- Conditional batting-order output (`team_a_first` / `team_b_first`) is already in response contracts.
- Completed-match evaluation and head-to-head summary are integrated.
- Live/upcoming/historical state filtering exists in catalog + dashboard.
- Provider/cache fallback paths are in place for live cricket ingestion.

## Weak or Off-Scope Areas
- Auth/login is still required for core forecasting routes and blocks direct project use.
- Frontend starts at login/signup and carries profile/logout UI not needed for a personal forecasting project.
- Test suite is auth-coupled, which adds setup noise to non-auth forecasting validation.
- Some metadata copy in dashboard overemphasizes runtime mode text instead of match forecasting context.

## Auth Value Assessment
- Current auth flow does not protect a multi-tenant product requirement in this repo.
- For this project scope, auth is low-value and increases maintenance + test friction.
- Removing auth simplifies backend contracts and makes forecasting/evaluation the first-class experience.

## Data Availability Snapshot
- Historical: cricket and football processed datasets are available locally.
- Upcoming/live: provider path exists; availability depends on external provider response and keys.
- Existing UI already handles empty upcoming/live lists, but startup flow is still auth-first.

## Context Signals Already Supported
- Venue context and matchup context from feature pipeline.
- Team/player recency context and head-to-head history.
- Batting-order conditional branch outputs for cricket.
- Evaluation context for completed historical matches.

## Phase 7 Changes
- Remove auth requirement from backend forecasting/status/research routes.
- Remove login/signup/session flow from frontend and open directly to dashboard.
- Remove dead auth client helpers and related UI text.
- Keep routes/responses for forecasting, evaluation, head-to-head, and provider status stable.
- Reword README to reflect a forecasting-first personal ML/backend scope.
