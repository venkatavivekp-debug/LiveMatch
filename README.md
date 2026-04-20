# LiveMatch

Probabilistic sports forecasting for cricket and football, built around multi-scenario predictions, uncertainty-aware summaries, and a simple dashboard for exploring match outcomes.

## Overview

LiveMatch forecasts matches as a range of plausible outcomes rather than a single fixed prediction. Instead of only asking “who wins?”, it tries to show how a match can evolve under different scoring conditions, how uncertain the model is, and where the forecast is stable or fragile.

I built this project to practice the full pipeline behind an ML-backed analytics product: data preparation, model inference, API design, evaluation, and frontend presentation. The goal was to make the output useful and honest, especially when the model is uncertain.

The dashboard lets you:
- select a match
- run a forecast
- inspect multiple scenarios
- compare likely outcomes
- review player signals
- check how the model performed on completed matches

## How it works

### Scenario-based forecasting

LiveMatch does not reduce a match to one headline score. It summarizes a distribution of possible outcomes into several scenario views such as Low, Baseline, High, and Aggressive. Each scenario includes:
- projected scores or goals
- a likely winner
- a relative scenario weight
- short reasoning tied to the match context

These scenarios are meant to represent different parts of the outcome space, not four arbitrary picks.

### Probabilistic forecasting

The favored team and win probability come from aggregating scenario-level outcome mass rather than relying on a single deterministic rule. The system also surfaces uncertainty through:
- scenario spread
- disagreement between scenario winners
- variance across ensemble outputs where available
- risk labeling in the final summary

### Beyond just “who wins”

For cricket, the forecasting path also considers batting-order branches. That means the same match can be evaluated under:
- one team batting first
- the other team batting first

This helps expose toss sensitivity and makes the forecast more realistic than a single static prediction.

## System architecture

### Frontend
The frontend is built with React and plain CSS. It provides:
- a forecasting dashboard
- mode switching for live, upcoming, and historical matches
- scenario cards
- uncertainty visualization
- player signal panels
- historical evaluation panels for completed matches

### Backend
The backend is built with FastAPI and Pydantic. It is responsible for:
- serving match catalogs
- normalizing live and historical match data
- calling the forecasting pipeline
- building final response contracts
- generating summary fields like favored team, win probability, score band, and risk level

### ML layer
The ML code under `ml/` handles:
- feature preparation
- model inference
- scenario extraction
- uncertainty handling
- calibration-related utilities
- historical evaluation helpers

The system supports multi-head or ensemble-style inference and uses scenario extraction logic to turn raw output distributions into readable match forecasts.

### Provider layer
For cricket, live and upcoming matches come from CricAPI:
- `/v1/matches` for scheduled and upcoming rows
- `/v1/currentMatches` for live rows

If the API key is missing or the upstream provider returns nothing usable, the app shows clean empty states instead of fabricating data.

## Features

- Multi-scenario forecasts with scenario-level winner, score, weight, and reasons
- Live, upcoming, and historical modes for match selection
- Uncertainty-aware summaries including spread, interval range, and risk level
- Batting-order branch analysis for cricket
- Player signal panels for batsmen, bowlers, and impact players when supported by data
- Historical evaluation on completed matches
- Head-to-head context in the dashboard
- Clean UI focused on forecast readability rather than static dashboards

## How to read the output

### Win probability
Win probability represents the favored team’s share of scenario weight after normalizing winners to the actual match teams. It should be read together with the risk level, not in isolation.

A low probability with a high risk level usually means:
- the outcome space is wide
- scenarios disagree
- the model sees meaningful uncertainty

### Scenario cards
Each scenario shows one plausible path the match could take. These are not duplicates of the same forecast with different labels. When the model sees meaningful variation, the scenarios can disagree on totals and even on the winner.

### Risk level
Risk is based on how stable or unstable the forecast is across scenarios and branches. A high-risk forecast means the model still sees a likely edge, but not a clean, dominant one.

### Historical evaluation
For completed matches, LiveMatch compares the forecast against the actual outcome. This helps show whether:
- the predicted winner matched reality
- the real score stayed inside the forecast range
- one scenario aligned better than the others

## Project structure

```text
backend/    FastAPI app, routes, schemas, services
frontend/   React dashboard and CSS styling
ml/         feature logic, inference, evaluation, calibration
data/       processed datasets and supporting data files
docs/       architecture notes, audits, and project writeups
scripts/    setup and refresh helpers
