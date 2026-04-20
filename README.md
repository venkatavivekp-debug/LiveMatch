# LiveMatch

Probabilistic sports forecasting for cricket (main) and football: multiple scenarios per match, uncertainty on the surface, and a small dashboard to explore results.

## Overview

LiveMatch predicts how a match might play out as a **set of plausible futures**, not a single headline number. I built it to practice end-to-end ML plumbing—data, inference, API, UI—and to keep the math honest about disagreement and spread.

The UI is a forecasting dashboard: pick a match, run a forecast, read scenarios, risk, and (for finished games) how the model did.

## How it works

**Scenario-based forecasting**  
The model doesn’t just emit one winner. It summarizes a distribution into several scenarios (Low, Baseline, High, Aggressive-style labels). Each scenario has scores (or goals), a winner, weights, and short reasons. They’re meant to represent different parts of the outcome space, not four arbitrary picks.

**Probabilistic thinking**  
Win probability and “who’s favored” come from **aggregating scenario-level mass** (probabilities / weights), not from a single deterministic rule. The summary also carries **uncertainty**: spread across scenarios, disagreement between winners, and ensemble variance where applicable.

**Not only “who wins”**  
Cricket paths include **batting-order branches** (who bats first vs chase), so the same match can look different under toss-sensitive logic. Risk level and copy try to reflect wide spreads and conflicting scenarios, not just a confidence badge.

## System architecture

**Frontend** — React, plain CSS for layout and light motion. Calls the FastAPI backend; shows hero forecast, scenario cards, charts, player blocks, and historical evaluation when relevant.

**Backend** — FastAPI, Pydantic models for requests/responses. A prediction service builds the forecast payload, normalizes scenarios, and attaches a **forecast summary** (favored team, win probability from scenario weights, risk text, bands). Catalog routes merge **historical** data with **live provider** rows when you ask for live or upcoming cricket.

**ML layer** — Training and inference under `ml/`: ensemble / multi-head style inference, scenario extraction from samples or fallbacks, cricket branching, calibration-style intervals where wired in, and evaluation helpers for backtests.

**Provider layer** — Cricket live/upcoming via **CricAPI** (`/v1/matches` for scheduled-style rows, `/v1/currentMatches` for live). Missing API key or upstream errors surface as empty or cache-backed behavior, not fake fixtures.

## Features

- Multi-scenario forecasts with per-scenario scores, winner, weight, and reasons  
- **Live**, **upcoming**, and **historical** modes for match lists (cricket live feed depends on the provider)  
- Uncertainty summary (spread, intervals) and risk level in the API and UI  
- Player signal blocks when the model exposes strong enough candidates  
- Historical **evaluation**: winner/range checks and scenario fit for completed matches  
- Head-to-head history in the sidebar for context  

## Example output (how to read it)

**Win probability**  
In the API summary, win probability is the **favored team’s share of scenario weight** between the two sides (after normalizing winners to the match’s team names). The ring in the UI is tied to that when the backend summary is present. A separate “confidence” style number can reflect how tight or messy the overall forecast is—don’t confuse the two if you dig into the JSON.

**Scenarios**  
Each row is one plausible path. They’re ordered and labeled so you can compare totals and winners side by side. They will **disagree** on purpose when the model sees real spread.

**Why it isn’t deterministic**  
Different samples and branches don’t collapse to one score. If two scenarios pick different winners, that’s a feature of the setup, not a bug—the summary and risk fields are there to make that visible.

## Setup

**Clone**

```bash
git clone <your-repo-url>
cd LiveMatch
```

**Backend**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir .
```

Run from the `backend` directory so imports resolve. Default API is often `http://localhost:8000`.

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Point the UI at the API (see env below).

**Environment variables**

Copy `backend/.env.example` to `backend/.env` and adjust.

| Variable | Role |
|----------|------|
| `LIVE_CRICKET_API_KEY` | CricAPI key for live/upcoming cricket rows. Empty = no live calls; lists may be empty for those modes. |
| `REALTIME_PROVIDER` | e.g. `cricapi` for real provider; mock options exist for local dev. |
| `DATA_PROVIDER` | Historical catalog source (e.g. `local-demo`). |

Frontend: `frontend/.env.example` — set `VITE_API_BASE_URL` if the API isn’t on localhost:8000.

**ML / data**  
Processed features and checkpoints live under `data/processed/` and `ml/artifacts/` as set up in your clone. Training is optional if you only run inference with bundled artifacts.

## Notes and limitations

- **External API**: Live and upcoming cricket depend on CricAPI (or your configured provider). No key, rate limits, or bad responses mean fewer or no rows—**the app won’t invent matches** to fill the UI.  
- **Empty lists**: In live or upcoming mode you may see zero matches. That’s expected when the provider returns nothing usable or filters remove incomplete rows (TBC, finished games, etc.).  
- **Not betting advice**: This is a learning and demo project. Don’t use it for wagering or as production tipping infrastructure.  
- **Football** is supported in the stack but cricket is the deepest path (branching, provider).  

## Future work

- Sharper calibration reporting in the UI  
- Clearer inline labels for cricket scenario scores vs innings semantics  
- Optional Docker compose for one-command local run  
- More tests around provider edge cases when the API shape drifts  

---

If something breaks after a provider or dependency update, check the backend logs first, then confirm `.env` and that `GET /matches` returns rows for your sport and mode.
