# LiveMatch Current Architecture Audit (April 15, 2026)

## Scope

This audit reviews the existing in-repo implementation before additional changes:

- backend API routes, schemas, services, and persistence
- ML model/training/inference/evaluation pipeline
- fallback vs trained mode behavior
- experiment/dataset/job/metrics tracking
- tests and docs quality

## Current Strengths

### 1. Backend structure is already modular and practical

The backend uses a clear FastAPI organization:

- `api/routes` for route handlers
- `schemas` for Pydantic contracts
- `services` for orchestration/business logic
- `services/providers` for data/realtime adapters
- `db` for SQLAlchemy setup and models

This is strong for a personal MS-level backend-heavy project and avoids monolithic route logic.

### 2. Core product and research routes exist

The following route families are present and wired:

- auth: `/auth/register`, `/auth/login`
- catalog: `/sports`, `/tournaments`, `/matches`
- prediction: `/predict`, `/predict/batch`, `/forecast/scenarios`, `/forecast/uncertainty`
- insights/status: `/insights/live`, `/players/top`, `/model/status`
- research workflow: `/datasets/register`, `/datasets/list`, `/experiments/create`, `/experiments/list`, `/training/start`, `/training/status/{job_id}`, `/metrics/{experiment_id}`

### 3. TimeMCL-inspired modeling is meaningfully implemented

Existing ML code already includes:

- shared encoder + K heads (`ml/model.py`)
- WTA-style objective (`wta_diverse_loss`)
- diversity regularization term
- multi-scenario inference outputs with uncertainty summary

This is already above a toy single-point predictor.

### 4. Fallback mode is stable and explicit

Fallback behavior is robust across ML + backend:

- no hard dependency on torch for runtime inference
- backend can serve predictions in fallback mode
- setup scripts support optional torch installation

### 5. Basic experiment persistence exists

SQLAlchemy models and endpoints already track:

- datasets
- experiments
- training jobs
- model artifacts
- evaluation metrics rows

This provides a useful base for MLflow-style lifecycle concepts without forcing heavy infra.

### 6. Baseline tests exist and pass

Current tests cover:

- API smoke/auth behavior
- predict contract shape
- fallback mode execution
- research endpoint smoke
- one ML metrics utility test

## Current Weaknesses / Gaps

### 1. Training pipeline is not fully research-rigorous yet

Current `ml/train.py` uses train/val split only.

Gaps:

- no explicit test split artifact
- no reproducible split metadata saved as a structured run manifest
- limited run-level params/metadata persisted for auditability

### 2. Model status endpoint is not rich enough for research demos

`/model/status` currently omits useful fields such as:

- checkpoint timestamp/version info
- latest evaluation summary metrics
- active model artifact metadata (when available)

### 3. Evaluation pipeline can be better integrated with training lifecycle

`ml/evaluate.py` computes useful metrics but integration is minimal.

Gaps:

- no standardized `latest` model metrics artifact with timestamp + run linkage
- no robust "latest summary" contract consumed by backend status/reporting

### 4. Registry layer is useful but still shallow

Current experiment/job persistence is functional but not yet strongly MLflow-style in schema semantics:

- training jobs store request/output JSON, but no explicit tracked parameter/metric snapshots normalized at job completion
- artifact metadata can be enriched with checkpoint hash, training sample sizes, split ratios, and run tags

### 5. TimeMCL alignment can be documented more explicitly in code/docs

TimeMCL mapping is present but should be made more explicit in:

- model/loss comments (hard WTA vs optional soft-WTA extension)
- training metadata and docs labeling paper-inspired vs engineering extension

### 6. Explanation consistency can be tightened

Explanations are structured and mostly good, but consistency can improve for deterministic wording and explicit baseline/delta coverage across all branches.

## Upgrade Plan (In Place)

### Phase 1: Training/Evaluation hardening

- upgrade `ml/config.py` with explicit split and run metadata knobs
- add train/val/test split flow in `ml/train.py`
- save a run metadata artifact (`training_run.json`) including params, splits, and best epoch
- produce stable `latest_evaluation_summary.json` for backend consumption

### Phase 2: TimeMCL core clarity

- extend WTA loss with optional soft-WTA temperature path (without breaking hard-WTA default)
- keep direct TimeMCL-inspired elements explicit in comments/docs

### Phase 3: Model status and registry enrichment

- expand `/model/status` response with:
  - model version/checkpoint metadata
  - latest evaluation metrics snippet
  - active artifact reference
- enrich research service artifact registration metadata

### Phase 4: Explanation consistency and deterministic phrasing

- normalize explanation factor shape and wording templates
- ensure baseline/delta/unit present where meaningful

### Phase 5: Tests/docs tightening

- add tests for model-status enhanced fields
- add tests for train/eval artifact shape utilities where feasible
- strengthen README + docs on Research Alignment with explicit honesty about:
  - TimeMCL-inspired parts
  - standard forecasting engineering
  - project-specific extensions

## Non-Goals for this pass

- no rewrite from scratch
- no heavy frontend refactor
- no false claim of exact TimeMCL/DeepAR/TFT/PatchTST reproduction
- no unnecessary infra (e.g., full MLflow server deployment)

## Expected Outcome

After this pass, LiveMatch should feel more like a backend-first research demo platform:

- stronger experiment-grade ML lifecycle
- clearer TimeMCL alignment in both code and reporting
- richer model status visibility
- cleaner reproducibility and evaluation artifacts
- preserved existing functionality and endpoints
