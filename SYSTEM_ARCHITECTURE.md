# Intent2Model — System Architecture & Flow (Detailed)

This document explains how the Intent2Model AutoML platform works end-to-end: frontend wizard, backend APIs, training pipeline, prediction flow, system bias & evaluation design, and a **Logs and Changes** section kept up to date with notable updates.

---

## 1. Overview

**Intent2Model** is an LLM-guided, epistemically honest AutoML platform that:

1. Lets you **upload a CSV** and define **intent** (which column to predict, task type).
2. **Trains** one or more models autonomously (planning, preprocessing, model comparison, holdout validation, quality gates, auto-repair).
3. **Caches** the best model and metadata per run so you can **predict** (with uncertainty), **download** notebook/README/report, and **switch** which model is active for a run.

Design principles:

- **Frontend**: Next.js (React) — wizard steps Upload → Define Intent → Train → Deploy & Stats. HTTP (REST) + optional WebSocket for live run logs.
- **Backend**: FastAPI — uploads, train, predict, downloads, health, run state. Training is delegated to an **autonomous executor** that uses **agents** (planner, compiler, diagnosis, repair) and **ML** (profiler, pipeline builder, trainer, evaluator).
- **System bias & evaluation**: Holdout validation, aggressive feature pruning, opinionated model order with complexity penalty, target transformation (regression), residual-based retry, locked classification metric, and prediction uncertainty are built in to reduce real-world error and over-trust in CV.

---

## 2. Project Structure (Detailed)

```
intent2model/
├── frontend/
│   ├── app/
│   │   ├── layout.tsx, page.tsx, globals.css
│   │   └── favicon.ico
│   ├── components/
│   │   ├── intent-2-model-wizard.tsx   # Main wizard (upload, intent, train, deploy, predict)
│   │   ├── intent-2-model.tsx          # Alternative chat-style UI
│   │   ├── ErrorReporter.tsx, VisualEditsMessenger.tsx
│   │   └── ui/                         # Button, Card, Input, Label, Progress, Tabs, Badge, Separator
│   ├── lib/utils.ts
│   ├── package.json, next.config.ts, tsconfig.json
│   └── public/
├── backend/
│   ├── main.py                         # FastAPI app: all HTTP endpoints, caches, run state, predict (uncertainty)
│   ├── agents/
│   │   ├── automl_agent.py             # plan_automl() — LLM or rule-based fallback → AutoMLPlan
│   │   ├── autonomous_executor.py     # execute_with_auto_fix: prune → plan → config → train → gates → retry
│   │   ├── plan_compiler.py            # compile_preprocessing_code, compile_model_code, compile_pipeline_code
│   │   ├── plan_normalizer.py         # normalize_plan_dict, _generate_feature_transforms_from_profile
│   │   ├── planner_agent.py           # LLM planner (steps 0–6)
│   │   ├── diagnosis_agent.py         # Diagnose failures → plan_changes (target_transformation, feature_transforms)
│   │   ├── recovery_agent.py          # Suggest column alternatives
│   │   ├── error_gating.py            # check_regression_error_gates, check_holdout_baseline_sanity, check_model_quality_minimum
│   │   ├── pipeline_validator.py      # validate_feature_transforms, validate_pipeline_before_training
│   │   ├── llm_interface.py           # LLM provider abstraction
│   │   └── ...
│   ├── ml/
│   │   ├── profiler.py                # profile_dataset: n_rows, numeric_cols, categorical_cols, missing_percent, candidate_targets
│   │   ├── pipeline_builder.py        # build_pipeline, MODEL_COMPLEXITY, get_model_complexity; shallow GB, depth-limited RF
│   │   ├── trainer.py                 # train_classification, train_regression (holdout, baseline sanity), compare_models (order, complexity penalty)
│   │   └── evaluator.py               # evaluate_dataset, prune_features_aggressive, infer_classification_primary_metric
│   ├── schemas/
│   │   ├── pipeline_schema.py         # AutoMLPlan, FeatureTransform, ModelCandidate, target_transformation
│   │   ├── failure_schema.py          # FailureReport
│   │   └── run_state_schema.py        # RunState, AgentEvent
│   └── utils/
│       └── artifact_generator.py      # generate_notebook (_embed_data_cell), generate_readme, generate_model_report
├── start.sh, stop.sh
├── backend/run.sh                      # Run backend from backend/
├── requirements.txt
└── SYSTEM_ARCHITECTURE.md              # This document
```

---

## 3. Frontend: Wizard Flow (Detailed)

**Entry**: `frontend/components/intent-2-model-wizard.tsx`.

### 3.1 State

| State | Type | Purpose |
|-------|------|--------|
| `step` | 1–4 | Current wizard step |
| `files`, `datasetId`, `availableColumns`, `datasetSummary` | — | Upload & dataset |
| `featureColumns` | string[] | Set after train (everything except target) |
| `trainedModel` | object | `run_id`, metrics, `feature_columns`, etc. |
| `predictInputs`, `predictionResult`, `isPredicting` | — | Prediction panel |
| `currentRunId`, `runState`, `liveLogs`, `backendLogTail` | — | Run progress & logs |
| `BACKEND_HTTP_BASE`, `BACKEND_WS_BASE` | `http(s)://host:8000`, `ws://host:8000` | Backend URLs (host = window.location.hostname for LAN) |

### 3.2 Step 1 — Upload

- **POST `/upload`**: body = multipart file. Response: `dataset_id`, profile (n_rows, numeric_cols, categorical_cols, etc.), column list.
- Frontend stores `datasetId`, `availableColumns`, advances to step 2.

### 3.3 Step 2 — Define Intent

- User selects **target** column, optional **task** (classification/regression), optional **metric**. Free-form intent can be parsed by backend if needed.
- These are sent as **POST `/train`** body: `dataset_id`, `target`, `task`, `metric`.

### 3.4 Step 3 — Train

- **POST `/train`** returns `run_id` quickly; training runs in background.
- Frontend polls **GET `/run/latest-id`** (e.g. 200ms) then **GET `/runs/{run_id}`** (e.g. 150ms) for `RunState`: `events`, `current_step`, `progress`, `status`.
- On completion, response (or final fetch) provides `trainedModel` with `run_id`, `feature_columns`, metrics, `all_models`, etc.
- Logs: `runState.events` and/or **GET `/logs/backend`** as fallback.

### 3.5 Step 4 — Deploy & Stats

- **Model selector**: **POST `/run/select-model`** with `run_id`, `model_name` to set active pipeline for predict & artifacts.
- **Downloads**: **GET `/download/{run_id}/notebook`**, `/model`, `/readme`, `/report`.
- **Predict**: **POST `/predict`** with `run_id`, `features` (object). Response:
  - Regression: `{ prediction, uncertainty? }`.
  - Classification: `{ prediction, prediction_encoded, probabilities?, low_confidence? }`.
- Frontend checks `resp.ok`; on error sets `predictionResult = { error: detail }`; on success shows `prediction` (or “—” if missing). For classification it can show `probabilities` and `low_confidence`.

---

## 4. Backend: API Reference

| Method | Path | Purpose |
|--------|------|--------|
| GET | `/health` | Liveness + LLM status (current model, available) |
| POST | `/upload` | Upload CSV → dataset_id, profile |
| POST | `/train` | Start training → run_id; training runs in process (blocking until done) |
| GET | `/run/latest-id` | Current run ID for polling |
| GET | `/runs/{run_id}` | Run state (events, current_step, progress, status) |
| POST | `/run/select-model` | Set active model for run (predict & artifacts) |
| POST | `/predict` | Predict with cached model; returns uncertainty (regression) / probabilities + low_confidence (classification) |
| GET | `/download/{run_id}/notebook` | Jupyter notebook (embedded data, best model from plan) |
| GET | `/download/{run_id}/model` | Pickled pipeline |
| GET | `/download/{run_id}/readme` | README.md |
| GET | `/download/{run_id}/report` | Model report (all models, explanations) |
| GET | `/logs/backend` | Tail of backend log file |

**Caches (in-memory)**:

- **dataset_cache**: `dataset_id` → pandas DataFrame.
- **trained_models_cache**: `run_id` → dict with `model`, `target`, `task`, `feature_columns`, `label_encoder`, `config`, `df`, `metrics`, `all_models`, `holdout_residual_std`, `target_transformation`, etc.

---

## 5. Train Flow (Backend) — Step by Step

1. **POST `/train`** (main.py)
   - Resolve dataset: `request.dataset_id` or most recent in `dataset_cache`.
   - Resolve target: case/partial match, recovery agent suggestions.
   - Infer task from target dtype/cardinality if not provided.
   - **Classification**: set metric via `infer_classification_primary_metric(df, target)` (no raw accuracy default).
   - **Regression**: metric from request or plan (e.g. r2, rmse, mae).
   - Create `run_id`, init `run_state_store[run_id]`, log “Run created”.
   - Call **AutonomousExecutor.execute_with_auto_fix(df, target, task, metric, model_candidates, ...)**.

2. **AutonomousExecutor** (agents/autonomous_executor.py)
   - **Prune**: `prune_features_aggressive(df, target, task)` → replace `df` with pruned DataFrame; restrict `plan.feature_transforms` to remaining columns.
   - **Classification**: set `locked_metric = infer_classification_primary_metric(...)`; do not let planner change metric mid-run.
   - **Loop** (up to `max_attempts`):
     - **Plan**: `plan_automl(df, ...)` (LLM or fallback) → AutoMLPlan.
     - **Config**: `_plan_to_config(plan, profile)` → config (task, feature_transforms, target_transformation if set).
     - **Train**: `compare_models(df, target, task, metric, model_candidates, config)` — models tried in **opinionated order** (simple first); ranking uses **effective_score = cv_score - 0.05 * model_complexity**.
     - **Regression**: After CV, **holdout** (5–10%) is evaluated; **holdout_mae** vs **baseline_mae** (median predictor). If model MAE > 0.75 × baseline MAE → **fail** (check_holdout_baseline_sanity). Store `holdout_residual_std` for prediction uncertainty.
     - **Gates**: Regression — check_regression_error_gates, detect_variance_fit_illusion, analyze_residuals. If heteroscedastic → set `plan.target_transformation = "log1p"` for next attempt (residual-based retry). Classification — check_model_quality_minimum (reinforcing gate).
     - On failure: create FailureReport, diagnose, apply plan changes, retry. On success: return result (plan, best_model, metrics, holdout_residual_std, target_transformation, etc.).

3. **main.py** (after executor returns)
   - Store in **trained_models_cache[run_id]**: pipeline, feature_columns, target, task, label_encoder, df, config, metrics, all_models, **holdout_residual_std**, **target_transformation**, trace, preprocessing_recommendations.
   - Return response with run_id, metrics, feature_columns, all_models, etc.

---

## 6. System Bias & Evaluation (Permanent Design)

These rules are applied **deterministically** (no LLM) to reduce over-trust in CV and improve real-world usefulness.

### 6.1 Holdout Validation (Stop Over-Trusting CV)

- **Where**: `ml/trainer.py` (train_regression).
- After model selection via CV and final fit on train_val:
  - A **never-touched holdout** (5–10% of data) is predicted.
  - **Baseline MAE** = mean(|y_holdout - median(y_train_val)|).
  - **Model holdout MAE** = mean(|y_holdout - pred_holdout|) (in original space; if target_transformation is log1p, predictions are inverted with expm1).
  - **Rule**: If model MAE > 0.75 × baseline MAE → **FAIL** (check_holdout_baseline_sanity in `agents/error_gating.py`).
  - **holdout_residual_std** = std(y_holdout - pred_holdout) is stored for prediction uncertainty.

### 6.2 Aggressive Feature Pruning

- **Where**: `ml/evaluator.py` — `prune_features_aggressive(df, target, task)`.
- Before any model training (called at start of autonomous executor):
  - Drop **variance ≈ 0** (constant columns).
  - Drop **missing ratio > 40%**.
  - Drop **cardinality > 0.8 × n_rows** (ID-like).
  - **Regression**: Univariate correlation (or mutual info for categoricals) with target; drop **bottom 30–40%** weakest; never drop so many that zero features remain.
- Executor restricts `plan.feature_transforms` to columns present in the pruned df.

### 6.3 Opinionated Model Search (Anti-Overfitting)

- **Where**: `ml/pipeline_builder.py` (MODEL_COMPLEXITY, get_model_complexity), `ml/trainer.py` (REGRESSION_MODEL_ORDER, CLASSIFICATION_MODEL_ORDER, compare_models).
- **Search order**: Linear/Ridge/Lasso first → GradientBoosting (shallow) → RandomForest (depth-limited) → SVM/XGB last.
- **Complexity penalty**: effective_score = cv_score - 0.05 × model_complexity (complexity 0–4). Models are ranked by effective_score so complex models must earn their place.
- **Shallow GB**: max_depth=3. **Depth-limited RF**: max_depth=10.

### 6.4 Target Transformation (Regression)

- **Where**: `ml/trainer.py` (train_regression), `agents/autonomous_executor.py` (_plan_to_config, residual retry).
- **log1p**: If config has `target_transformation == "log1p"` and y.min() > -0.99, fit on `np.log1p(y)`. Result stores `target_transformation: "log1p"`. At predict time, prediction is inverted with `np.expm1`.
- **Residual-based retry**: If regression evaluation detects **heteroscedastic** residuals, executor sets `plan.target_transformation = "log1p"` for the next attempt (no LLM).

### 6.5 Residual Analysis → Action

- **Where**: `agents/error_gating.py` (analyze_residuals), `agents/autonomous_executor.py` (_evaluate_regression_model).
- After training, residuals vs predictions are analyzed; heteroscedasticity is detected. If detected, target_transformation is applied on retry (see 6.4).

### 6.6 Classification: No Raw Accuracy Default

- **Where**: `ml/evaluator.py` (infer_classification_primary_metric), `backend/main.py` (POST /train), `agents/autonomous_executor.py` (locked_metric).
- **Inferred metric**: Imbalanced → recall (or roc_auc); risk-sensitive (e.g. fraud, churn) → recall; balanced → f1. Metric is **locked** across retries; planner cannot change it mid-run.

### 6.7 Surface Uncertainty in Predictions

- **Where**: `backend/main.py` (POST /predict), cached `holdout_residual_std`, `target_transformation`.
- **Regression**: Response includes `uncertainty` (holdout_residual_std when available). If model was trained with log1p, prediction is returned in original space (expm1).
- **Classification**: Response includes `probabilities`; `low_confidence: true` when max(probabilities) < 0.6.

---

## 7. Predict Flow (Backend + Frontend)

1. User has `trainedModel.run_id` and is on Deploy & Stats; fills feature inputs and clicks Predict.
2. Frontend **POST `/predict`** with `run_id`, `features` (column → value; numbers when valid).
3. Backend: `model_info = trained_models_cache[run_id]`; `model` = pipeline (preprocessor + model); `input_data = pd.DataFrame([request.features])` reordered to `feature_columns`; `prediction = model.predict(input_data)[0]`.
4. **Regression**: If `target_transformation == "log1p"`, prediction = expm1(prediction). Return `{ prediction, uncertainty }` (uncertainty = holdout_residual_std).
5. **Classification**: Decode with label_encoder; try predict_proba → `probabilities`; set `low_confidence` if max(prob) < 0.6. Return `{ prediction, prediction_encoded, probabilities, low_confidence }`.
6. Frontend: If !resp.ok → set predictionResult to `{ error: detail }`. Else set predictionResult to body; show prediction or “—”; for classification show probabilities and low_confidence when present.

---

## 8. Artifact Generation

- **Notebook**: `utils/artifact_generator.py` — `generate_notebook(df, target, task, config, metrics, ...)`. Uses plan (plan_compiler) for preprocessing/model/pipeline code. **Load Data** cell: `_embed_data_cell(df)` embeds base64 CSV so notebook runs standalone. Best model from config or derived from all_models.
- **README**: `generate_readme(...)`.
- **Report**: `generate_model_report(all_models, target, task, dataset_info, trace, ...)`.

---

## 9. Agents & Modules (Reference)

| Module | Role |
|--------|------|
| automl_agent | plan_automl: LLM or fallback → AutoMLPlan (feature_transforms, model_candidates, primary_metric). |
| autonomous_executor | execute_with_auto_fix: prune → plan → config → train → gates → residual retry / diagnosis → retry. |
| plan_compiler | compile_preprocessing_code, compile_model_code, compile_pipeline_code, compile_metrics_code (for notebook). |
| plan_normalizer | normalize_plan_dict, _generate_feature_transforms_from_profile. |
| diagnosis_agent | Diagnose FailureReport → suggested_stop, recovery_confidence, plan_changes (target_transformation, feature_transforms). |
| error_gating | compute_target_stats, compute_normalized_metrics, check_regression_error_gates, check_holdout_baseline_sanity, check_model_quality_minimum, detect_target_transformation_need, analyze_residuals, detect_variance_fit_illusion. |
| pipeline_validator | validate_feature_transforms, validate_pipeline_before_training. |
| profiler | profile_dataset. |
| pipeline_builder | build_pipeline, MODEL_COMPLEXITY, get_model_complexity; shallow GB, depth-limited RF. |
| trainer | train_classification, train_regression (holdout, baseline sanity, target_transformation), compare_models (order, complexity penalty). |
| evaluator | evaluate_dataset, prune_features_aggressive, infer_classification_primary_metric. |

---

## 10. Run State & Logging

- **run_state_store** (main.py): Per run_id, structure includes run_id, status, current_step, attempt_count, progress, events (list of { ts, step_name, message, status?, payload? }). **GET /runs/{run_id}** returns this.
- **Backend log**: Written to file(s); **GET /logs/backend** returns tail. Frontend uses as fallback.
- **WebSocket**: Optional; frontend closes only when OPEN to avoid “closed before connection” on cleanup.

---

## 11. Troubleshooting

| Symptom | Likely cause | Action |
|--------|----------------|--------|
| Prediction: undefined | Backend 4xx/5xx; UI used to set result to error body. | Fixed: check resp.ok, set predictionResult.error; show “—” if prediction missing. Check Network tab for /predict. |
| Missing features / Prediction failed | Column names or types don’t match training. | Send same feature_columns; use numbers where pipeline expects numeric. |
| Holdout sanity failed | Model MAE > 0.75 × baseline MAE on holdout. | Improve features/model or accept refusal; check metrics. |
| No trained model / Run not found | Backend restarted (cache cleared) or wrong run_id. | Re-train or use correct run_id. |
| Notebook NameError: df | Load Data cell didn’t define df. | Fixed: _embed_data_cell(df) embeds base64 CSV; re-download notebook. |
| Backend won’t start | Syntax/import error. | Run `python -c "import main"` from backend/ and fix errors. |
| LLM unavailable | Rate limits or API key. | Health shows status; training works with rule-based fallback. |

---

## 12. Summary

- **Frontend**: Wizard (upload → intent → train → deploy). Async train; poll run state and logs. Predict with error handling and optional uncertainty/probabilities/low_confidence.
- **Backend**: FastAPI, in-memory caches. Train: autonomous executor (prune → plan → config → compare_models with holdout + baseline sanity, gates, residual retry). Predict: cached pipeline, uncertainty (regression), probabilities + low_confidence (classification), target_transformation inverse when applicable.
- **System bias & evaluation**: Holdout validation, feature pruning, model order + complexity penalty, target transformation (log1p) and residual-based retry, locked classification metric, and prediction uncertainty are built in and documented above.

---

## 13. Logs and Changes

*Keep this section updated when making notable changes to the system.*

### 2025-01-29 — System bias & evaluation refactor (permanent)

- **Holdout validation**: After CV, a never-touched holdout (5–10%) is evaluated; if model MAE > 0.75 × baseline MAE, training fails (`check_holdout_baseline_sanity`). `holdout_residual_std` stored for prediction uncertainty.
- **Feature pruning**: `prune_features_aggressive` (evaluator) — drop zero variance, missing > 40%, cardinality > 0.8×n_rows; regression: drop bottom 30–40% weakest by univariate correlation/MI. Executor runs pruning before training and restricts plan feature_transforms to pruned columns.
- **Model order & complexity**: Opinionated order (linear/ridge/lasso first, GB/RF, SVM/XGB last). effective_score = cv_score - 0.05×complexity; shallow GB (max_depth=3), depth-limited RF (max_depth=10). See pipeline_builder MODEL_COMPLEXITY, trainer REGRESSION_MODEL_ORDER / CLASSIFICATION_MODEL_ORDER.
- **Target transformation**: Regression supports config `target_transformation: "log1p"`; fit on log1p(y), store transformation, invert at predict (expm1). Residual-based retry: if heteroscedastic, set log1p for next attempt.
- **Classification metric**: `infer_classification_primary_metric` (recall/f1/roc_auc by imbalance and risk); metric locked in main and executor (no raw accuracy default).
- **Prediction uncertainty**: Regression response includes `uncertainty` (holdout_residual_std). Classification includes `probabilities` and `low_confidence` when max(prob) < 0.6. Cache stores `holdout_residual_std` and `target_transformation`.

### 2025-01-29 — Prediction UI and notebook fixes

- **Prediction undefined fix**: Frontend checks `resp.ok` on `/predict`; on error sets `predictionResult = { error: detail }`; shows “—” when prediction is missing.
- **Notebook standalone**: `_embed_data_cell(df)` embeds base64 CSV in “Load Data” cell so notebook runs without external files. Best model derived from all_models when config.model missing.
- **Backend syntax**: Fixed unclosed parenthesis in main.py and artifact_generator.py (best-model lambda); replaced with helper functions.

### (Template for future entries)

- **YYYY-MM-DD — Short title**
  - Bullet points describing what changed and where (files / modules).
