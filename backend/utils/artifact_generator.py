"""
Generate artifacts: Jupyter notebook, pickle file, charts, README.

Notebook is a LIVE REASONING SURFACE: attempt-based, repair diffs visible, refusal = no final model.
"""

import json
import pickle
import base64
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import numpy as np


def _md_cell(source: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def _code_cell(source: List[str]) -> Dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "source": source}


def _repair_diff_summary(entry: Dict[str, Any]) -> str:
    """One-line summary of what changed (RepairPlan diff)."""
    diag = entry.get("diagnosis") or {}
    plan_changes = diag.get("plan_changes") or diag.get("repair_plan") or {}
    parts = []
    if plan_changes.get("target_transformation"):
        parts.append(f"target_transform → {plan_changes['target_transformation']}")
    if plan_changes.get("drop_features"):
        parts.append(f"drop_features → {plan_changes['drop_features']}")
    if plan_changes.get("add_features"):
        parts.append(f"add_features → {plan_changes['add_features']}")
    if plan_changes.get("replace_model"):
        parts.append(f"replace_model → {plan_changes['replace_model']}")
    if plan_changes.get("reorder_models"):
        parts.append(f"reorder_models → {plan_changes['reorder_models']}")
    if plan_changes.get("change_encoding"):
        parts.append(f"change_encoding → {plan_changes['change_encoding']}")
    return "; ".join(parts) if parts else "(no structured diff)"


def _build_attempt_based_cells(
    df: pd.DataFrame,
    target: str,
    task: str,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Build attempt-based notebook cells: Attempt 1 → Repair Proposal → Attempt 2 → ...
    Returns (cells, is_refused_or_low_confidence).
    """
    from schemas.pipeline_schema import ExecutionPlan
    from agents.plan_compiler import (
        compile_preprocessing_code_from_execution_plan,
        compile_model_code_from_execution_plan,
        compile_metrics_code_from_execution_plan,
        compile_pipeline_code_from_execution_plan,
        IncompleteExecutionPlan,
        RefuseCodeGeneration,
    )
    cells: List[Dict[str, Any]] = []
    structural_plan = (config or {}).get("structural_plan") or {}
    execution_plans = (config or {}).get("execution_plans") or []
    failure_history = (config or {}).get("failure_history") or []
    refused = (config or {}).get("refused", False)

    # Structural Plan (once) — LLM writes code (Cursor-style); code below is LLM output
    if structural_plan and isinstance(structural_plan, dict):
        sp = structural_plan
        body = (
            "The **code** in this notebook was **written by an LLM** (Cursor-style), not compiled from a plan.\n\n"
            f"**Target:** {sp.get('inferred_target', '—')} (confidence: {sp.get('target_confidence', 0):.2f})\n"
            f"**Task:** {sp.get('task_type', '—')}\n"
            f"**Feature semantics:** {len(sp.get('feature_semantics', {}))} features\n"
        )
        if sp.get("leakage_candidates"):
            body += f"\n**Leakage candidates:** {', '.join(sp['leakage_candidates'][:10])}\n"
        cells.append(_md_cell(f"## Structural Plan (What is this dataset?)\n\n{body}\n"))

    if not execution_plans:
        return cells, refused

    is_low_confidence = False
    for i, ep_dict in enumerate(execution_plans):
        if not isinstance(ep_dict, dict):
            continue
        attempt_num = i + 1
        plan_quality = ep_dict.get("plan_quality", "high_confidence")
        if plan_quality == "fallback_low_confidence":
            is_low_confidence = True

        # ## Attempt N
        models_str = ", ".join(
            m.get("model_name", "?") if isinstance(m, dict) else getattr(m, "model_name", "?")
            for m in (ep_dict.get("model_candidates") or [])[:6]
        )
        ep_md = (
            f"## Attempt {attempt_num}\n\n"
            f"**ExecutionPlan v{attempt_num}**\n"
            f"- target_transformation: {ep_dict.get('target_transformation', 'none')}\n"
            f"- model_candidates: {models_str}\n"
            f"- plan_quality: {plan_quality}\n"
            f"- primary_metric: {ep_dict.get('primary_metric', '—')}\n"
        )
        if ep_dict.get("reasoning_md"):
            ep_md += f"\n**Reasoning:** {ep_dict['reasoning_md'][:300]}...\n" if len(ep_dict.get("reasoning_md", "")) > 300 else f"\n**Reasoning:** {ep_dict['reasoning_md']}\n"
        cells.append(_md_cell(ep_md))

        # Code: LLM output (Cursor-style) first; fallback to plan_compiler
        ep = None
        try:
            ep = ExecutionPlan(**ep_dict)
        except Exception:
            pass

        model_name = "random_forest"
        if ep and ep.model_candidates:
            m0 = (ep.model_candidates or [{}])[0]
            model_name = m0.get("model_name", "random_forest") if isinstance(m0, dict) else getattr(m0, "model_name", "random_forest")

        llm_cells: List[List[str]] = []
        try:
            from agents.notebook_code_agent import generate_notebook_code_llm
            import os
            provider = os.getenv("LLM_PROVIDER", "gemini_cli")
            llm_cells = generate_notebook_code_llm(
                ep_dict,
                list(df.columns),
                target,
                task,
                model_name,
                llm_provider=provider,
            ) or []
        except Exception as _:
            pass

        if len(llm_cells) >= 1:
            # Use LLM output directly (Cursor-style) — 1 block = all-in-one cell; 3–4 = separate cells
            cells.append(_md_cell(f"### Code written by LLM (Attempt {attempt_num})"))
            for block in llm_cells:
                cells.append(_code_cell(block))
        else:
            # Fallback: compiler-generated code
            if ep is None:
                cells.append(_md_cell("*ExecutionPlan could not be parsed — code generation skipped.*"))
                if i < len(failure_history):
                    _append_repair_proposal_cells(cells, failure_history[i], attempt_num)
                continue

            preproc_src = None
            model_src = None
            pipeline_src = None
            metrics_src = None
            try:
                preproc_src = compile_preprocessing_code_from_execution_plan(ep)
            except (IncompleteExecutionPlan, RefuseCodeGeneration, Exception) as e:
                preproc_src = ["# Code generation refused or incomplete\n", f"# {str(e)[:200]}\n", "\n"] + _SAFE_FALLBACK_PREPROC_LINES
            try:
                model_src = compile_model_code_from_execution_plan(ep, task, model_name)
            except (IncompleteExecutionPlan, RefuseCodeGeneration, Exception) as e:
                model_src = ["# Model code refused or incomplete\n", f"# {str(e)[:200]}\n"]
            try:
                pipeline_src = compile_pipeline_code_from_execution_plan(ep)
            except (IncompleteExecutionPlan, RefuseCodeGeneration, Exception) as e:
                pipeline_src = ["# Pipeline code refused or incomplete\n", f"# {str(e)[:200]}\n"]
            try:
                metrics_src = compile_metrics_code_from_execution_plan(ep, task)
            except (IncompleteExecutionPlan, RefuseCodeGeneration, Exception):
                metrics_src = None

            cells.append(_md_cell(f"### Code (compiler fallback) v{attempt_num}"))
            _preproc_lines = preproc_src if isinstance(preproc_src, list) else [preproc_src] if preproc_src else _SAFE_FALLBACK_PREPROC_LINES
            cells.append(_code_cell(_preproc_lines))
            cells.append(_code_cell(model_src if isinstance(model_src, list) else [model_src]))
            cells.append(_code_cell(pipeline_src if isinstance(pipeline_src, list) else [pipeline_src]))
            train_eval_lines = [
                "pipeline.fit(X_train, y_train)\n",
                "y_pred = pipeline.predict(X_test)\n",
                "\n",
                "# Evaluate\n",
            ]
            if metrics_src:
                train_eval_lines.extend(
                    line if line.endswith("\n") else line + "\n"
                    for line in (metrics_src.split("\n") if isinstance(metrics_src, str) else metrics_src)
                )
            else:
                train_eval_lines.append("# Metrics code not generated\n")
            cells.append(_code_cell(train_eval_lines))

        # Metrics / Failure gates (if this attempt failed)
        if i < len(failure_history):
            fail_entry = failure_history[i]
            fr = (fail_entry.get("failure_report") or {})
            failed_gates = fr.get("failed_gates") or []
            met = fr.get("metrics") or {}
            cells.append(_md_cell(
                f"### Failure gates triggered (Attempt {attempt_num})\n\n"
                f"**Failed gates:**\n" + "\n".join(f"- {g}" for g in failed_gates[:15]) + "\n\n"
                f"**Metrics:** " + ", ".join(f"{k}={v:.4f}" for k, v in list(met.items())[:8] if isinstance(v, (int, float)))
            ))
            _append_repair_proposal_cells(cells, fail_entry, attempt_num)

    return cells, (refused or is_low_confidence)


# Safe fallback preprocessing cell: always defines transformers and preprocessor (no NameError).
# Used when compiler refuses in attempt-based notebook so downstream cells (pipeline, etc.) have valid names.
_SAFE_FALLBACK_PREPROC_LINES = [
    "# ⚠️ Plan refused/incomplete — minimal safe preprocessing so pipeline and later cells run.\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.impute import SimpleImputer\n",
    "\n",
    "numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n",
    "categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()\n",
    "transformers = []\n",
    "if numeric_cols:\n",
    "    transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols))\n",
    "if categorical_cols:\n",
    "    transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols))\n",
    "preprocessor = ColumnTransformer(transformers, remainder='drop')\n",
]


def _append_repair_proposal_cells(cells: List[Dict[str, Any]], fail_entry: Dict[str, Any], attempt_num: int) -> None:
    """Append ## Repair Proposal (diff + why)."""
    diag = fail_entry.get("diagnosis") or {}
    diagnosis_md = diag.get("diagnosis_md", "No diagnosis text.")
    diff_summary = _repair_diff_summary(fail_entry)
    cells.append(_md_cell(
        f"## Repair Proposal (after Attempt {attempt_num})\n\n"
        f"**What will change and why:**\n\n{diagnosis_md[:1500]}{'...' if len(diagnosis_md) > 1500 else ''}\n\n"
        f"**Structured diff:** {diff_summary}\n"
    ))


def _embed_data_cell(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return a notebook code cell that defines `df` from embedded base64 CSV
    so the notebook runs standalone without external data files.
    """
    csv_str = df.to_csv(index=False)
    b64 = base64.b64encode(csv_str.encode("utf-8")).decode("ascii")
    # Chunk long base64 for readability (optional); single string is fine for execution
    source = [
        "# Embedded dataset (base64 CSV) — notebook runs standalone\n",
        "import base64\n",
        "from io import StringIO\n",
        "\n",
        f"_data_b64 = {repr(b64)}\n",
        "_csv = base64.b64decode(_data_b64).decode('utf-8')\n",
        "df = pd.read_csv(StringIO(_csv))\n",
        "\n",
        "print(f'Dataset shape: {df.shape}')\n",
        "print(f'Columns: {list(df.columns)}')\n",
        "df.head()\n",
    ]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": source,
    }


def generate_notebook(
    df: pd.DataFrame,
    target: str,
    task: str,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    feature_importance: Optional[Dict[str, float]] = None,
    model: Any = None
) -> str:
    """
    Generate a Jupyter notebook with complete training code.
    
    CRITICAL: Code MUST be generated from AutoMLPlan, not hardcoded boilerplate.
    """
    from schemas.pipeline_schema import AutoMLPlan
    from agents.plan_compiler import (
        compile_preprocessing_code,
        compile_model_code,
        compile_metrics_code,
        compile_pipeline_code,
        validate_plan_for_execution,
        IncompleteExecutionPlan,
        RefuseCodeGeneration,
    )
    
    automl_plan_dict = (config or {}).get("automl_plan") or {}
    
    # Convert dict to AutoMLPlan if needed
    plan = None
    if automl_plan_dict and isinstance(automl_plan_dict, dict):
        try:
            plan = AutoMLPlan(**automl_plan_dict)
            # Validate plan before generating code
            try:
                validate_plan_for_execution(plan)
            except RuntimeError as e:
                # For low-confidence plans, still generate but with warnings
                if "low-confidence" in str(e).lower():
                    print(f"⚠️  Warning: {e}")
                else:
                    raise
        except Exception as e:
            print(f"⚠️  Could not parse AutoMLPlan: {e}. Using fallback code generation.")
            plan = None
    
    automl_plan = automl_plan_dict  # Keep dict for markdown sections
    structural_plan = (config or {}).get("structural_plan")
    execution_plans = (config or {}).get("execution_plans") or []
    failure_history = (config or {}).get("failure_history") or []
    refused = (config or {}).get("refused", False)

    # Agentic: attempt-based notebook (live reasoning surface) when we have execution_plans
    use_attempt_based = bool(execution_plans)
    attempt_based_cells: List[Dict[str, Any]] = []
    is_refused_or_low = refused
    full_llm_notebook_cells: Optional[List[Dict[str, Any]]] = None
    if use_attempt_based and execution_plans:
        # Try full-notebook LLM first (whole notebook by LLM — crazy good, reasoning, descriptive)
        try:
            import os
            from agents.notebook_code_agent import generate_full_notebook_llm
            last_ep = execution_plans[-1] if isinstance(execution_plans[-1], dict) else {}
            cfg = config or {}
            model_name = cfg.get("model") or "random_forest"
            if (cfg.get("all_models") or []) and not model_name:
                pm = (cfg.get("primary_metric") or "r2").lower()
                lower_is_better = pm in ("rmse", "mae")
                def _score(m):
                    v = m.get("primary_metric") or m.get("cv_mean")
                    if v is None:
                        return float("inf") if lower_is_better else 0
                    return -v if lower_is_better else v
                best = max(
                    (m for m in cfg["all_models"] if isinstance(m, dict)),
                    key=_score,
                    default={},
                )
                model_name = best.get("model_name") or "random_forest"
            full_llm_notebook_cells = generate_full_notebook_llm(
                execution_plan=last_ep,
                structural_plan=structural_plan,
                columns=list(df.columns),
                target=target,
                task=task,
                model_name=model_name,
                metrics=metrics,
                llm_provider=os.getenv("LLM_PROVIDER", "gemini_cli"),
            )
        except Exception as _:
            full_llm_notebook_cells = None
    if use_attempt_based and not (full_llm_notebook_cells and len(full_llm_notebook_cells) >= 4):
        attempt_based_cells, is_refused_or_low = _build_attempt_based_cells(df, target, task, config)

    # Legacy: single-plan sections (only when NOT using attempt-based)
    agentic_cells: List[Dict[str, Any]] = []
    if not use_attempt_based and structural_plan and isinstance(structural_plan, dict):
        sp = structural_plan
        body = (
            f"**Target:** {sp.get('inferred_target', '—')} (confidence: {sp.get('target_confidence', 0):.2f})\n"
            f"**Task:** {sp.get('task_type', '—')}\n"
            f"**Feature semantics:** {len(sp.get('feature_semantics', {}))} features classified\n"
        )
        if sp.get("leakage_candidates"):
            body += f"\n**Leakage candidates:** {', '.join(sp['leakage_candidates'][:10])}\n"
        if sp.get("dataset_warnings"):
            body += f"\n**Warnings:** {'; '.join(sp['dataset_warnings'][:5])}\n"
        agentic_cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## Structural Plan (What is this dataset?)\n\n{body}\n"]})
    if not use_attempt_based and execution_plans:
        parts = []
        for i, ep in enumerate(execution_plans[:5], 1):
            if not isinstance(ep, dict):
                continue
            parts.append(
                f"**Attempt {i}:** target_transform={ep.get('target_transformation', 'none')}, "
                f"models={[m.get('model_name') if isinstance(m, dict) else getattr(m, 'model_name', '?') for m in (ep.get('model_candidates') or [])][:5]}, "
                f"quality={ep.get('plan_quality', '—')}"
            )
        agentic_cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## Execution Plans (per attempt)\n\n" + "\n\n".join(parts) + "\n"]})
    if not use_attempt_based and failure_history:
        parts = []
        for entry in failure_history[:5]:
            attempt = entry.get("attempt", "?")
            diag = entry.get("diagnosis", {}) or {}
            parts.append(f"**Attempt {attempt}:** {diag.get('diagnosis_md', '')[:200]}...")
        agentic_cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## Failures & Repairs\n\n" + "\n\n".join(parts) + "\n"]})

    # Preprocessing / model / pipeline / metrics code: from plan with fallback on compiler error
    _preproc_source = None
    _model_source = None
    _pipeline_source = None
    _metrics_source = None
    if plan:
        try:
            _preproc_source = compile_preprocessing_code(plan, df)
        except (IncompleteExecutionPlan, RefuseCodeGeneration, Exception) as _e:
            print(f"⚠️  Compiler could not build preprocessing from plan: {_e}. Using fallback.")
            _preproc_source = None
        try:
            _model_source = compile_model_code(plan, config.get("model"), task)
        except (IncompleteExecutionPlan, RefuseCodeGeneration, ValueError, Exception) as _e:
            print(f"⚠️  Compiler could not build model from plan: {_e}. Using fallback.")
            _model_source = None
        try:
            _pipeline_source = compile_pipeline_code(plan)
        except (IncompleteExecutionPlan, RefuseCodeGeneration, Exception) as _e:
            print(f"⚠️  Compiler could not build pipeline from plan: {_e}. Using fallback.")
            _pipeline_source = None
        try:
            _metrics_source = compile_metrics_code(plan, task)
        except (IncompleteExecutionPlan, RefuseCodeGeneration, Exception) as _e:
            print(f"⚠️  Compiler could not build metrics from plan: {_e}. Using fallback.")
            _metrics_source = None
    _fallback_preproc = [
        "# ⚠️ Plan not available or compiler could not build transformers - using fallback preprocessing\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
        "from sklearn.impute import SimpleImputer\n",
        "\n",
        "numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n",
        "categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()\n",
        "transformers = []\n",
        "if numeric_cols:\n",
        "    transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols))\n",
        "if categorical_cols:\n",
        "    transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols))\n",
        "preprocessor = ColumnTransformer(transformers, remainder='drop')\n",
    ]
    _fallback_model = [
        "# ⚠️ Plan not available - using fallback model\n",
        f"from sklearn.ensemble import {'RandomForestClassifier' if task == 'classification' else 'RandomForestRegressor'}\n",
        f"model = {'RandomForestClassifier' if task == 'classification' else 'RandomForestRegressor'}(n_estimators=200, random_state=42)\n",
    ]
    _fallback_pipeline = [
        "from sklearn.pipeline import Pipeline\n",
        "pipeline = Pipeline([\n",
        "    ('preprocessor', preprocessor),\n",
        "    ('model', model)\n",
        "])\n",
    ]
    _fallback_metrics = [
        "# ⚠️ Plan not available - using fallback metrics\n",
        "from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error\n",
    ]

    # Derive best model from all_models if config.model is missing/unknown
    cfg = config or {}
    model_name = cfg.get("model") or ""
    if (not model_name or model_name == "unknown") and cfg.get("all_models"):
        primary_metric = cfg.get("primary_metric") or ("accuracy" if "classification" in task else "r2")
        reverse = primary_metric.lower() not in ("rmse", "mae")
        all_models = cfg["all_models"]

        def _best_score(m):
            raw = m.get("primary_metric") or m.get("cv_mean")
            if reverse:
                return raw if raw is not None else 0
            return -(raw if raw is not None else float("inf"))

        best = max(all_models, key=_best_score)
        model_name = best.get("model_name") or "unknown"
        if model_name and model_name != "unknown":
            config = {**cfg, "model": model_name}

    # Expect markdown sections in automl_plan if present
    md_sections = []
    if isinstance(automl_plan, dict) and automl_plan:
        # Always show sections if they exist (even if fallback text)
        planning_source = automl_plan.get("planning_source", "unknown")
        
        # If LLM was used, show LLM-generated content (even if empty, don't say "LLM unavailable")
        # Only show fallback text if planning_source is actually "fallback"
        is_llm_plan = planning_source == "llm" or planning_source == "auto_repair"
        fallback_text = " (LLM unavailable - using rule-based fallback)" if planning_source == "fallback" else ""
        
        # For LLM plans, use generic descriptions if markdown is missing (don't say "LLM unavailable")
        md_sections = [
            ("STEP 0 — TASK INFERENCE", automl_plan.get("task_inference_md", "") or ("Task inference based on dataset analysis." if is_llm_plan else f"Rule-based fallback task inference{fallback_text}.")),
            ("STEP 1 — DATASET INTELLIGENCE", automl_plan.get("dataset_intelligence_md", "") or ("Dataset intelligence analysis." if is_llm_plan else f"Rule-based fallback dataset intelligence{fallback_text}.")),
            ("STEP 2 — TRANSFORMATION STRATEGY", automl_plan.get("transformation_strategy_md", "") or ("Transformation strategy based on feature analysis." if is_llm_plan else f"Rule-based fallback transformation strategy{fallback_text}.")),
            ("STEP 3 — MODEL CANDIDATE SELECTION", automl_plan.get("model_selection_md", "") or ("Model selection based on task and dataset characteristics." if is_llm_plan else f"Rule-based fallback model selection{fallback_text}.")),
            ("STEP 4 — TRAINING & VALIDATION", automl_plan.get("training_validation_md", "") or "Use cross-validation by default with task-appropriate metrics."),
            ("STEP 5 — ERROR & BEHAVIOR ANALYSIS", automl_plan.get("error_behavior_analysis_md", "") or "Analyze residuals/confusion matrix and error slices."),
            ("STEP 6 — EXPLAINABILITY", automl_plan.get("explainability_md", "") or "Use feature_importances_ when available and align post-encoding names."),
        ]
        # Add planning source note with confidence warning
        plan_quality = automl_plan.get("plan_quality", "high_confidence")
        if planning_source and planning_source != "unknown":
            warning = ""
            if plan_quality == "fallback_low_confidence":
                warning = "⚠️ **LOW-CONFIDENCE FALLBACK PLAN**\n\nThis plan was generated using rule-based fallbacks because the LLM was unavailable or returned invalid responses. **Results may be suboptimal.**\n\n"
            elif plan_quality == "medium_confidence":
                warning = "⚠️ **MEDIUM-CONFIDENCE PLAN**\n\nSome decisions have low confidence scores. Review carefully.\n\n"
            
            planning_error = automl_plan.get("planning_error", "")
            error_note = f"\n**Error:** {planning_error}" if planning_error else ""
            
            md_sections.insert(0, ("PLANNING SOURCE", 
                f"{warning}**Planning Method:** {planning_source.upper()}{error_note}\n\n"
                f"**Target Confidence:** {automl_plan.get('target_confidence', 1.0):.2f}\n"
                f"**Task Confidence:** {automl_plan.get('task_confidence', 1.0):.2f}\n"
                f"**Plan Quality:** {plan_quality.replace('_', ' ').title()}\n"))

    # When full-LLM notebook succeeded: minimal base (imports + load data only); rest = LLM cells
    use_full_llm_notebook = use_attempt_based and full_llm_notebook_cells and len(full_llm_notebook_cells) >= 4

    if use_full_llm_notebook:
        base_cells = [
            {"cell_type": "markdown", "metadata": {}, "source": ["## Setup\n\nImports and data load. The rest of this notebook was **generated by an LLM** with step-by-step reasoning and descriptive explanations.\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["### Import Libraries"]},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from sklearn.model_selection import train_test_split, cross_val_score\n",
                    "from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder\n",
                    "from sklearn.impute import SimpleImputer\n",
                    "from sklearn.pipeline import Pipeline\n",
                    "from sklearn.compose import ColumnTransformer\n",
                    "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n",
                    "from sklearn.linear_model import LogisticRegression, LinearRegression\n",
                    "from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score\n",
                    "import pickle\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "sns.set_style('whitegrid')\n",
                    "plt.rcParams['figure.figsize'] = (12, 6)\n",
                ]
            },
            {"cell_type": "markdown", "metadata": {}, "source": ["### Load Data\n\nThe dataframe `df` is loaded below (embedded). All following cells were generated by the LLM.\n"]},
            _embed_data_cell(df),
        ]
        base_cells.extend(full_llm_notebook_cells)
        if not is_refused_or_low:
            base_cells.append(_md_cell("---\n\n## Save & Use Model\n\nSave the trained pipeline and run a quick prediction example."))
            base_cells.append(_code_cell([
                "with open('model.pkl', 'wb') as f:\n",
                "    pickle.dump(pipeline, f)\n",
            ] + (["\n", "with open('label_encoder.pkl', 'wb') as f:\n", "    pickle.dump(le, f)\n"] if task == "classification" else []) + ["\n", "print('Model saved to model.pkl')\n"]))
            base_cells.append(_code_cell([
                "# Runnable example: predict on first test row\n",
                "sample = X_test.iloc[[0]]\n",
                "pred = pipeline.predict(sample)\n",
                "print(f'Prediction (first test row): {pred[0]}')\n",
                "\n",
                "# Your own data: same columns as X, e.g. new_row = pd.DataFrame({...}); pipeline.predict(new_row)\n",
            ]))
    else:
        # Standard base cells: title, optional legacy md_sections, imports, load data, prepare
        base_cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# ML Model Training: Predicting {target}\n",
                    f"\n",
                    f"**Live reasoning surface** — attempt-based execution log.\n",
                    f"\n",
                    f"The **code** below was **written by an LLM** (Cursor-style), not compiled.\n",
                    f"\n",
                    f"**Task:** {task} | **Target:** {target}\n"
                ]
            },
            *([
                {"cell_type": "markdown", "metadata": {}, "source": [f"## {title}\n\n{body}\n"]}
                for title, body in md_sections
                if body and str(body).strip() and not use_attempt_based
            ]),
            {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Import Libraries"]},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from sklearn.model_selection import train_test_split, cross_val_score\n",
                    "from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder\n",
                    "from sklearn.impute import SimpleImputer\n",
                    "from sklearn.pipeline import Pipeline\n",
                    "from sklearn.compose import ColumnTransformer\n",
                    "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n",
                    "from sklearn.linear_model import LogisticRegression, LinearRegression\n",
                    "from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score\n",
                    "import pickle\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "sns.set_style('whitegrid')\n",
                    "plt.rcParams['figure.figsize'] = (12, 6)\n",
                ]
            },
            {"cell_type": "markdown", "metadata": {}, "source": ["## 2. Load Data"]},
            _embed_data_cell(df),
            {"cell_type": "markdown", "metadata": {}, "source": ["## 3. Prepare Data"]},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": (
                    [
                        f"# Separate features and target\n",
                        f"X = df.drop(columns=['{target}'])\n",
                        f"y = df['{target}']\n",
                        "\n",
                    ]
                    + (["le = LabelEncoder()\n", "y = le.fit_transform(y.astype(str))\n"] if task == "classification" else [])
                    + [
                        "\n",
                        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                        "print(f'Training set: {X_train.shape}')\n",
                        "print(f'Test set: {X_test.shape}')\n",
                    ]
                )
            },
        ]

    # Attempt-based: either full-LLM notebook (already complete) or attempt log + Outcome
    if use_attempt_based:
        if use_full_llm_notebook:
            # Full-LLM notebook is already complete (Setup + LLM cells + Save & Use). Do NOT append legacy ## 4–10.
            notebook = {
                "cells": base_cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}},
                "nbformat": 4,
                "nbformat_minor": 4,
            }
            return json.dumps(notebook, indent=1)
        base_cells.extend(attempt_based_cells)
        refusal_reason = (config or {}).get("refusal_reason") or ""
        if is_refused_or_low:
            base_cells.append(_md_cell(
                "## Outcome: Refusal / Incomplete\n\n"
                "Training was **refused** or plan confidence was low. No final model is presented.\n\n"
                f"**Reason:** {refusal_reason or 'Error gates failed or plan_quality was fallback_low_confidence.'}\n\n"
                "No Save Model or Make Predictions cells — this is not a successful run."
            ))
        else:
            base_cells.append(_md_cell("## Outcome: Success\n\nFinal model was trained and passed error gates."))
            base_cells.append(_md_cell("### Save Model"))
            base_cells.append(_code_cell([
                "with open('model.pkl', 'wb') as f:\n",
                "    pickle.dump(pipeline, f)\n",
            ] + (["\n", "with open('label_encoder.pkl', 'wb') as f:\n", "    pickle.dump(le, f)\n"] if task == "classification" else []) + ["\n", "print('Model saved.')\n"]))
            base_cells.append(_md_cell("### Make Predictions"))
            base_cells.append(_code_cell([
                "# Runnable example: first test row\n",
                "sample = X_test.iloc[[0]]\n",
                "pred = pipeline.predict(sample)\n",
                "print(f'Prediction: {pred[0]}')\n",
                "\n",
                "# Your data: new_data = pd.DataFrame({...}); pipeline.predict(new_data)\n",
            ]))
        notebook = {
            "cells": base_cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}},
            "nbformat": 4,
            "nbformat_minor": 4,
        }
        return json.dumps(notebook, indent=1)

    # Legacy: linear report (single plan) — only when NOT attempt-based
    notebook = {
        "cells": [
            *base_cells,
            *agentic_cells,
            {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Build Preprocessing Pipeline (from AutoMLPlan)"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [_preproc_source] if _preproc_source else _fallback_preproc},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 5. Build Model (from AutoMLPlan)"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [_model_source] if _model_source else _fallback_model},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 6. Assemble Pipeline (from AutoMLPlan)"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [_pipeline_source] if _pipeline_source else _fallback_pipeline},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 7. Train Model"]},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "pipeline.fit(X_train, y_train)\n",
                    "y_pred = pipeline.predict(X_test)\n",
                    "\n",
                    "# Evaluate using metrics from AutoMLPlan\n",
                ]
                + ([_metrics_source] if _metrics_source else _fallback_metrics)
                + [
                    "\n",
                    "# Calculate metrics\n",
                ]
                + (_generate_metrics_evaluation_code(plan, task) if plan else (["score = accuracy_score(y_test, y_pred)\n", "print(f'Accuracy: {score:.4f}')\n"] if task == "classification" else ["score = r2_score(y_test, y_pred)\n", "print(f'R2 Score: {score:.4f}')\n"])),
            },
            {"cell_type": "markdown", "metadata": {}, "source": ["## 8. Feature Importance (from plan.explainability_md)"]},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Get feature importance (aligned with plan)\n",
                    "if hasattr(pipeline.named_steps['model'], 'feature_importances_'):\n",
                    "    importances = pipeline.named_steps['model'].feature_importances_\n",
                    "    \n",
                    "    # Get feature names after preprocessing (aligned with plan, not dtype-based)\n",
                    "    # NOTE: Do NOT use numeric_cols or categorical_cols - they may not be defined\n",
                    "    try:\n",
                    "        preprocessor = pipeline.named_steps['preprocessor']\n",
                    "        feature_names = []\n",
                    "        # Get feature names from preprocessor transformers\n",
                    "        if hasattr(preprocessor, 'transformers_'):\n",
                    "            for name, transformer, cols in preprocessor.transformers_:\n",
                    "                if hasattr(transformer, 'get_feature_names_out'):\n",
                    "                    feature_names.extend(transformer.get_feature_names_out(cols))\n",
                    "                elif hasattr(transformer, 'named_steps'):\n",
                    "                    # Pipeline transformer\n",
                    "                    for step_name, step_transformer in transformer.named_steps.items():\n",
                    "                        if hasattr(step_transformer, 'get_feature_names_out'):\n",
                    "                            feature_names.extend(step_transformer.get_feature_names_out(cols))\n",
                    "                            break\n",
                    "                else:\n",
                    "                    # Fallback: use column names\n",
                    "                    feature_names.extend([f'{name}_{col}' for col in cols])\n",
                    "        else:\n",
                    "            # Preprocessor not fitted yet - use generic names\n",
                    "            feature_names = [f'feature_{i}' for i in range(len(importances))]\n",
                    "    except Exception as e:\n",
                    "        # Fallback: use generic feature names\n",
                    "        feature_names = [f'feature_{i}' for i in range(len(importances))]\n",
                    "    \n",
                    "    # Create importance DataFrame\n",
                    "    importance_df = pd.DataFrame({\n",
                    "        'feature': feature_names[:len(importances)],\n",
                    "        'importance': importances\n",
                    "    }).sort_values('importance', ascending=False)\n",
                    "    \n",
                    "    # Plot\n",
                    "    plt.figure(figsize=(10, 6))\n",
                    "    sns.barplot(data=importance_df.head(10), x='importance', y='feature')\n",
                    "    plt.title('Top 10 Feature Importance')\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "    \n",
                    "    print(importance_df)\n",
                    "else:\n",
                    "    print('Feature importance not available for this model type.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 9. Save Model"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": (
                    [
                        "# Save the trained model\n",
                        "with open('model.pkl', 'wb') as f:\n",
                        "    pickle.dump(pipeline, f)\n",
                    ] + (
                        ["\n", "# Save label encoder if used\n", "with open('label_encoder.pkl', 'wb') as f:\n", "    pickle.dump(le, f)\n"]
                        if task == 'classification'
                        else []
                    ) + [
                        "\n",
                        "print('Model saved successfully!')"
                    ]
                )
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 10. Make Predictions\n\nUse the trained pipeline to predict. Below: one runnable example (first test row) and a template for your own data."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Example: predict on first row of test set (runnable as-is)\n",
                    "sample = X_test.iloc[[0]]\n",
                    "pred = pipeline.predict(sample)\n",
                    "print(f'Sample prediction (first test row): {pred[0]}')\n",
                    "\n",
                    "# For your own data: build a DataFrame with same columns as X, then:\n",
                    "# new_data = pd.DataFrame({col: [value], ... for col in X.columns})\n",
                    "# print(pipeline.predict(new_data))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return json.dumps(notebook, indent=1)


def generate_readme(
    target: str,
    task: str,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    feature_importance: Optional[Dict[str, float]] = None,
    dataset_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a comprehensive README.md file.
    """
    readme = f"""# ML Model: Predicting {target}

This model was auto-generated by **Intent2Model** - an LLM-guided AutoML platform.

## 📊 Model Overview

- **Task Type:** {task.title()}
- **Target Column:** `{target}`
- **Model Architecture:** {config.get('model', 'Unknown')}
- **Preprocessing:** {', '.join(config.get('preprocessing', [])) or 'None'}

## 📈 Performance Metrics

"""
    
    for metric_name, metric_value in metrics.items():
        readme += f"- **{metric_name.upper()}:** {metric_value:.4f}\n"
    
    readme += f"""
## 🔧 Model Details

### Preprocessing Steps
"""
    
    preprocessing = config.get('preprocessing', [])
    if preprocessing:
        for step in preprocessing:
            readme += f"- {step.replace('_', ' ').title()}\n"
    else:
        readme += "- No preprocessing applied\n"
    
    readme += f"""
### Model Configuration
- **Algorithm:** {config.get('model', 'Unknown')}
- **Task:** {task}

## 📦 Files Included

- `model.pkl` - Trained model (pickle format)
- `training_notebook.ipynb` - Complete Jupyter notebook with training code
- `README.md` - This file
- `charts/` - Visualization charts (if generated)

## 🚀 Usage

### Load and Use the Model

```python
import pickle
import pandas as pd

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare your data (same format as training data)
new_data = pd.DataFrame({{
    # Your feature columns here
}})

# Make prediction
prediction = model.predict(new_data)
print(f'Prediction: {{prediction}}')
```

## 📊 Feature Importance

"""
    
    if feature_importance:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        readme += "Top features by importance:\n\n"
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            readme += f"{i}. **{feature}**: {importance:.4f}\n"
    else:
        readme += "Feature importance not available.\n"
    
    readme += f"""
## 🔄 Retraining

To retrain the model, use the provided Jupyter notebook (`training_notebook.ipynb`).

## 📝 Notes

- Model was trained using cross-validation
- All preprocessing steps are included in the pipeline
- The model can be used directly for predictions

## 🤖 Generated by Intent2Model

This model was automatically generated by Intent2Model's LLM-guided AutoML system.
For more information, visit: https://github.com/lakshitsachdeva/intent2model

---
*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return readme


def save_model_pickle(model: Any, filepath: str) -> str:
    """
    Save model as pickle file and return base64 encoded string.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_chart_image(
    chart_type: str,
    data: Dict[str, Any],
    title: str = ""
) -> str:
    """
    Generate chart as base64 encoded image.
    """
    plt.figure(figsize=(10, 6))
    
    if chart_type == "metrics":
        metrics_data = data.get("data", [])
        names = [d["name"] for d in metrics_data]
        values = [d["value"] for d in metrics_data]
        
        plt.bar(names, values, color='steelblue')
        plt.title(title or "Performance Metrics")
        plt.ylabel("Score")
        plt.xticks(rotation=45, ha='right')
        
    elif chart_type == "feature_importance":
        features = list(data.keys())[:10]
        importances = [data[f] for f in features]
        
        plt.barh(features, importances, color='coral')
        plt.title(title or "Feature Importance")
        plt.xlabel("Importance")
        
    elif chart_type == "cv_scores":
        scores = data.get("scores", [])
        folds = [f"Fold {i+1}" for i in range(len(scores))]
        
        plt.plot(folds, scores, marker='o', linewidth=2, markersize=8)
        plt.title(title or "Cross-Validation Scores")
        plt.ylabel("Score")
        plt.xlabel("Fold")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return image_base64


def generate_model_report(
    all_models: list,
    target: str,
    task: str,
    dataset_info: Dict[str, Any],
    trace: list = None,
    preprocessing_recommendations: list = None
) -> str:
    """
    Generate a detailed markdown report with all model explanations, comparisons, and recommendations.
    """
    report_lines = []
    
    # Header
    report_lines.append(f"# Model Training Report: Predicting {target}\n")
    report_lines.append(f"**Generated by:** Intent2Model AutoML Platform\n")
    report_lines.append(f"**Task Type:** {task.capitalize()}\n")
    report_lines.append(f"**Target Column:** {target}\n")
    report_lines.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("\n---\n")
    
    # Dataset Info
    if dataset_info:
        report_lines.append("## Dataset Overview\n")
        report_lines.append(f"- **Rows:** {dataset_info.get('n_rows', 'N/A')}")
        report_lines.append(f"- **Columns:** {dataset_info.get('n_cols', 'N/A')}")
        report_lines.append(f"- **Numeric Columns:** {len(dataset_info.get('numeric_cols', []))}")
        report_lines.append(f"- **Categorical Columns:** {len(dataset_info.get('categorical_cols', []))}")
        report_lines.append("\n---\n")
    
    # Preprocessing Recommendations
    if preprocessing_recommendations:
        report_lines.append("## Preprocessing Recommendations\n")
        for rec in preprocessing_recommendations:
            report_lines.append(f"### {rec.get('type', 'General').capitalize()}")
            report_lines.append(f"**Why:** {rec.get('why', 'N/A')}")
            report_lines.append(f"**Suggestion:** {rec.get('suggestion', 'N/A')}")
            report_lines.append("")
        report_lines.append("---\n")
    
    # Training Trace
    if trace:
        report_lines.append("## Training Process\n")
        for i, step in enumerate(trace, 1):
            report_lines.append(f"{i}. {step}")
        report_lines.append("\n---\n")
    
    # Model Comparison
    report_lines.append("## Model Comparison\n")
    report_lines.append("| Model | Primary Metric | CV Mean | CV Std | Status |")
    report_lines.append("|-------|---------------|---------|--------|--------|")
    
    for idx, model in enumerate(all_models):
        model_name = model.get('model_name', 'Unknown').replace('_', ' ').title()
        primary_metric = model.get('primary_metric')
        if primary_metric is None:
            # Try to get from metrics dict
            metrics_dict = model.get('metrics', {})
            if metrics_dict:
                # Use first metric value as fallback
                primary_metric = next(iter(metrics_dict.values()), 0)
            else:
                primary_metric = 0
        primary_metric = float(primary_metric) if primary_metric is not None else 0.0
        cv_mean = float(model.get('cv_mean', 0)) if model.get('cv_mean') is not None else 0.0
        cv_std = float(model.get('cv_std', 0)) if model.get('cv_std') is not None else 0.0
        status = "⭐ Best" if idx == 0 else "Available"
        
        report_lines.append(f"| {model_name} | {primary_metric:.4f} | {cv_mean:.4f} | {cv_std:.4f} | {status} |")
    
    report_lines.append("\n---\n")
    
    # Detailed Model Explanations
    report_lines.append("## Detailed Model Analysis\n")
    
    for idx, model in enumerate(all_models):
        model_name = model.get('model_name', 'Unknown').replace('_', ' ').title()
        is_best = idx == 0
        
        report_lines.append(f"### {model_name} {'⭐ (Recommended)' if is_best else ''}\n")
        
        # Metrics
        metrics_dict = model.get('metrics', {})
        if metrics_dict:
            report_lines.append("#### Performance Metrics\n")
            for metric_name, value in metrics_dict.items():
                if value is not None:
                    try:
                        report_lines.append(f"- **{metric_name}:** {float(value):.4f}")
                    except (ValueError, TypeError):
                        report_lines.append(f"- **{metric_name}:** {value}")
            report_lines.append("")
        
        # Primary metric highlight
        if model.get('primary_metric') is not None:
            report_lines.append(f"**Primary Metric Score:** {float(model.get('primary_metric', 0)):.4f}\n")
        
        # Full Explanation (if available from LLM explainer)
        explanation = model.get('explanation', {})
        if explanation:
            if isinstance(explanation, dict):
                if explanation.get('explanation'):
                    report_lines.append("#### Analysis\n")
                    report_lines.append(explanation.get('explanation'))
                    report_lines.append("")
                
                if explanation.get('strengths'):
                    report_lines.append("#### Strengths\n")
                    report_lines.append(explanation.get('strengths'))
                    report_lines.append("")
                
                if explanation.get('recommendation'):
                    report_lines.append("#### Recommendations\n")
                    report_lines.append(explanation.get('recommendation'))
                    report_lines.append("")
            elif isinstance(explanation, str):
                report_lines.append("#### Analysis\n")
                report_lines.append(explanation)
                report_lines.append("")
        
        # CV Scores
        if model.get('cv_scores'):
            report_lines.append("#### Cross-Validation Scores\n")
            cv_scores = model.get('cv_scores', [])
            report_lines.append(f"Individual fold scores: {', '.join([f'{s:.4f}' for s in cv_scores])}")
            report_lines.append(f"Mean: {model.get('cv_mean', 0):.4f} ± {model.get('cv_std', 0):.4f}")
            report_lines.append("")
        
        # Feature Importance (if available)
        if model.get('feature_importance'):
            report_lines.append("#### Top Feature Importance\n")
            feat_imp = model.get('feature_importance', {})
            sorted_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, imp in sorted_features:
                report_lines.append(f"- **{feat}:** {imp:.4f}")
            report_lines.append("")
        
        report_lines.append("---\n")
    
    # Conclusion
    report_lines.append("## Conclusion\n")
    if all_models:
        best_model = all_models[0]
        best_name = best_model.get('model_name', 'Unknown').replace('_', ' ').title()
        report_lines.append(f"The **{best_name}** model performed best for this dataset and task.")
        report_lines.append("Consider the detailed analysis above when making deployment decisions.")
        report_lines.append("\nFor questions or further analysis, refer to the generated Jupyter notebook.")
    
    return "\n".join(report_lines)


def _generate_metrics_evaluation_code(plan, task: str) -> List[str]:
    """Generate metrics evaluation code from AutoMLPlan."""
    lines = []
    
    primary = plan.primary_metric
    additional = plan.additional_metrics
    
    is_classification = "classification" in task
    
    # Primary metric - handle variations like "f1_score_macro", "f1", etc.
    if is_classification:
        if primary == "accuracy" or "accuracy" in primary.lower():
            lines.append(f"primary_score = accuracy_score(y_test, y_pred)\n")
            lines.append(f"print(f'Primary Metric (Accuracy): {{primary_score:.4f}}')\n")
        elif "precision" in primary.lower():
            lines.append(f"from sklearn.metrics import precision_score\n")
            avg = "macro" if "macro" in primary.lower() else ("micro" if "micro" in primary.lower() else "weighted")
            lines.append(f"primary_score = precision_score(y_test, y_pred, average='{avg}', zero_division=0)\n")
            lines.append(f"print(f'Primary Metric (Precision {avg}): {{primary_score:.4f}}')\n")
        elif "recall" in primary.lower():
            lines.append(f"from sklearn.metrics import recall_score\n")
            avg = "macro" if "macro" in primary.lower() else ("micro" if "micro" in primary.lower() else "weighted")
            lines.append(f"primary_score = recall_score(y_test, y_pred, average='{avg}', zero_division=0)\n")
            lines.append(f"print(f'Primary Metric (Recall {avg}): {{primary_score:.4f}}')\n")
        elif "f1" in primary.lower():
            lines.append(f"from sklearn.metrics import f1_score\n")
            avg = "macro" if "macro" in primary.lower() else ("micro" if "micro" in primary.lower() else "weighted")
            lines.append(f"primary_score = f1_score(y_test, y_pred, average='{avg}', zero_division=0)\n")
            lines.append(f"print(f'Primary Metric (F1 {avg}): {{primary_score:.4f}}')\n")
        elif primary == "roc_auc":
            lines.append(f"from sklearn.metrics import roc_auc_score\n")
            lines.append(f"try:\n")
            lines.append(f"    y_pred_proba = pipeline.predict_proba(X_test)\n")
            lines.append(f"    if len(np.unique(y_test)) == 2:\n")
            lines.append(f"        primary_score = roc_auc_score(y_test, y_pred_proba[:, 1])\n")
            lines.append(f"    else:\n")
            lines.append(f"        primary_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')\n")
            lines.append(f"    print(f'Primary Metric (ROC-AUC): {{primary_score:.4f}}')\n")
            lines.append(f"except Exception as e:\n")
            lines.append(f"    print(f'ROC-AUC not available: {{e}}')\n")
        
        # Additional metrics
        lines.append("\n# Additional metrics from plan:\n")
        for metric in additional:
            if metric == "precision" and primary != "precision":
                lines.append(f"print(f'Precision: {{precision_score(y_test, y_pred, average=\"macro\", zero_division=0):.4f}}')\n")
            elif metric == "recall" and primary != "recall":
                lines.append(f"print(f'Recall: {{recall_score(y_test, y_pred, average=\"macro\", zero_division=0):.4f}}')\n")
            elif metric == "f1" and primary not in ["f1", "f1_score"]:
                lines.append(f"print(f'F1: {{f1_score(y_test, y_pred, average=\"macro\", zero_division=0):.4f}}')\n")
        
        lines.append("\nprint(classification_report(y_test, y_pred))\n")
    else:
        # Regression
        if primary == "rmse":
            lines.append(f"from sklearn.metrics import mean_squared_error\n")
            lines.append(f"primary_score = np.sqrt(mean_squared_error(y_test, y_pred))\n")
            lines.append(f"print(f'Primary Metric (RMSE): {{primary_score:.4f}}')\n")
        elif primary == "mae":
            lines.append(f"from sklearn.metrics import mean_absolute_error\n")
            lines.append(f"primary_score = mean_absolute_error(y_test, y_pred)\n")
            lines.append(f"print(f'Primary Metric (MAE): {{primary_score:.4f}}')\n")
        elif primary in ["r2", "r2_score"]:
            lines.append(f"primary_score = r2_score(y_test, y_pred)\n")
            lines.append(f"print(f'Primary Metric (R²): {{primary_score:.4f}}')\n")
        
        # Additional metrics
        lines.append("\n# Additional metrics from plan:\n")
        for metric in additional:
            if metric == "mae" and primary != "mae":
                lines.append(f"print(f'MAE: {{mean_absolute_error(y_test, y_pred):.4f}}')\n")
            elif metric in ["r2", "r2_score"] and primary not in ["r2", "r2_score"]:
                lines.append(f"print(f'R²: {{r2_score(y_test, y_pred):.4f}}')\n")
            elif metric == "rmse" and primary != "rmse":
                lines.append(f"print(f'RMSE: {{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}}')\n")
    
    return lines
