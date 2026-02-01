"""
Notebook code agent: LLM writes the whole notebook (Cursor-style).
Full notebook = intro, reasoning, code, save, predict — crazy good at reasoning and descriptive.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


def _to_source_lines(text: str) -> List[str]:
    """Turn string into notebook source list (each line ending with \\n)."""
    return [line + "\n" for line in text.strip().split("\n")] if text.strip() else []


def _parse_full_notebook_blocks(response: str) -> List[Dict[str, Any]]:
    """
    Parse LLM response into notebook cells. Expects alternating ```markdown and ```python blocks.
    Returns list of {"cell_type": "markdown"|"code", "source": [...]}.
    """
    cells: List[Dict[str, Any]] = []
    # Match ```markdown ... ``` or ```python ... ``` or ``` ... ``` (treat as code)
    pattern = re.compile(r"```(markdown|md|python)?\s*\n(.*?)```", re.DOTALL)
    for m in pattern.finditer(response):
        lang = (m.group(1) or "").strip().lower()
        content = (m.group(2) or "").strip()
        if not content:
            continue
        lines = _to_source_lines(content)
        if not lines:
            continue
        if lang in ("markdown", "md"):
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "source": lines})
    return cells


def generate_full_notebook_llm(
    execution_plan: Dict[str, Any],
    structural_plan: Optional[Dict[str, Any]],
    columns: List[str],
    target: str,
    task: str,
    model_name: str,
    metrics: Optional[Dict[str, float]] = None,
    llm_provider: str = "gemini",
) -> Optional[List[Dict[str, Any]]]:
    """
    Generate the COMPLETE rest of the notebook via LLM (after imports + load data).
    Returns list of notebook cell dicts (markdown + code). CRAZY GOOD: reasoning, self-critique, descriptive.
    """
    try:
        from utils.api_key_manager import get_api_key
        from agents.llm_interface import LLMInterface
    except Exception:
        return None

    ep = execution_plan or {}
    sp = structural_plan or {}
    feature_transforms = ep.get("feature_transforms") or []
    primary_metric = ep.get("primary_metric") or ("accuracy" if "classification" in task else "r2")
    models_str = ", ".join(
        m.get("model_name", "?") if isinstance(m, dict) else getattr(m, "model_name", "?")
        for m in (ep.get("model_candidates") or [])[:6]
    )
    transform_summary = _summarize_transforms(feature_transforms, columns)
    metrics_str = ""
    if metrics:
        metrics_str = "\n".join(f"- **{k}:** {v}" for k, v in list(metrics.items())[:15] if isinstance(v, (int, float)))

    system_prompt = """You are an expert ML engineer and educator. You generate a COMPLETE, production-grade Jupyter notebook that is CRAZY GOOD — amazing at reasoning with itself and making things better.

CRITICAL BEHAVIORS:
1. **Reason with yourself** — In markdown, think step-by-step: "We're predicting X because...", "We chose this model because...", "A risk here is... so we...". Show your reasoning chain. Ask and answer: "What could go wrong?" and "How can we improve this?"
2. **Self-critique** — After key steps, add a short reflection: "We're watching for overfitting because the dataset is small.", "We scaled numerics so that distance-based models don't dominate." Make the notebook feel like a thoughtful engineer explaining and improving as they go.
3. **Descriptive & educational** — Use rich markdown: clear headers, bullet points, **bold** for key terms. Every code block should have a markdown cell before (or after) that explains WHAT we're doing and WHY. A reader should learn from this notebook.
4. **Production-grade code** — Clean, runnable Python. Use ONLY these variable names: df (already loaded), X, y, X_train, y_train, X_test, y_test, preprocessor, model, pipeline, y_pred. For classification use le = LabelEncoder() on y. Code must run after a cell that defined `df`.

5. **Notebook as a story** — Every cell must answer ONE question: Why this model? Why this transform? Why this failed? Why this worked? No "dump everything" cells. Narrative > completeness.

Output format: alternate between markdown and code. Use ```markdown ... ``` for markdown cells and ```python ... ``` for code cells. Order:

1. **Title + intro** (markdown): What we're predicting, why it matters, one-line context. Be engaging.
2. **Reasoning & plan** (markdown): Step-by-step reasoning — why this target, why these features, why this model. Include "What we're watching for" and one self-critique. Be descriptive.
3. **Prepare data** (python): X = df.drop(columns=[target]), y = df[target]. Classification: LabelEncoder on y. train_test_split(X, y, test_size=0.2, random_state=42). Print shapes.
4. **Preprocessing** (markdown): Why this preprocessing (scale, encode, impute). One line on "what could go wrong" (e.g. leakage, missing values).
5. **Preprocessing** (python): ColumnTransformer — numeric: impute+scale, categorical: onehot. Define preprocessor.
6. **Model** (markdown): Why this model for this task. Brief alternative considered.
7. **Model** (python): Instantiate model (e.g. RandomForestRegressor, LogisticRegression). Define model.
8. **Pipeline & train** (markdown): What the pipeline does and why we combine preprocessor + model.
9. **Pipeline & train** (python): pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)]). fit(X_train, y_train). y_pred = pipeline.predict(X_test).
10. **Evaluate** (markdown): What we're measuring and why. One line on how to interpret the metric.
11. **Evaluate** (python): Print primary metric (r2_score, accuracy_score, etc.) and optionally classification_report or MSE. Descriptive comments in code.
12. **Save & predict** (markdown): How to save and use the model in practice.
13. **Save & predict** (python): pickle.dump(pipeline, ...). Example: new_row = pd.DataFrame({...}); pipeline.predict(new_row). Print prediction.

Output ONLY the sequence of ```markdown and ```python blocks. No other text before or after."""

    prompt = f"""Generate the complete notebook (everything after data load). The dataframe is already in variable `df`.

**Target:** {target}
**Task:** {task}
**Primary metric:** {primary_metric}
**Columns:** {columns}
**Chosen model:** {model_name}
**Model candidates (this attempt):** {models_str}

**Feature transform plan:**
{transform_summary}

**Structural plan (context):**
- Inferred target: {sp.get('inferred_target', '—')} (confidence: {sp.get('target_confidence', 1):.2f})
- Task type: {sp.get('task_type', '—')}
- Feature semantics: {len(sp.get('feature_semantics', {}))} features

{f'**Observed metrics (for narrative):**\\n{metrics_str}' if metrics_str else ''}

Generate the full notebook content as alternating ```markdown and ```python blocks. Be CRAZY GOOD: reason step-by-step, self-critique, and make every section descriptive. Every cell must answer one question (why this model, why this transform, why this failed, why this worked). Narrative over completeness. Start with the title/intro markdown cell."""

    try:
        api_key = get_api_key(provider=llm_provider)
        llm = LLMInterface(provider=llm_provider, api_key=api_key)
        response = llm.generate(prompt, system_prompt)
        if not response or not response.strip():
            return None
        cells = _parse_full_notebook_blocks(response)
        if len(cells) >= 4:
            return cells
        return None
    except Exception as e:
        print(f"⚠️  Full notebook LLM generation failed: {e}")
        return None


def _extract_python_blocks(text: str) -> List[List[str]]:
    """Extract ```python ... ``` blocks; return list of cell sources (each = list of lines with \\n)."""
    blocks: List[List[str]] = []
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    for m in pattern.finditer(text):
        code = m.group(1).strip()
        if not code:
            continue
        lines = [line + "\n" for line in code.split("\n")]
        if lines:
            blocks.append(lines)
    return blocks


def generate_notebook_code_llm(
    execution_plan: Dict[str, Any],
    columns: List[str],
    target: str,
    task: str,
    model_name: str,
    llm_provider: str = "gemini",
) -> Optional[List[List[str]]]:
    """
    Ask the LLM to generate Python code for the notebook (Cursor-style).
    Returns list of code cell sources (each = list of lines ending with \\n), or None on failure.
    """
    try:
        from utils.api_key_manager import get_api_key
        from agents.llm_interface import LLMInterface
    except Exception:
        return None

    ep = execution_plan or {}
    feature_transforms = ep.get("feature_transforms") or []
    models_str = ", ".join(
        m.get("model_name", "?") if isinstance(m, dict) else getattr(m, "model_name", "?")
        for m in (ep.get("model_candidates") or [])[:5]
    )
    primary_metric = ep.get("primary_metric") or ("accuracy" if "classification" in task else "r2")

    system_prompt = """You are an expert ML engineer. You write Python code for a Jupyter notebook.
Output ONLY valid Python code. Use these variable names exactly so the notebook runs:
- df: DataFrame already loaded in a previous cell
- X, y: features and target (y = df[target], X = df.drop(columns=[target]))
- X_train, y_train, X_test, y_test: from train_test_split(X, y, ...)
- preprocessor: ColumnTransformer (numeric + categorical)
- model: the sklearn model instance
- pipeline: Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
- pipeline.fit(X_train, y_train), y_pred = pipeline.predict(X_test)
Do not use markdown. Output 4 separate code blocks wrapped in ```python ... ``` in this order:
1) Preprocessing: build X, y from df; build preprocessor (ColumnTransformer) for numeric and categorical columns.
2) Model: instantiate the model (e.g. RandomForestRegressor, LogisticRegression).
3) Pipeline: create sklearn Pipeline with preprocessor and model; then train_test_split; fit; predict.
4) Evaluate: print primary metric (e.g. r2_score, accuracy_score) and optionally other metrics."""

    prompt = f"""Generate notebook code for this ML task.

**Target column:** {target}
**Task:** {task}
**Primary metric:** {primary_metric}
**Columns:** {columns}
**Chosen model:** {model_name}
**All model candidates for this attempt:** {models_str}

**Feature transform summary (from plan):**
{_summarize_transforms(feature_transforms, columns)}

Write 4 Python code blocks (each in ```python ... ```):
1. Preprocessing: X = df.drop(columns=['{target}']), y = df['{target}']; build ColumnTransformer (numeric: impute+scale, categorical: impute+onehot).
2. Model: instantiate the model for {task} (e.g. {"LogisticRegression()" if "classification" in task else "RandomForestRegressor()"} or the chosen one).
3. Pipeline: Pipeline with preprocessor and model; train_test_split(X, y, test_size=0.2, random_state=42); pipeline.fit(X_train, y_train); y_pred = pipeline.predict(X_test).
4. Evaluate: print {primary_metric} and any other metrics. Use sklearn.metrics.

Output only the 4 ```python ... ``` blocks, nothing else."""

    try:
        api_key = get_api_key(provider=llm_provider)
        llm = LLMInterface(provider=llm_provider, api_key=api_key)
        response = llm.generate(prompt, system_prompt)
        if not response or not response.strip():
            return None
        blocks = _extract_python_blocks(response)
        if len(blocks) >= 3:
            # If we got 3 blocks we can merge train+eval; if 4 we're good
            return blocks
        # Single block? Split by big comments or use as one cell
        if len(blocks) == 1:
            return blocks
        if len(blocks) == 2:
            return blocks
        return None
    except Exception as e:
        print(f"⚠️  Notebook LLM code generation failed: {e}")
        return None


def _summarize_transforms(feature_transforms: List[Any], columns: List[str]) -> str:
    if not feature_transforms:
        return "Use numeric columns: scale; categorical: one-hot. Drop target."
    lines = []
    for ft in feature_transforms[:20]:
        if isinstance(ft, dict):
            name = ft.get("name", "?")
            drop = ft.get("drop", False)
            kind = ft.get("kind", "unknown")
            enc = ft.get("encode", "none")
            scale = ft.get("scale", "none")
            if drop:
                lines.append(f"- {name}: drop")
            else:
                lines.append(f"- {name}: kind={kind}, encode={enc}, scale={scale}")
        else:
            lines.append(str(ft)[:80])
    return "\n".join(lines) if lines else "Standard preprocessing for numeric and categorical."
