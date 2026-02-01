"""
Reasoning Diff + What to Try Next + Bottleneck Detection + Honesty Gate.

ML Engineer Agent: judgment over features. Produces after every run:
- Reasoning Diff (what changed, why, expected vs actual, hypothesis validated)
- What I'd Try Next (3–4 concrete experiments: what / why / signal)
- Forked-experiment language (this run isolates X; this fork tests Y)
- Data bottleneck detection (target noise, feature insufficiency, size, irreducible)
- Metric reality check (call out misleading metrics; downgrade confidence)
- Hard stop when appropriate ("Further modeling unlikely without better data").
"""

from typing import Dict, Any, List, Optional


def reasoning_diff_md(
    prev_state: Dict[str, Any],
    current_result: Dict[str, Any],
    session: Dict[str, Any],
    task: str,
    target: str,
    primary: str,
    same_as_before: bool,
) -> str:
    """
    Structured Reasoning Diff: what changed vs previous run, why, expected vs actual, hypothesis.
    Human-readable, model-agnostic. Uses forked-experiment language.
    """
    prev_ms = prev_state or {}
    prev_attempt = prev_ms.get("attempt_number") or 0
    attempt = (prev_attempt + 1) if prev_attempt else 1

    prev_model = prev_ms.get("current_model") or prev_ms.get("previous_model")
    current_model = current_result.get("model_name")
    if isinstance(current_model, dict):
        current_model = current_model.get("model_name") if current_model else None
    all_models = current_result.get("all_models") or []
    tried = [m.get("model_name") for m in all_models if m.get("model_name")]

    lines = ["### Reasoning diff\n"]
    # What changed
    if prev_model and current_model and prev_model != current_model:
        lines.append(f"- **What changed:** Previous best was **{prev_model}**; this run selected **{current_model}**. This fork isolates the effect of trying a different model family.")
    elif tried:
        lines.append(f"- **What changed:** This run tried: {', '.join(tried)}. Best: **{current_model}**.")
    else:
        lines.append(f"- **What changed:** First run; baseline established with **{current_model}**.")

    # Why it was changed
    if attempt > 1 and not prev_model:
        lines.append("- **Why:** First training attempt for this target.")
    elif attempt > 1:
        lines.append("- **Why:** You requested another run (e.g. \"try something stronger\"); we extended the model set to test whether a more complex model would improve.")

    # Expected vs actual
    if same_as_before and len(all_models) > 1:
        lines.append("- **Expected:** Stronger or additional models might improve the metric.")
        lines.append("- **Actual:** Metric stayed effectively the same. The hypothesis that \"more complex models would help\" was **not validated** for this target with current features.")
    elif same_as_before:
        lines.append("- **Actual:** Result in line with previous run; no regression.")
    else:
        lines.append("- **Actual:** Best metric for this run: **" + primary + "**.")

    return "\n".join(lines) + "\n\n"


def what_to_try_next(
    prev_state: Dict[str, Any],
    current_result: Dict[str, Any],
    session: Dict[str, Any],
    task: str,
    target: str,
    same_as_before: bool,
    failed_models: List[Dict[str, Any]],
    error_analysis: Dict[str, Any],
) -> str:
    """
    3–4 concrete next experiments. Each: what to change, why it might help, what signal confirms/falsifies.
    Guidance only; do not auto-run.
    """
    suggestions = []
    all_models = current_result.get("all_models") or []

    # If some models failed (e.g. XGBoost), suggest fixing that first
    for m in failed_models[:2]:
        name = m.get("model_name", "?")
        err = (m.get("error") or "")[:80]
        if "import" in err.lower() or "not installed" in err.lower():
            suggestions.append({
                "what": f"Install or fix **{name}** (e.g. `pip install xgboost`).",
                "why": "That model was skipped; including it may improve results.",
                "signal": "Run again; that model should appear in the comparison and either beat the current best or not.",
            })
        else:
            suggestions.append({
                "what": f"Debug **{name}**: {err}.",
                "why": "Resolving the error lets this fork test whether that model helps.",
                "signal": "Model trains; compare its metric to current best.",
            })

    # If plateau, suggest data/preprocessing
    if same_as_before and len(all_models) >= 2:
        suggestions.append({
            "what": "Try **different preprocessing** (e.g. scaling, encoding, or drop a feature).",
            "why": "Model choice may not be the bottleneck; preprocessing can change the effective feature space.",
            "signal": "Metric improves or a different model wins; if not, bottleneck may be data.",
        })
        suggestions.append({
            "what": "Add or engineer **one new feature** (e.g. interaction, domain-derived).",
            "why": "Feature insufficiency often caps performance; one good feature can break a plateau.",
            "signal": "Metric improves; if it does not, consider target noise or dataset size.",
        })
        suggestions.append({
            "what": "Explicitly test **target noise / label quality** (e.g. check mislabels, borderline cases).",
            "why": "If labels are noisy, headline accuracy/R² can be misleading; fixing labels may be the only lever.",
            "signal": "Clean a subset and retrain; if metric jumps, noise was the bottleneck.",
        })

    # If first run or not plateau, still give 2–3 options
    if len(suggestions) < 3:
        suggestions.append({
            "what": "Try **stronger or different models** (\"try something stronger\").",
            "why": "This isolates the effect of model choice.",
            "signal": "A different model wins; if all perform similarly, model is likely not the limiting factor.",
        })
        suggestions.append({
            "what": "Try a **different target** or **drop a feature** (\"use X as target\", \"drop Y\").",
            "why": "This fork tests whether the bottleneck is target choice or a leaky/noisy feature.",
            "signal": "Metric changes meaningfully; if it drops a lot, previous target or feature was important.",
        })

    # Cap at 4
    suggestions = suggestions[:4]
    lines = ["### What I'd try next (guidance only)\n"]
    for i, s in enumerate(suggestions, 1):
        lines.append(f"{i}. **{s['what']}**")
        lines.append(f"   - Why: {s['why']}")
        lines.append(f"   - Signal: {s['signal']}")
    return "\n".join(lines) + "\n\n"


def bottleneck_and_honesty_md(
    same_as_before: bool,
    primary_float: Optional[float],
    task: str,
    metrics: Dict[str, Any],
    error_analysis: Dict[str, Any],
    n_rows: int,
    n_features: int,
) -> str:
    """
    Data bottleneck detection + metric reality check + hard stop when appropriate.
    """
    lines = []

    # Bottleneck detection when plateau
    if same_as_before:
        lines.append("**Bottleneck check:** Performance did not improve. Consider whether the limit is:")
        lines.append("- **Target noise** — labels inconsistent or borderline.")
        lines.append("- **Feature insufficiency** — missing a signal that would separate outcomes.")
        lines.append("- **Dataset size** — too few samples for the model class.")
        lines.append("- **Irreducible error** — the problem may not be predictable from these inputs.")
        lines.append("If multiple model families give similar results, **the model is likely not the limiting factor**.\n")

    # Metric reality check: call out when headline metric might mislead
    if task == "classification" and metrics:
        acc = metrics.get("accuracy")
        cm = error_analysis.get("confusion_matrix")
        class_labels = error_analysis.get("class_labels") or []
        if acc is not None and cm and len(cm) > 1:
            try:
                import numpy as np
                cm_arr = np.array(cm)
            except Exception:
                cm_arr = None
            if cm_arr is not None:
                row_sums = cm_arr.sum(axis=1)
                row_sums = np.where(row_sums == 0, 1, row_sums)
                recalls = np.diag(cm_arr) / row_sums
                if len(recalls) > 0 and float(np.min(recalls)) < 0.5 and acc and acc > 0.8:
                    worst_idx = int(np.argmin(recalls))
                    worst_class = class_labels[worst_idx] if worst_idx < len(class_labels) else f"class_{worst_idx}"
                    lines.append(f"**Reality check:** Accuracy is **{acc:.1%}** but **{worst_class}** has low recall ({recalls[worst_idx]:.1%}). Headline metric can be misleading; check the confusion matrix.\n")

    if task == "regression" and metrics:
        r2 = metrics.get("r2")
        rmse = metrics.get("rmse") or metrics.get("RMSE")
        if r2 is not None and r2 > 0.7 and rmse is not None:
            # Could compare rmse to target std; if rmse is large vs std, say so
            target_std = metrics.get("target_std")
            if target_std and target_std > 0 and rmse > 1.5 * target_std:
                lines.append(f"**Reality check:** R² is **{r2:.3f}** but RMSE ({rmse:.3f}) is large relative to target variation. Absolute errors may still be high in practice.\n")

    # Hard stop when appropriate
    if same_as_before and n_rows < 500 and n_features <= 10:
        lines.append("**Stopping is valid:** If you've tried several options (models, preprocessing, target), **further modeling may be unlikely to help without better data** (more samples, better labels, or new features).\n")

    return "\n".join(lines) if lines else ""


def build_reasoning_block(
    prev_state: Dict[str, Any],
    current_result: Dict[str, Any],
    session: Dict[str, Any],
    task: str,
    target: str,
    primary: str,
    primary_float: Optional[float],
    same_as_before: bool,
    all_models: List[Dict[str, Any]],
    error_analysis: Dict[str, Any],
    metrics: Dict[str, Any],
    n_rows: int,
    n_features: int,
) -> str:
    """
    Single block of markdown: reasoning diff + what to try next + bottleneck + honesty.
    """
    failed_models = [m for m in (all_models or []) if m.get("failed")]
    out = reasoning_diff_md(
        prev_state, current_result, session, task, target, primary, same_as_before
    )
    out += what_to_try_next(
        prev_state, current_result, session, task, target,
        same_as_before, failed_models, error_analysis or {}
    )
    bottleneck = bottleneck_and_honesty_md(
        same_as_before, primary_float, task, metrics or {}, error_analysis or {}, n_rows, n_features
    )
    if bottleneck:
        out += bottleneck
    return out.strip()
