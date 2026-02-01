"""
Execution planner for agentic loop.

- get_structural_plan_and_first_execution: once per dataset → (StructuralPlan, ExecutionPlan).
- derive_execution_plan: per attempt (after failure) → apply RepairPlan to previous ExecutionPlan.
- apply_repair_plan: RepairPlan + ExecutionPlan → new ExecutionPlan.

Compiler accepts ONLY ExecutionPlan. No silent reuse; no heuristics inside compiler.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from schemas.pipeline_schema import (
    AutoMLPlan,
    ExecutionPlan,
    FeatureTransform,
    ModelCandidate,
    RepairPlan,
    StructuralPlan,
    FeatureKind,
)


def automl_plan_to_structural_and_execution(plan: AutoMLPlan) -> Tuple[StructuralPlan, ExecutionPlan]:
    """
    Split an AutoMLPlan (from LLM or fallback) into StructuralPlan and ExecutionPlan.
    Used once per dataset to bootstrap the agentic loop.
    """
    feature_semantics: Dict[str, FeatureKind] = {}
    feature_confidence: Dict[str, float] = {}
    for ft in plan.feature_transforms:
        feature_semantics[ft.name] = ft.kind
        feature_confidence[ft.name] = ft.kind_confidence

    structural = StructuralPlan(
        plan_schema_version=plan.plan_schema_version,
        inferred_target=plan.inferred_target,
        target_confidence=plan.target_confidence,
        alternative_targets=plan.alternative_targets,
        task_type=plan.task_type,
        task_confidence=plan.task_confidence,
        feature_semantics=feature_semantics,
        feature_confidence=feature_confidence,
        leakage_candidates=[],  # Could be derived from plan if we had it
        dataset_warnings=[],
    )

    target_transform = (plan.target_transformation or "none").lower()
    if target_transform not in ("none", "log", "log1p", "quantile", "robust"):
        target_transform = "none"

    execution = ExecutionPlan(
        target_transformation=target_transform,
        feature_transforms=list(plan.feature_transforms),
        model_candidates=list(plan.model_candidates),
        primary_metric=plan.primary_metric,
        additional_metrics=list(plan.additional_metrics),
        execution_confidence=plan.metric_selection_confidence,
        plan_quality=plan.plan_quality,
        planning_source="llm" if plan.planning_source == "llm" else "fallback",
        reasoning_md=getattr(plan, "transformation_strategy_md", "") or "",
    )
    return structural, execution


def apply_repair_plan(execution_plan: ExecutionPlan, repair: RepairPlan) -> ExecutionPlan:
    """
    Apply a RepairPlan (structured diff) to an ExecutionPlan to produce the next ExecutionPlan.
    """
    ep_dict = execution_plan.model_dump()
    ep_dict["reasoning_md"] = execution_plan.reasoning_md + "\n[Repair applied]"

    if repair.change_target_transform is not None:
        ep_dict["target_transformation"] = repair.change_target_transform

    # Feature transforms: drop, then add, then change_encoding
    feature_transforms = list(execution_plan.feature_transforms)

    if repair.drop_features:
        drop_set = set(repair.drop_features)
        feature_transforms = [ft for ft in feature_transforms if ft.name not in drop_set]

    if repair.add_features:
        existing_names = {ft.name for ft in feature_transforms}
        for name in repair.add_features:
            if name not in existing_names:
                feature_transforms.append(
                    FeatureTransform(
                        name=name,
                        inferred_dtype="unknown",
                        kind="unknown",
                        drop=False,
                        encode="none",
                        scale="none",
                        impute="none",
                        notes_md="Added by repair",
                        transform_confidence=0.5,
                    )
                )
                existing_names.add(name)

    if repair.change_encoding:
        encoding_map = repair.change_encoding
        new_transforms = []
        for ft in feature_transforms:
            ft_dict = ft.model_dump() if hasattr(ft, "model_dump") else dict(ft)
            if ft.name in encoding_map:
                enc = encoding_map[ft.name]
                if enc in ("one_hot", "ordinal", "frequency"):
                    ft_dict["encode"] = enc
            new_transforms.append(ft_dict)
        ep_dict["feature_transforms"] = new_transforms
    else:
        ep_dict["feature_transforms"] = [
            ft.model_dump() if hasattr(ft, "model_dump") else ft for ft in feature_transforms
        ]

    # Model candidates: replace, add, remove, reorder
    if repair.replace_model is not None:
        ep_dict["model_candidates"] = [
            ModelCandidate(model_name=repair.replace_model, reason_md="Replaced by repair", params={}).model_dump()
        ]
    else:
        model_candidates = [
            mc.model_dump() if hasattr(mc, "model_dump") else mc
            for mc in execution_plan.model_candidates
        ]

        if repair.remove_models:
            remove_set = set(repair.remove_models)
            model_candidates = [mc for mc in model_candidates if mc.get("model_name") not in remove_set]

        if repair.add_models:
            existing = {mc.get("model_name") for mc in model_candidates}
            for m in repair.add_models:
                if m not in existing:
                    model_candidates.append(
                        ModelCandidate(model_name=m, reason_md="Added by repair", params={}).model_dump()
                    )
                    existing.add(m)

        if repair.reorder_models:
            order = {name: i for i, name in enumerate(repair.reorder_models)}
            model_candidates.sort(key=lambda mc: order.get(mc.get("model_name"), 999))

        ep_dict["model_candidates"] = model_candidates

    return ExecutionPlan(**ep_dict)


def plan_changes_to_repair_plan(plan_changes: Dict[str, Any]) -> RepairPlan:
    """
    Convert diagnosis.plan_changes (or diagnosis.repair_plan) dict to RepairPlan.
    Used after failure to get structured diff for next ExecutionPlan.
    """
    if not plan_changes:
        return RepairPlan()
    change_target = None
    if plan_changes.get("target_transformation"):
        t = plan_changes["target_transformation"]
        if t in ("log", "log1p", "quantile", "robust"):
            change_target = t
    drop_features = None
    if plan_changes.get("drop_features") and isinstance(plan_changes["drop_features"], list):
        drop_features = list(plan_changes["drop_features"])
    elif plan_changes.get("feature_transforms"):
        drops = [
            c.get("feature") for c in plan_changes["feature_transforms"]
            if isinstance(c, dict) and c.get("action") == "drop" and c.get("feature")
        ]
        if drops:
            drop_features = drops
    add_features = None
    if plan_changes.get("add_features") and isinstance(plan_changes["add_features"], list):
        add_features = list(plan_changes["add_features"])
    replace_model = plan_changes.get("model_selection")
    add_models = plan_changes.get("add_models")
    if add_models is not None and not isinstance(add_models, list):
        add_models = [add_models] if add_models else None
    remove_models = plan_changes.get("remove_models")
    if remove_models is not None and not isinstance(remove_models, list):
        remove_models = [remove_models] if remove_models else None
    reorder_models = None
    if plan_changes.get("reorder_models") and isinstance(plan_changes["reorder_models"], list):
        reorder_models = list(plan_changes["reorder_models"])
    change_encoding = None
    if plan_changes.get("change_encoding") and isinstance(plan_changes["change_encoding"], dict):
        change_encoding = {
            k: v for k, v in plan_changes["change_encoding"].items()
            if v in ("one_hot", "ordinal", "frequency")
        }
        if not change_encoding:
            change_encoding = None
    return RepairPlan(
        change_target_transform=change_target,
        drop_features=drop_features,
        add_features=add_features,
        replace_model=replace_model,
        add_models=add_models,
        remove_models=remove_models,
        reorder_models=reorder_models,
        change_encoding=change_encoding,
    )


def execution_plan_to_config(execution_plan: ExecutionPlan, task: str) -> Dict[str, Any]:
    """
    Convert ExecutionPlan to the config dict expected by pipeline_builder and trainer.
    Compiler does not call this; executor does. Kept here for cohesion.
    """
    return {
        "task": task,
        "target_transformation": execution_plan.target_transformation if execution_plan.target_transformation != "none" else None,
        "feature_transforms": [ft.model_dump() if hasattr(ft, "model_dump") else ft for ft in execution_plan.feature_transforms],
        "primary_metric": execution_plan.primary_metric,
        "model_candidates": [mc.model_name for mc in execution_plan.model_candidates],
        "plan_quality": execution_plan.plan_quality,
        "planning_source": execution_plan.planning_source,
    }


def merged_plan_dict(structural: StructuralPlan, execution: ExecutionPlan) -> Dict[str, Any]:
    """
    Build an AutoMLPlan-compatible dict from StructuralPlan + ExecutionPlan.
    Used for backward compat: notebook and API still expect a single plan dict.
    """
    return {
        "plan_schema_version": structural.plan_schema_version,
        "inferred_target": structural.inferred_target,
        "target_confidence": structural.target_confidence,
        "alternative_targets": structural.alternative_targets,
        "task_type": structural.task_type,
        "task_confidence": structural.task_confidence,
        "task_inference_md": f"Task: {structural.task_type} (structural plan).",
        "dataset_intelligence_md": f"Target: {structural.inferred_target}. Feature semantics from structural plan.",
        "transformation_strategy_md": execution.reasoning_md or "Execution plan transforms.",
        "model_selection_md": f"Models: {[mc.model_name for mc in execution.model_candidates]}.",
        "training_validation_md": "Cross-validation with execution plan metrics.",
        "error_behavior_analysis_md": "Error analysis from evaluation.",
        "explainability_md": "Feature importance when available.",
        "primary_metric": execution.primary_metric,
        "additional_metrics": list(execution.additional_metrics),
        "metric_selection_confidence": execution.execution_confidence,
        "feature_transforms": [ft.model_dump() if hasattr(ft, "model_dump") else ft for ft in execution.feature_transforms],
        "model_candidates": [mc.model_dump() if hasattr(mc, "model_dump") else mc for mc in execution.model_candidates],
        "planning_source": execution.planning_source,
        "planning_error": None,
        "plan_quality": execution.plan_quality,
        "target_transformation": execution.target_transformation if execution.target_transformation != "none" else None,
    }
