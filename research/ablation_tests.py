"""
Ablation tests for Intent2Model research.

Disables components to measure their impact on performance and failure rate.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from ml.profiler import profile_dataset
from ml.trainer import train_classification, train_regression
from ml.evaluator import evaluate_dataset
from agents.question_agent import generate_questions
from agents.planner_agent import plan_pipeline, _rule_based_plan
from schemas.pipeline_schema import UserIntent, PipelineConfig


class AblationConfig:
    """Configuration for ablation tests."""
    
    def __init__(
        self,
        use_questioning: bool = True,
        use_evaluator_warnings: bool = True,
        use_llm_planning: bool = True,
        use_explainer: bool = True
    ):
        self.use_questioning = use_questioning
        self.use_evaluator_warnings = use_evaluator_warnings
        self.use_llm_planning = use_llm_planning
        self.use_explainer = use_explainer


def run_ablation_test(
    df: pd.DataFrame,
    correct_target: str,
    correct_task: str,
    ablation_config: AblationConfig
) -> Dict:
    """
    Run a single ablation test with specified components disabled.
    
    Args:
        df: Input DataFrame
        correct_target: Correct target column
        correct_task: Correct task type
        ablation_config: AblationConfig specifying which components to disable
        
    Returns:
        Dictionary with results and metrics
    """
    profile = profile_dataset(df)
    
    # Questioning agent (if enabled)
    user_intent = UserIntent()
    questions_asked = 0
    
    if ablation_config.use_questioning:
        questions = generate_questions(profile)
        questions_asked = len(questions)
        # Simulate perfect user answers
        user_intent.target_column = correct_target
        user_intent.task_type = correct_task
        user_intent.priority_metric = "accuracy" if correct_task == "classification" else "r2"
    else:
        # No questioning - use defaults
        user_intent.target_column = correct_target
        user_intent.task_type = correct_task
        user_intent.priority_metric = "accuracy" if correct_task == "classification" else "r2"
    
    # Planning (with or without LLM)
    try:
        if ablation_config.use_llm_planning:
            config = plan_pipeline(profile, user_intent, llm_provider="openai")
        else:
            # Rule-based only
            config_dict = _rule_based_plan(profile, user_intent)
            config = PipelineConfig(**config_dict)
        
        planning_success = True
    except Exception as e:
        planning_success = False
        config = None
        print(f"Planning failed: {e}")
    
    # Training
    training_success = False
    metrics = None
    warnings = []
    
    if planning_success and config:
        try:
            # Evaluate (with or without warnings)
            if ablation_config.use_evaluator_warnings:
                eval_result = evaluate_dataset(df, config.target, config.task)
                warnings = eval_result["warnings"]
            else:
                warnings = []
            
            # Train
            if config.task == "classification":
                result = train_classification(
                    df, config.target, config.metric,
                    {"task": config.task, "preprocessing": config.preprocessing, "model": config.model_candidates[0]}
                )
            else:
                result = train_regression(
                    df, config.target, config.metric,
                    {"task": config.task, "preprocessing": config.preprocessing, "model": config.model_candidates[0]}
                )
            
            metrics = result["metrics"]
            training_success = True
        except Exception as e:
            print(f"Training failed: {e}")
    
    # Explanation (if enabled)
    explanation = None
    if ablation_config.use_explainer and training_success:
        try:
            from agents.explainer_agent import explain_results
            explanation = explain_results(
                metrics, warnings, result.get("feature_importance"), config.dict() if config else None
            )
        except Exception as e:
            print(f"Explanation failed: {e}")
    
    return {
        "planning_success": planning_success,
        "training_success": training_success,
        "questions_asked": questions_asked,
        "warnings_count": len(warnings),
        "metrics": metrics,
        "config": config.dict() if config else None,
        "explanation_generated": explanation is not None,
        "ablation_config": {
            "use_questioning": ablation_config.use_questioning,
            "use_evaluator_warnings": ablation_config.use_evaluator_warnings,
            "use_llm_planning": ablation_config.use_llm_planning,
            "use_explainer": ablation_config.use_explainer
        }
    }


def run_ablation_suite(
    datasets: List[Tuple[pd.DataFrame, str, str]],
    ablation_configs: Optional[List[AblationConfig]] = None
) -> pd.DataFrame:
    """
    Run full ablation test suite.
    
    Args:
        datasets: List of (DataFrame, correct_target, correct_task) tuples
        ablation_configs: List of AblationConfigs to test. If None, uses default suite.
        
    Returns:
        DataFrame with ablation results
    """
    if ablation_configs is None:
        # Default ablation configurations
        ablation_configs = [
            AblationConfig(use_questioning=True, use_evaluator_warnings=True, use_llm_planning=True, use_explainer=True),  # Full system
            AblationConfig(use_questioning=False, use_evaluator_warnings=True, use_llm_planning=True, use_explainer=True),  # No questioning
            AblationConfig(use_questioning=True, use_evaluator_warnings=False, use_llm_planning=True, use_explainer=True),  # No warnings
            AblationConfig(use_questioning=True, use_evaluator_warnings=True, use_llm_planning=False, use_explainer=True),  # No LLM planning
            AblationConfig(use_questioning=False, use_evaluator_warnings=False, use_llm_planning=False, use_explainer=False),  # Baseline (rule-based only)
        ]
    
    results = []
    
    for i, (df, correct_target, correct_task) in enumerate(datasets):
        for j, ablation_config in enumerate(ablation_configs):
            result = run_ablation_test(df, correct_target, correct_task, ablation_config)
            result["dataset_id"] = i
            result["ablation_id"] = j
            results.append(result)
    
    return pd.DataFrame(results)


def analyze_ablation_results(results_df: pd.DataFrame) -> Dict:
    """
    Analyze ablation test results.
    
    Returns:
        Dictionary with analysis metrics
    """
    analysis = {}
    
    # Success rates
    analysis["planning_success_rate"] = results_df["planning_success"].mean()
    analysis["training_success_rate"] = results_df["training_success"].mean()
    
    # Average questions asked
    analysis["avg_questions"] = results_df["questions_asked"].mean()
    
    # Average warnings
    analysis["avg_warnings"] = results_df["warnings_count"].mean()
    
    # Performance by ablation config
    ablation_groups = results_df.groupby("ablation_id")
    analysis["by_config"] = {}
    
    for ablation_id, group in ablation_groups:
        config_name = f"config_{ablation_id}"
        analysis["by_config"][config_name] = {
            "planning_success_rate": group["planning_success"].mean(),
            "training_success_rate": group["training_success"].mean(),
            "avg_questions": group["questions_asked"].mean(),
            "avg_warnings": group["warnings_count"].mean()
        }
    
    return analysis
