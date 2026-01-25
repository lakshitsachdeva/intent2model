"""
User simulation for Intent2Model research.

Simulates novice user mistakes and missing information to measure system robustness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from ml.profiler import profile_dataset
from ml.trainer import train_classification, train_regression
from ml.evaluator import evaluate_dataset
from agents.question_agent import generate_questions
from agents.planner_agent import plan_pipeline
from schemas.pipeline_schema import UserIntent, QuestionResponse


class SimulatedUser:
    """Simulates different types of users with varying levels of expertise."""
    
    def __init__(self, user_type: str = "novice"):
        """
        Initialize simulated user.
        
        Args:
            user_type: "novice", "intermediate", or "expert"
        """
        self.user_type = user_type
        self.mistakes_made = []
    
    def answer_questions(
        self,
        questions: List,
        profile: Dict,
        correct_target: str,
        correct_task: str
    ) -> UserIntent:
        """
        Simulate user answering questions (with potential mistakes).
        
        Args:
            questions: List of Question objects
            profile: Dataset profile
            correct_target: Correct target column name
            correct_task: Correct task type
            
        Returns:
            UserIntent with answers (potentially incorrect)
        """
        intent = UserIntent()
        
        for question in questions:
            answer = self._generate_answer(
                question, profile, correct_target, correct_task
            )
            
            if question.question_id == "target_column":
                intent.target_column = answer
            elif question.question_id == "task_type":
                intent.task_type = answer if answer in ["classification", "regression"] else None
            elif question.question_id == "priority_metric":
                intent.priority_metric = answer
            elif question.question_id == "business_context":
                intent.business_context = answer
            
            intent.answers.append(
                QuestionResponse(question_id=question.question_id, answer=answer)
            )
        
        return intent
    
    def _generate_answer(
        self,
        question,
        profile: Dict,
        correct_target: str,
        correct_task: str
    ) -> str:
        """Generate answer based on user type (may include mistakes)."""
        if self.user_type == "expert":
            return self._expert_answer(question, correct_target, correct_task)
        elif self.user_type == "intermediate":
            return self._intermediate_answer(question, profile, correct_target, correct_task)
        else:  # novice
            return self._novice_answer(question, profile, correct_target, correct_task)
    
    def _expert_answer(self, question, correct_target: str, correct_task: str) -> str:
        """Expert always answers correctly."""
        if question.question_id == "target_column":
            return correct_target
        elif question.question_id == "task_type":
            return correct_task
        elif question.question_id == "priority_metric":
            return "accuracy" if correct_task == "classification" else "r2"
        else:
            return "Expert use case"
    
    def _intermediate_answer(
        self,
        question,
        profile: Dict,
        correct_target: str,
        correct_task: str
    ) -> str:
        """Intermediate user makes occasional mistakes."""
        import random
        
        # 80% chance of correct answer
        if random.random() < 0.8:
            return self._expert_answer(question, correct_target, correct_task)
        
        # 20% chance of mistake
        if question.question_id == "target_column":
            # Pick wrong column
            candidates = profile.get("candidate_targets", [])
            wrong_targets = [c for c in candidates if c != correct_target]
            if wrong_targets:
                self.mistakes_made.append(f"Wrong target: {wrong_targets[0]}")
                return wrong_targets[0]
            return correct_target
        
        elif question.question_id == "task_type":
            # Wrong task type
            wrong_task = "regression" if correct_task == "classification" else "classification"
            self.mistakes_made.append(f"Wrong task: {wrong_task}")
            return wrong_task
        
        return self._expert_answer(question, correct_target, correct_task)
    
    def _novice_answer(
        self,
        question,
        profile: Dict,
        correct_target: str,
        correct_task: str
    ) -> str:
        """Novice user makes more mistakes and may provide incomplete info."""
        import random
        
        # 60% chance of correct answer
        if random.random() < 0.6:
            return self._expert_answer(question, correct_target, correct_task)
        
        # 40% chance of mistake or incomplete answer
        if question.question_id == "target_column":
            # May pick wrong column or provide incomplete name
            candidates = profile.get("candidate_targets", [])
            wrong_targets = [c for c in candidates if c != correct_target]
            if wrong_targets and random.random() < 0.7:
                self.mistakes_made.append(f"Wrong target: {wrong_targets[0]}")
                return wrong_targets[0]
            # Sometimes provide partial name
            if random.random() < 0.3:
                partial = correct_target[:len(correct_target)//2]
                self.mistakes_made.append(f"Partial target: {partial}")
                return partial
            return correct_target
        
        elif question.question_id == "task_type":
            # May pick wrong task or be unsure
            if random.random() < 0.6:
                wrong_task = "regression" if correct_task == "classification" else "classification"
                self.mistakes_made.append(f"Wrong task: {wrong_task}")
                return wrong_task
            else:
                # Unsure - return vague answer
                self.mistakes_made.append("Vague task answer")
                return "not sure"
        
        elif question.question_id == "priority_metric":
            # May pick wrong metric for task
            if correct_task == "classification":
                wrong_metric = "rmse"  # Regression metric
                self.mistakes_made.append(f"Wrong metric: {wrong_metric}")
                return wrong_metric
            else:
                wrong_metric = "accuracy"  # Classification metric
                self.mistakes_made.append(f"Wrong metric: {wrong_metric}")
                return wrong_metric
        
        return self._expert_answer(question, correct_target, correct_task)


def simulate_user_interaction(
    df: pd.DataFrame,
    correct_target: str,
    correct_task: str,
    user_type: str = "novice"
) -> Dict:
    """
    Simulate a complete user interaction.
    
    Returns:
        Dictionary with:
        - questions_asked: Number of questions
        - mistakes_detected: List of mistakes
        - config_generated: PipelineConfig
        - success: Whether pipeline was successfully created
        - metrics: Training metrics if successful
    """
    # Profile dataset
    profile = profile_dataset(df)
    
    # Generate questions
    questions = generate_questions(profile)
    
    # Simulate user
    user = SimulatedUser(user_type=user_type)
    user_intent = user.answer_questions(questions, profile, correct_target, correct_task)
    
    # Plan pipeline
    try:
        config = plan_pipeline(profile, user_intent, llm_provider="openai")
        success = True
    except Exception as e:
        config = None
        success = False
        print(f"Pipeline planning failed: {e}")
    
    # Try to train if config is valid
    metrics = None
    if success and config:
        try:
            if config.task == "classification":
                result = train_classification(df, config.target, config.metric, {"task": config.task, "preprocessing": config.preprocessing, "model": config.model_candidates[0]})
            else:
                result = train_regression(df, config.target, config.metric, {"task": config.task, "preprocessing": config.preprocessing, "model": config.model_candidates[0]})
            metrics = result["metrics"]
        except Exception as e:
            print(f"Training failed: {e}")
    
    return {
        "questions_asked": len(questions),
        "mistakes_made": user.mistakes_made,
        "config_generated": config.dict() if config else None,
        "success": success,
        "metrics": metrics,
        "user_type": user_type
    }


def run_simulation_experiment(
    datasets: List[Tuple[pd.DataFrame, str, str]],
    user_types: List[str] = ["novice", "intermediate", "expert"],
    n_runs: int = 10
) -> pd.DataFrame:
    """
    Run simulation experiment across multiple datasets and user types.
    
    Args:
        datasets: List of (DataFrame, correct_target, correct_task) tuples
        user_types: List of user types to simulate
        n_runs: Number of runs per dataset/user_type combination
        
    Returns:
        DataFrame with experiment results
    """
    results = []
    
    for df, correct_target, correct_task in datasets:
        for user_type in user_types:
            for run in range(n_runs):
                result = simulate_user_interaction(df, correct_target, correct_task, user_type)
                result["dataset"] = f"dataset_{len(results)}"
                result["run"] = run
                results.append(result)
    
    return pd.DataFrame(results)
