"""
Logging utilities for Intent2Model.

Handles run logging and experiment tracking.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid


def get_runs_dir() -> Path:
    """Get the runs storage directory."""
    runs_dir = Path(__file__).parent.parent / "storage" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def create_run_id() -> str:
    """Generate a unique run ID."""
    return str(uuid.uuid4())


def log_run(
    run_id: str,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Log a run to storage.
    
    Args:
        run_id: Unique run identifier
        data: Run data (metrics, model info, etc.)
        metadata: Optional metadata (timestamp, user info, etc.)
        
    Returns:
        Path to the saved run file
    """
    runs_dir = get_runs_dir()
    run_file = runs_dir / f"{run_id}.json"
    
    run_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "data": data,
        "metadata": metadata or {}
    }
    
    with open(run_file, "w") as f:
        json.dump(run_data, f, indent=2, default=str)
    
    return run_file


def load_run(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a run from storage.
    
    Args:
        run_id: Run identifier
        
    Returns:
        Run data dictionary or None if not found
    """
    runs_dir = get_runs_dir()
    run_file = runs_dir / f"{run_id}.json"
    
    if not run_file.exists():
        return None
    
    with open(run_file, "r") as f:
        return json.load(f)


def list_runs() -> List[str]:
    """
    List all run IDs.
    
    Returns:
        List of run IDs
    """
    runs_dir = get_runs_dir()
    return [f.stem for f in runs_dir.glob("*.json")]
