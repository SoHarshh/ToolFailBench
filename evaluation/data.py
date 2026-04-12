"""
Shared task loading utilities for ToolFailBench runners.
"""
import json
from pathlib import Path

TASK_DIR = Path(__file__).parent.parent / "tasks"
ALL_DOMAINS = ["finance", "medical", "code", "chemistry", "eda", "geospatial", "legal", "cybersecurity", "nutrition"]


def load_tasks(domains: list[str] = ALL_DOMAINS) -> list[dict]:
    """Load tasks from one or more domain JSON files."""
    tasks = []
    for domain in domains:
        task_file = TASK_DIR / domain / "tasks.json"
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")
        with open(task_file) as f:
            domain_tasks = json.load(f)
        tasks.extend(domain_tasks)
        print(f"  Loaded {len(domain_tasks)} tasks from {domain}")
    return tasks
