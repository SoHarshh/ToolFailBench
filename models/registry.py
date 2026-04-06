"""
Model registry loader for ToolFailBench.
Reads all per-model YAML files from models/configs/{family}/*.yaml
"""
import yaml
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent / "configs"

REQUIRED_FIELDS = ["id", "hf_model_id", "family", "size", "tier", "category", "inference_backend"]


def _load_model_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid model config at {path} — expected a YAML mapping.")
    missing = [f for f in REQUIRED_FIELDS if f not in cfg]
    if missing:
        raise ValueError(f"Model config {path.name} is missing required fields: {missing}")
    return cfg


def load_registry() -> list[dict]:
    """Load and return all model configs, sorted by tier then id."""
    configs = []
    for yaml_file in sorted(CONFIGS_DIR.rglob("*.yaml")):
        configs.append(_load_model_config(yaml_file))
    if not configs:
        raise RuntimeError(f"No model configs found in {CONFIGS_DIR}")
    return sorted(configs, key=lambda m: (m["tier"], m["id"]))


def get_model_config(model_id: str) -> dict:
    for m in load_registry():
        if m["id"] == model_id:
            return m
    raise ValueError(f"Model '{model_id}' not found in registry. Check models/configs/.")


def get_models_for_tier(tier: int) -> list[dict]:
    return [m for m in load_registry() if m["tier"] == tier]
