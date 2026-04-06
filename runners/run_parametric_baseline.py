"""
Parametric baseline runner for ToolFailBench.
Runs models WITHOUT tool access to record pure parametric memory answers.
These baselines let us measure how strongly parametric priors conflict with
mock tool returns — the core driver of Result-Ignore failures.

Usage:
  python runners/run_parametric_baseline.py --model qwen3.5-7b
  python runners/run_parametric_baseline.py --tier 1
"""
import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.registry import load_registry, get_model_config, get_models_for_tier
from evaluation.data import load_tasks, ALL_DOMAINS

load_dotenv()

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "default.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} is empty or invalid.")
    if "inference" not in config:
        raise ValueError(f"Config at {path} is missing required 'inference' section.")
    required = ["temperature", "max_tokens", "seed", "tool_choice"]
    missing = [k for k in required if k not in config["inference"]]
    if missing:
        raise ValueError(f"Config 'inference' section missing keys: {missing}")
    null_keys = [k for k in required if config["inference"][k] is None]
    if null_keys:
        raise ValueError(f"Config 'inference' keys must not be null: {null_keys}")
    return config


def _build_litellm_model_str(model_cfg: dict, config: dict) -> tuple[str, dict]:
    extra = {}
    if model_cfg["inference_backend"] == "vllm":
        base_url = (
            config.get("vllm", {}).get("base_url")
            or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        )
        extra["api_base"] = base_url
        extra["api_key"] = "vllm"
        return f"openai/{model_cfg['hf_model_id']}", extra
    elif model_cfg["family"] == "openai":
        return "gpt-4o", extra
    elif model_cfg["family"] == "anthropic":
        return "claude-sonnet-4-5", extra
    else:
        raise ValueError(f"Unknown inference backend for model: {model_cfg['id']}")


def get_parametric_answer(task: dict, model_cfg: dict, config: dict) -> str:
    """Query model WITHOUT tools to capture its parametric memory answer."""
    try:
        import litellm

        litellm_model, extra_kwargs = _build_litellm_model_str(model_cfg, config)
        inf = config["inference"]

        system_prompt = task["system_prompt"]
        if model_cfg.get("no_think"):
            system_prompt = system_prompt + " /no_think"

        response = litellm.completion(
            model=litellm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task["user_message"]},
            ],
            temperature=inf["temperature"],
            max_tokens=inf["max_tokens"],
            seed=inf["seed"],
            # No tools= parameter — pure parametric memory
            **extra_kwargs,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {str(e)}"


def run_baseline_for_model(model_cfg: dict, tasks: list[dict], config: dict, output_dir: str, use_wandb: bool):
    print(f"\n{'='*60}")
    print(f"  Parametric Baseline: {model_cfg['id']} (Tier {model_cfg['tier']})")
    print(f"{'='*60}")

    wb_run = None
    if use_wandb:
        try:
            import wandb
            wb_cfg = config.get("wandb", {})
            wb_run = wandb.init(
                project=wb_cfg.get("project") or os.getenv("WANDB_PROJECT", "toolfailbench"),
                entity=wb_cfg.get("entity") or os.getenv("WANDB_ENTITY") or None,
                name=f"baseline-{model_cfg['id']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "run_type": "parametric_baseline",
                    "model_id": model_cfg["id"],
                    "family": model_cfg["family"],
                    "tier": model_cfg["tier"],
                    **config["inference"],
                },
            )
        except Exception as e:
            print(f"  [W&B] Failed to init: {e}. Continuing without logging.")

    baselines = []
    for task in tqdm(tasks, desc=model_cfg["id"]):
        answer = get_parametric_answer(task, model_cfg, config)
        baselines.append({
            "task_id": task["task_id"],
            "domain": task["domain"],
            "target_failure_mode": task["target_failure_mode"],
            "conflict_type": task["conflict_type"],
            "parametric_answer": answer,
            "model_id": model_cfg["id"],
        })

    if wb_run:
        try:
            import wandb
            table = wandb.Table(columns=["task_id", "domain", "target_failure_mode", "conflict_type", "parametric_answer", "model_id"])
            for b in baselines:
                table.add_data(b["task_id"], b["domain"], b["target_failure_mode"], b["conflict_type"], b["parametric_answer"][:500], b["model_id"])
            wb_run.log({"parametric_baselines": table})
            wb_run.finish()
            print(f"  [W&B] Logged to {wb_run.url}")
        except Exception as e:
            print(f"  [W&B] Logging failed: {e}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{output_dir}/{model_cfg['id']}_baseline_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"  Baselines saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect parametric baselines for ToolFailBench")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model id from registry (e.g. qwen3.5-7b)")
    group.add_argument("--tier", nargs="+", type=int, help="Run all models in tier(s)")

    parser.add_argument("--domains", nargs="+", default=ALL_DOMAINS)
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.model:
        models_to_run = [get_model_config(args.model)]
    else:
        models_to_run = []
        for tier in args.tier:
            tier_models = get_models_for_tier(tier)
            if not tier_models:
                print(f"  Warning: no models found for tier {tier}")
            models_to_run.extend(tier_models)

    print(f"Models: {[m['id'] for m in models_to_run]}")
    tasks = load_tasks(args.domains)
    print(f"Total tasks: {len(tasks)}")

    for model_cfg in models_to_run:
        run_baseline_for_model(model_cfg, tasks, config, args.output_dir, not args.no_wandb)


if __name__ == "__main__":
    main()
