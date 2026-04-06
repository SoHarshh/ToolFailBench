"""
Main evaluation runner for ToolFailBench.
Loads tasks, runs models from the registry, logs to Weights & Biases.

Usage:
  python runners/run_eval.py --model qwen3.5-7b --domains finance medical code
  python runners/run_eval.py --tier 1 --domains finance medical code
  python runners/run_eval.py --tier 1 2 3 4
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

from evaluation.detect import classify_failure_mode
from evaluation.metrics import compute_all_metrics
from evaluation.report import generate_summary_table, save_results_json

# Allow importing models/registry.py from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.registry import load_registry, get_model_config, get_models_for_tier

load_dotenv()

ROOT = Path(__file__).parent.parent
TASK_DIR = ROOT / "tasks"
CONFIG_PATH = ROOT / "configs" / "default.yaml"
DOMAINS = ["finance", "medical", "code"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} is empty or invalid — expected a YAML mapping.")
    if "inference" not in config:
        raise ValueError(f"Config at {path} is missing required 'inference' section.")
    required_inference_keys = ["temperature", "max_tokens", "seed", "tool_choice"]
    missing = [k for k in required_inference_keys if k not in config["inference"]]
    if missing:
        raise ValueError(f"Config 'inference' section is missing keys: {missing}")
    null_keys = [k for k in required_inference_keys if config["inference"][k] is None]
    if null_keys:
        raise ValueError(f"Config 'inference' keys must not be null: {null_keys}")
    return config


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def load_tasks(domains: list[str]) -> list[dict]:
    tasks = []
    for domain in domains:
        task_file = TASK_DIR / domain / "tasks.json"
        if task_file.exists():
            with open(task_file) as f:
                domain_tasks = json.load(f)
                tasks.extend(domain_tasks)
                print(f"  Loaded {len(domain_tasks)} tasks from {domain}")
    return tasks


# ---------------------------------------------------------------------------
# Inference dispatch
# ---------------------------------------------------------------------------

def _build_litellm_model_str(model_cfg: dict, config: dict) -> tuple[str, dict]:
    """
    Returns (litellm_model_string, extra_kwargs) for the given model config.
    vLLM models are called via the local OpenAI-compatible server.
    Closed models go directly through their provider API.
    """
    extra = {}

    if model_cfg["inference_backend"] == "vllm":
        base_url = (
            config.get("vllm", {}).get("base_url")
            or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        )
        # litellm uses "openai/<model_name>" for custom OpenAI-compatible endpoints
        litellm_model = f"openai/{model_cfg['hf_model_id']}"
        extra["api_base"] = base_url
        extra["api_key"] = "vllm"  # vLLM server doesn't check the key but litellm requires it

    elif model_cfg["family"] == "openai":
        litellm_model = "gpt-4o"

    elif model_cfg["family"] == "anthropic":
        litellm_model = "claude-sonnet-4-5"

    else:
        raise ValueError(f"Unknown inference backend for model: {model_cfg['id']}")

    return litellm_model, extra


def run_single_task(task: dict, model_cfg: dict, config: dict) -> dict:
    """Run one task against a model. Returns result dict."""
    try:
        import litellm

        litellm_model, extra_kwargs = _build_litellm_model_str(model_cfg, config)
        inf = config.get("inference", {})

        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool["parameters"],
                    },
                },
            }
            for tool in task["available_tools"]
        ]

        system_prompt = task["system_prompt"]
        if model_cfg.get("no_think"):
            system_prompt = system_prompt + " /no_think"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task["user_message"]},
        ]

        call_kwargs = dict(
            model=litellm_model,
            messages=messages,
            tools=tools,
            tool_choice=inf["tool_choice"],
            temperature=inf["temperature"],
            max_tokens=inf["max_tokens"],
            seed=inf["seed"],
            **extra_kwargs,
        )

        response = litellm.completion(**call_kwargs)

        tool_calls = []
        agent_answer = ""
        choice = response.choices[0]

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                )
            # Inject mock tool return and get final answer
            tool_messages = messages + [choice.message]
            for tc in choice.message.tool_calls:
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(task["mock_tool_return"]),
                    }
                )
            follow_up = litellm.completion(
                model=litellm_model,
                messages=tool_messages,
                tools=tools,
                temperature=inf["temperature"],
                max_tokens=inf["max_tokens"],
                seed=inf["seed"],
                **extra_kwargs,
            )
            agent_answer = follow_up.choices[0].message.content or ""
        else:
            agent_answer = choice.message.content or ""

        agent_trace = {"tool_calls": tool_calls}
        classification = classify_failure_mode(task, agent_trace, agent_answer)

        return {
            "task": task,
            "model_id": model_cfg["id"],
            "agent_trace": agent_trace,
            "agent_answer": agent_answer,
            "classification": classification,
        }

    except Exception as e:
        return {
            "task": task,
            "model_id": model_cfg["id"],
            "agent_trace": {"tool_calls": []},
            "agent_answer": f"ERROR: {str(e)}",
            "classification": "other_error",
        }


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def init_wandb(model_cfg: dict, config: dict):
    import wandb

    wb_cfg = config.get("wandb", {})
    run = wandb.init(
        project=wb_cfg.get("project") or os.getenv("WANDB_PROJECT", "toolfailbench"),
        entity=wb_cfg.get("entity") or os.getenv("WANDB_ENTITY") or None,
        name=f"{model_cfg['id']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_id": model_cfg["id"],
            "family": model_cfg["family"],
            "size": model_cfg["size"],
            "tier": model_cfg["tier"],
            "category": model_cfg["category"],
            "inference_backend": model_cfg["inference_backend"],
            **config.get("inference", {}),
        },
    )
    return run


def log_to_wandb(run, results: list[dict], config: dict):
    import wandb

    wb_cfg = config.get("wandb", {})

    if wb_cfg.get("log_metrics", True):
        metrics = compute_all_metrics(results)
        run.log({
            "tsr": metrics["tsr"],
            "rir": metrics["rir"],
            "ofr": metrics["ofr"],
            "ctur": metrics["ctur"],
            "total_tasks": metrics["total_tasks"],
        })

    if wb_cfg.get("log_predictions", True):
        table = wandb.Table(columns=[
            "task_id", "domain", "target_mode", "conflict_type",
            "classification", "agent_answer", "model_id",
        ])
        for r in results:
            task = r["task"]
            table.add_data(
                task["task_id"],
                task["domain"],
                task["target_failure_mode"],
                task["conflict_type"],
                r["classification"],
                r["agent_answer"][:500],  # truncate for display
                r["model_id"],
            )
        run.log({"predictions": table})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_model(model_cfg: dict, tasks: list[dict], config: dict, output_dir: str, use_wandb: bool):
    print(f"\n{'='*60}")
    print(f"  Model: {model_cfg['id']} (Tier {model_cfg['tier']}, {model_cfg['category']})")
    print(f"{'='*60}")

    wb_run = None
    if use_wandb:
        try:
            wb_run = init_wandb(model_cfg, config)
        except Exception as e:
            print(f"  [W&B] Failed to init: {e}. Continuing without logging.")

    results = []
    for task in tqdm(tasks, desc=model_cfg["id"]):
        result = run_single_task(task, model_cfg, config)
        results.append(result)

    print("\n" + generate_summary_table(results, model_cfg["id"]))

    if wb_run:
        try:
            log_to_wandb(wb_run, results, config)
            wb_run.finish()
            print(f"  [W&B] Logged to {wb_run.url}")
        except Exception as e:
            print(f"  [W&B] Logging failed: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{output_dir}/{model_cfg['id']}_{timestamp}.json"
    save_results_json(results, out_path)
    print(f"  Results saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ToolFailBench evaluation")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model id from registry (e.g. qwen2.5-7b)")
    group.add_argument("--tier", nargs="+", type=int, help="Run all models in tier(s) (e.g. --tier 1 2)")

    parser.add_argument("--domains", nargs="+", default=DOMAINS)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to YAML config")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    # Resolve which models to run
    if args.model:
        models_to_run = [get_model_config(args.model)]
    else:
        models_to_run = []
        for tier in args.tier:
            tier_models = get_models_for_tier(tier)
            if not tier_models:
                print(f"  Warning: no models found for tier {tier}")
            models_to_run.extend(tier_models)

    print(f"Models to run: {[m['id'] for m in models_to_run]}")
    print(f"Loading tasks for domains: {args.domains}")
    tasks = load_tasks(args.domains)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    print(f"Total tasks: {len(tasks)}")

    use_wandb = not args.no_wandb
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model_cfg in models_to_run:
        results = run_model(model_cfg, tasks, config, args.output_dir, use_wandb)
        all_results[model_cfg["id"]] = results

    if len(models_to_run) > 1:
        print(f"\n{'='*60}")
        print(f"  Completed {len(models_to_run)} models")
        for model_id, results in all_results.items():
            metrics = compute_all_metrics(results)
            print(f"  {model_id}: CTUR={metrics['ctur']:.2%} TSR={metrics['tsr']:.2%} RIR={metrics['rir']:.2%} OFR={metrics['ofr']:.2%}")


if __name__ == "__main__":
    main()
